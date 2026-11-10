#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把单个 MatMul 的 ONNX 模型转换成 Triton 的中间表示（TTIR）。

这个程序是一个**可运行的转换器 demo**，不是展示页，也不是一次性报告。
它完成下面这条完整流程：

    读取 .onnx 文件
      → 检查模型、取出矩阵形状
      → 计算分块大小和启动网格
      → 生成 Triton Python 源码
      → 调用 Triton 前端
      → 写出 .ttir 文件

支持范围（故意做得很小）：
    - 只支持一个算子：MatMul
    - 只支持二维矩阵、静态形状
    - 只支持 f32（32 位浮点数）
    - 只输出 TTIR，不启动 GPU，不生成 cubin

用法：
    python demo/onnx_to_ttir.py 输入.onnx -o 输出.ttir
    python demo/onnx_to_ttir.py 输入.onnx -o 输出.ttir --verbose
    python demo/onnx_to_ttir.py 输入.onnx -o 输出.ttir --emit-intermediates
"""

from __future__ import annotations

# 不生成 __pycache__，避免 demo 目录里出现缓存文件。
import sys

sys.dont_write_bytecode = True

# 让 Triton 输出 TTIR 时不带“源码第几行”的注释，文本更干净。
# 必须在 import triton 之前设置。
import os

os.environ["TRITON_DISABLE_LINE_INFO"] = "1"

import argparse
import importlib.util
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend


# =============================================================================
# 一、数据结构
# =============================================================================


@dataclass
class TensorInfo:
    """描述一个张量：名字、元素类型、形状、是不是模型里的常量。

    is_initializer 表示这个张量是不是 ONNX 里的 initializer（常量权重）。
    这个 demo 不处理权重内容，只关心它的形状和类型。
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    is_initializer: bool = False


@dataclass
class MatMulGraph:
    """从 ONNX 模型里读出来的最小信息。

    这里只保留一个 MatMul 所需的三个张量，以及 opset 版本。
    真实项目里会有一套完整的图结构；这个 demo 刻意做最小。
    """

    a: TensorInfo
    b: TensorInfo
    c: TensorInfo
    opset: int | None


@dataclass
class KernelPlan:
    """一个 kernel 的执行计划。

    它回答：
      - 矩阵有多大
      - 每个分块多大
      - 启动多少个 program
      - kernel 每个参数是什么类型
      - 哪些参数是编译期常量
    """

    name: str
    m: int
    k: int
    n: int
    block_m: int
    block_n: int
    block_k: int
    grid: tuple[int, int, int]
    signature: dict[str, str]
    constexprs: dict[str, int]
    num_warps: int
    num_stages: int


# ONNX 的数据类型编号 -> 我们使用的短名字。
# 这个 demo 只接受 f32；其他类型会给出明确错误。
ONNX_DTYPE_TO_NAME = {
    TensorProto.FLOAT: "f32",
    TensorProto.DOUBLE: "f64",
    TensorProto.FLOAT16: "f16",
    TensorProto.BFLOAT16: "bf16",
    TensorProto.INT8: "i8",
    TensorProto.INT16: "i16",
    TensorProto.INT32: "i32",
    TensorProto.INT64: "i64",
    TensorProto.UINT8: "u8",
    TensorProto.UINT16: "u16",
    TensorProto.UINT32: "u32",
    TensorProto.UINT64: "u64",
    TensorProto.BOOL: "i1",
}

# Triton 的参数类型字符串。
# 注意：Triton 用的是 "fp32"，不是 "f32"；写错会报 KeyError。
TRITON_POINTER_TYPE = "*fp32"


# =============================================================================
# 二、读取并检查 ONNX 模型
# =============================================================================


def tensor_info_from_value_info(value_info: Any) -> TensorInfo:
    """从 ONNX 的 ValueInfoProto 里取出名字、类型和形状。

    这个 demo 只接受静态形状：每个维度都必须是一个确定的数字。
    如果遇到动态维度（例如 batch），会直接报错，并说明原因。
    """

    tensor_type = value_info.type.tensor_type
    if tensor_type.elem_type not in ONNX_DTYPE_TO_NAME:
        raise ValueError(
            f"张量 {value_info.name} 的数据类型编号是 {tensor_type.elem_type}，"
            "这个 demo 暂不支持。"
        )

    dtype = ONNX_DTYPE_TO_NAME[tensor_type.elem_type]

    shape: list[int] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            raise ValueError(
                f"张量 {value_info.name} 有一个动态维度 '{dim.dim_param}'。"
                "这个 demo 只支持静态形状；请先把模型的形状固定下来。"
            )
        else:
            raise ValueError(
                f"张量 {value_info.name} 有一个未知维度。"
                "这个 demo 只支持静态形状。"
            )

    return TensorInfo(
        name=value_info.name,
        dtype=dtype,
        shape=tuple(shape),
    )


def load_and_validate_model(model_path: Path) -> MatMulGraph:
    """读取 ONNX 模型，检查它是不是“单个二维 MatMul”。

    检查内容：
      1. 文件能正常读取
      2. ONNX 官方检查器认为模型合法
      3. 图里只有一个节点，且这个节点是 MatMul
      4. 两个输入都是二维、静态、f32
      5. 形状能对上：A[M,K] @ B[K,N] = C[M,N]
    """

    model = onnx.load(str(model_path))

    # ONNX 官方检查器：检查图结构、类型、属性等是否合法。
    onnx.checker.check_model(model)

    # 让 ONNX 自带的形状推断跑一遍，补全中间结果的形状。
    # 如果推断失败，就继续使用原模型；这个 demo 也能自己算 MatMul 的输出形状。
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    graph = model.graph
    nodes = list(graph.node)

    if len(nodes) != 1:
        raise ValueError(
            f"这个 demo 只支持一个算子的模型，但模型里有 {len(nodes)} 个节点。"
        )

    node = nodes[0]
    if node.op_type != "MatMul":
        raise ValueError(
            f"这个 demo 只支持 MatMul，但模型里的算子是 {node.op_type}。"
        )

    node_inputs = [name for name in node.input if name]
    node_outputs = [name for name in node.output if name]

    if len(node_inputs) != 2 or len(node_outputs) != 1:
        raise ValueError(
            "MatMul 应该有两个输入、一个输出，但模型里的数量对不上。"
        )

    # 收集“名字 -> 张量信息”。
    info_map: dict[str, TensorInfo] = {}

    for value_info in graph.input:
        info_map[value_info.name] = tensor_info_from_value_info(value_info)
    for value_info in graph.output:
        info_map[value_info.name] = tensor_info_from_value_info(value_info)
    for value_info in graph.value_info:
        info_map[value_info.name] = tensor_info_from_value_info(value_info)

    # initializer（常量权重）没有 ValueInfoProto，需要从 TensorProto 里取。
    for initializer in graph.initializer:
        if initializer.data_type not in ONNX_DTYPE_TO_NAME:
            raise ValueError(
                f"常量 {initializer.name} 的数据类型编号是 {initializer.data_type}，"
                "这个 demo 暂不支持。"
            )
        info_map[initializer.name] = TensorInfo(
            name=initializer.name,
            dtype=ONNX_DTYPE_TO_NAME[initializer.data_type],
            shape=tuple(initializer.dims),
            is_initializer=True,
        )

    a_name, b_name = node_inputs
    c_name = node_outputs[0]

    if a_name not in info_map or b_name not in info_map:
        raise ValueError(
            f"找不到输入 {a_name} 或 {b_name} 的形状信息。"
            "这个 demo 需要模型里带有静态形状。"
        )

    a = info_map[a_name]
    b = info_map[b_name]

    # 这个 demo 只支持二维矩阵乘，也就是普通 MatMul，不支持批量 MatMul。
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError(
            f"这个 demo 只支持二维 MatMul，但 A 的形状是 {a.shape}，"
            f"B 的形状是 {b.shape}。"
        )

    if a.dtype != "f32" or b.dtype != "f32":
        raise ValueError(
            f"这个 demo 只支持 f32，但 A 的类型是 {a.dtype}，B 的类型是 {b.dtype}。"
        )

    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(
            f"形状对不上：A 的第二维是 {k}，B 的第一维是 {k2}，无法相乘。"
        )

    # 输出 C 的期望形状。
    expected_c_shape = (m, n)

    if c_name in info_map:
        c = info_map[c_name]
        if c.shape != expected_c_shape:
            raise ValueError(
                f"输出 C 的形状应该是 {expected_c_shape}，但模型里写的是 {c.shape}。"
            )
        if c.dtype != "f32":
            raise ValueError(f"这个 demo 只支持 f32，但输出 C 的类型是 {c.dtype}。")
    else:
        # 模型没有给输出形状时，根据 MatMul 的规则自己算。
        c = TensorInfo(name=c_name, dtype="f32", shape=expected_c_shape)

    # 取默认域的 opset 版本，仅用于打印。
    opset = None
    for item in model.opset_import:
        if item.domain in ("", "ai.onnx"):
            opset = item.version
            break

    return MatMulGraph(a=a, b=b, c=c, opset=opset)


# =============================================================================
# 三、计算 kernel 执行计划
# =============================================================================


def is_power_of_two(value: int) -> bool:
    """判断一个整数是不是 2 的幂（1、2、4、8、16……）。"""

    return value > 0 and (value & (value - 1)) == 0


def build_kernel_plan(
    graph: MatMulGraph,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> KernelPlan:
    """根据矩阵形状，计算分块、启动网格和 kernel 参数类型。"""

    m, k = graph.a.shape
    n = graph.b.shape[1]

    # 检查分块大小是否符合 Triton 的要求。
    for name, value in (
        ("BLOCK_M", block_m),
        ("BLOCK_N", block_n),
        ("BLOCK_K", block_k),
    ):
        if not is_power_of_two(value):
            raise ValueError(
                f"{name} 必须是 2 的幂（例如 16、32、64、128），当前是 {value}。"
            )

    if block_m < 16 or block_n < 16 or block_k < 16:
        raise ValueError(
            "tl.dot 要求分块的每一维至少为 16；"
            f"当前是 BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}。"
        )

    if block_m * block_n > 1_048_576:
        raise ValueError(
            f"单个分块最多 1,048,576 个元素，当前是 {block_m * block_n}。"
        )

    if not is_power_of_two(num_warps):
        raise ValueError(f"num_warps 必须是 2 的幂，当前是 {num_warps}。")
    if num_stages < 1:
        raise ValueError(f"num_stages 必须大于等于 1，当前是 {num_stages}。")

    # 启动网格：行方向、列方向各需要多少个 program。
    # 用“向上取整”的除法，因为最后一块可能不完整。
    grid = (
        (m + block_m - 1) // block_m,
        (n + block_n - 1) // block_n,
        1,
    )

    # 行主序（row-major）连续存储时的步长。
    # A 是 [M, K]，所以往下走一行加 K，往右走一列加 1。
    # B 是 [K, N]，所以往下走一行加 N，往右走一列加 1。
    # C 是 [M, N]，所以往下走一行加 N，往右走一列加 1。
    signature = {
        "a_ptr": TRITON_POINTER_TYPE,
        "b_ptr": TRITON_POINTER_TYPE,
        "c_ptr": TRITON_POINTER_TYPE,
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
    }

    constexprs = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
    }

    return KernelPlan(
        name="matmul_kernel",
        m=m,
        k=k,
        n=n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        grid=grid,
        signature=signature,
        constexprs=constexprs,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# =============================================================================
# 四、生成 Triton Python 源码
# =============================================================================


def render_kernel_source(plan: KernelPlan) -> str:
    """根据执行计划，生成一个 Triton kernel 的 Python 源码。

    注意：生成的是普通的 Python 代码，不是 TTIR 文本。
    TTIR 由 Triton 自己的前端在后面生成。
    """

    source = f'''# -*- coding: utf-8 -*-
# 本文件由 demo/onnx_to_ttir.py 自动生成。
# 它把 C = A @ B 写成一个 Triton kernel。

import triton
import triton.language as tl


@triton.jit
def {plan.name}(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """计算 C = A @ B。

    a_ptr, b_ptr, c_ptr: A、B、C 三块显存的起始地址
    M, N, K:            矩阵的真实大小
    stride_xx:          移动一行或一列时，地址要加多少
    BLOCK_M/N/K:        每个分块的大小（编译期常量）
    """

    # 当前 program 负责输出矩阵的哪一块。
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 这一块在真实矩阵里的行号和列号。
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 沿 K 方向的下标。
    offs_k = tl.arange(0, BLOCK_K)

    # 构造 A、B 的地址网格。
    # 以 A 为例：第 (m, k) 个元素的地址 = a_ptr + m * stride_am + k * stride_ak。
    # offs_m[:, None] 把行下标变成一列，offs_k[None, :] 把 K 下标变成一行，
    # 两者相加就得到一个二维的地址网格。
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # 累加器：保存当前输出块的部分结果，初始全部为 0。
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 沿 K 方向一块一块地累加；K 不一定能被 BLOCK_K 整除，所以向上取整。
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K

        # 边界遮罩：挡住超出矩阵范围的位置。
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        # 越界的位置补 0，这样不会影响结果。
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 矩阵乘加：acc = a @ b + acc。
        # input_precision="ieee" 表示按标准浮点规则计算，结果更容易和参考实现对上。
        acc = tl.dot(a, b, acc, input_precision="ieee")

        # 移动到下一个 K 块。
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 把结果写回 C；同样需要边界遮罩。
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
'''
    return source


def write_kernel_source(source: str, kernel_path: Path) -> None:
    """先做语法检查，再把源码写到真实文件。"""

    # compile() 只检查 Python 语法，不执行代码。
    compile(source, str(kernel_path), "exec")
    kernel_path.write_text(source, encoding="utf-8")


# =============================================================================
# 五、把 Triton Python 源码变成 TTIR
# =============================================================================


def load_generated_kernel(kernel_path: Path, function_name: str):
    """从生成的 .py 文件里加载 kernel 函数。

    为什么不能直接执行一段字符串？
    因为 Triton 的 @triton.jit 要通过 inspect 读取函数源码；
    动态 exec 出来的函数没有源码文件，Triton 会报错。
    所以必须先写到真实的 .py 文件，再 import。
    """

    module_name = f"generated_{kernel_path.stem}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载生成的 kernel：{kernel_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def extract_ttir(
    kernel_path: Path,
    plan: KernelPlan,
    ttir_path: Path,
    arch: int,
) -> dict[str, Any]:
    """调用 Triton 前端，把 Python kernel 变成 TTIR，并写出文件。

    这里只跑 Triton 前端：
        ASTSource.make_ir  →  backend.make_ttir
    不会调用 ptxas，也不会启动 GPU。
    """

    kernel_fn = load_generated_kernel(kernel_path, plan.name)

    # 目标 GPU：CUDA 后端，计算能力 arch（默认 70，也就是 V100）。
    target = GPUTarget("cuda", arch, 32)

    # backend 知道怎么把 Triton 的中间表示一步步降低到 GPU 代码。
    backend = make_backend(target)

    # 编译选项：warp 数、流水线 stage 数。
    options = backend.parse_options(
        {"num_warps": plan.num_warps, "num_stages": plan.num_stages}
    )

    # 创建 MLIR 上下文，并注册 Triton 需要的方言（一组指令集合）。
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    # ASTSource 把“Python 函数 + 参数类型 + 编译期常量”打包起来。
    source = ASTSource(
        fn=kernel_fn,
        signature=plan.signature,
        constexprs=plan.constexprs,
    )

    # 第一步：Python 源码 -> TTIR 模块（内存里的对象）。
    module = source.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )

    # 第二步：跑 TTIR 阶段的规范化 pass（内联、常量折叠、CSE 等）。
    ttir_module = backend.make_ttir(module, {}, options, target.arch)

    # 转成文本并写入目标文件。
    ttir_text = str(ttir_module)
    ttir_path.write_text(ttir_text, encoding="utf-8")

    # 验证：把刚写出的文本重新解析一遍，再调用 verify()。
    # 这一步能确认 TTIR 文本本身是合法的。
    parsed = ir.parse_mlir_module(str(ttir_path), context)
    parsed.verify()

    return {
        "lines": ttir_text.count("\n") + 1,
        "chars": len(ttir_text),
        "verified": True,
    }


# =============================================================================
# 六、命令行入口
# =============================================================================


def graph_to_dict(graph: MatMulGraph) -> dict[str, Any]:
    """把 MatMulGraph 整理成 JSON 可写的形式。"""

    def one(info: TensorInfo) -> dict[str, Any]:
        return {
            "name": info.name,
            "dtype": info.dtype,
            "shape": list(info.shape),
            "is_initializer": info.is_initializer,
        }

    return {
        "op": "MatMul",
        "opset": graph.opset,
        "a": one(graph.a),
        "b": one(graph.b),
        "c": one(graph.c),
    }


def plan_to_dict(plan: KernelPlan) -> dict[str, Any]:
    """把 KernelPlan 整理成 JSON 可写的形式。"""

    return {
        "name": plan.name,
        "m": plan.m,
        "k": plan.k,
        "n": plan.n,
        "block_m": plan.block_m,
        "block_n": plan.block_n,
        "block_k": plan.block_k,
        "grid": list(plan.grid),
        "signature": plan.signature,
        "constexprs": plan.constexprs,
        "num_warps": plan.num_warps,
        "num_stages": plan.num_stages,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把一个只包含 MatMul 的 ONNX 模型转换成 TTIR。"
    )
    parser.add_argument("input", type=Path, help="输入的 .onnx 文件")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出的 .ttir 文件（默认：与输入同名，扩展名换成 .ttir）",
    )
    parser.add_argument(
        "--block-m", type=int, default=64, help="输出分块的行数，默认 64"
    )
    parser.add_argument(
        "--block-n", type=int, default=32, help="输出分块的列数，默认 32"
    )
    parser.add_argument(
        "--block-k", type=int, default=32, help="沿 K 方向的分块大小，默认 32"
    )
    parser.add_argument("--num-warps", type=int, default=4, help="warp 数，默认 4")
    parser.add_argument("--num-stages", type=int, default=3, help="流水线 stage 数，默认 3")
    parser.add_argument(
        "--arch",
        type=int,
        default=70,
        help="目标 GPU 的计算能力，默认 70（V100）",
    )
    parser.add_argument(
        "--emit-intermediates",
        action="store_true",
        help="把中间结果（图结构、生成的 Triton 源码）保存到输出文件旁边",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个阶段的详细信息",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        print(f"错误：找不到输入文件 {input_path}")
        return 1

    output_path: Path = args.output
    if output_path is None:
        output_path = input_path.with_suffix(".ttir")
    if output_path.is_dir():
        print(f"错误：输出路径是一个目录，请指定文件名：{output_path}")
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.time()

    try:
        # ---------- 第 1 步：读取并检查 ONNX ----------
        if args.verbose:
            print("[1/5] 读取并检查 ONNX 模型")
            print(f"  文件: {input_path}")

        graph = load_and_validate_model(input_path)

        if args.verbose:
            print(f"  算子: MatMul（opset {graph.opset}）")
            print(f"  A   : {graph.a.dtype} {list(graph.a.shape)}")
            print(f"  B   : {graph.b.dtype} {list(graph.b.shape)}")
            print(f"  C   : {graph.c.dtype} {list(graph.c.shape)}")

        # ---------- 第 2 步：计算执行计划 ----------
        if args.verbose:
            print("[2/5] 计算分块和启动网格")

        plan = build_kernel_plan(
            graph,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
        )

        if args.verbose:
            print(
                f"  分块: BLOCK_M={plan.block_m}, BLOCK_N={plan.block_n}, "
                f"BLOCK_K={plan.block_k}"
            )
            print(f"  网格: {plan.grid}（共 {plan.grid[0] * plan.grid[1]} 个 program）")

        # ---------- 第 3 步：生成 Triton Python 源码 ----------
        if args.verbose:
            print("[3/5] 生成 Triton Python 源码")

        # 如果用户要求保留中间结果，就把 kernel 源码写到输出文件旁边；
        # 否则写到一个临时目录，生成完 TTIR 后自动清理。
        if args.emit_intermediates:
            kernel_path = output_path.with_suffix(".kernel.py")
            graph_path = output_path.with_suffix(".graph.json")
            temp_dir = None
        else:
            temp_dir = tempfile.TemporaryDirectory()
            kernel_path = Path(temp_dir.name) / "matmul_kernel.py"
            graph_path = None

        try:
            kernel_source = render_kernel_source(plan)
            write_kernel_source(kernel_source, kernel_path)

            if args.verbose:
                print(f"  源码: {kernel_path}")

            if args.emit_intermediates:
                graph_path.write_text(
                    json.dumps(graph_to_dict(graph), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                plan_path = output_path.with_suffix(".plan.json")
                plan_path.write_text(
                    json.dumps(plan_to_dict(plan), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if args.verbose:
                    print(f"  图结构: {graph_path}")
                    print(f"  执行计划: {plan_path}")

            # ---------- 第 4 步：调用 Triton 前端生成 TTIR ----------
            if args.verbose:
                print("[4/5] 调用 Triton 前端生成 TTIR")

            ttir_info = extract_ttir(
                kernel_path=kernel_path,
                plan=plan,
                ttir_path=output_path,
                arch=args.arch,
            )

            # ---------- 第 5 步：报告结果 ----------
            if args.verbose:
                print("[5/5] 写出并检查 TTIR")
                print(f"  文件: {output_path}")
                print(f"  行数: {ttir_info['lines']}")
                print(f"  字符数: {ttir_info['chars']}")
                print("  合法性检查: 通过")
        finally:
            # 临时目录会在离开这里时自动删除。
            if temp_dir is not None:
                temp_dir.cleanup()

    except ValueError as error:
        print(f"错误：{error}")
        return 1

    print()
    print("转换完成")
    print(f"  输入模型 : {input_path}")
    print(f"  输出 TTIR: {output_path}")
    print(f"  MatMul   : A{list(graph.a.shape)} @ B{list(graph.b.shape)} -> C{list(graph.c.shape)}")
    print(
        f"  分块     : BLOCK_M={plan.block_m}, BLOCK_N={plan.block_n}, "
        f"BLOCK_K={plan.block_k}"
    )
    print(f"  网格     : {plan.grid}")
    print(f"  TTIR     : {ttir_info['lines']} 行，{ttir_info['chars']} 个字符，已通过合法性检查")
    print(f"  耗时     : {time.time() - started_at:.2f} 秒")
    if args.emit_intermediates:
        print(f"  中间文件 : {output_path.with_suffix('.graph.json')}、"
              f"{output_path.with_suffix('.plan.json')}、"
              f"{output_path.with_suffix('.kernel.py')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
