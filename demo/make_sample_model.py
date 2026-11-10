#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成一个只包含 MatMul 的 ONNX 模型。

这个脚本只做一件事：
    造一个最简单的矩阵乘法模型，保存成 .onnx 文件，
    方便测试 onnx_to_ttir.py。

模型的数学含义：
    C = A @ B

    A 的形状是 [M, K]
    B 的形状是 [K, N]
    C 的形状是 [M, N]

用法：
    python demo/make_sample_model.py
    python demo/make_sample_model.py -o /tmp/my_model.onnx --m 64 --k 32 --n 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

# 这个文件所在目录。默认输出放在它下面的 out/ 目录里。
SCRIPT_DIR = Path(__file__).resolve().parent


def build_matmul_model(m: int, k: int, n: int) -> onnx.ModelProto:
    """构造一个 C = A @ B 的 ONNX 模型。

    参数都是正数。这个函数不做复杂校验，调用方负责检查。
    """

    # 三个张量的说明：名字、元素类型、形状。
    # 这里全部使用 f32（32 位浮点数），和转换器 demo 的支持范围一致。
    a_info = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
    b_info = helper.make_tensor_value_info("B", TensorProto.FLOAT, [k, n])
    c_info = helper.make_tensor_value_info("C", TensorProto.FLOAT, [m, n])

    # 一个 MatMul 节点：输入 A、B，输出 C。
    node = helper.make_node(
        op_type="MatMul",
        inputs=["A", "B"],
        outputs=["C"],
        name="matmul_0",
    )

    # 一张图：节点、图输入、图输出。
    graph = helper.make_graph(
        nodes=[node],
        name="single_matmul_graph",
        inputs=[a_info, b_info],
        outputs=[c_info],
    )

    # 模型：把图包起来，声明使用 opset 17。
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )

    # 固定 IR 版本，避免不同 ONNX 版本的默认值不一致。
    model.ir_version = 10

    # 用官方检查器确认模型结构合法。
    onnx.checker.check_model(model)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成一个只包含 MatMul 的 ONNX 模型，用来测试 ONNX -> TTIR 转换器。"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=SCRIPT_DIR / "out" / "sample_matmul.onnx",
        help="输出的 .onnx 文件路径（默认：demo/out/sample_matmul.onnx）",
    )
    parser.add_argument("--m", type=int, default=130, help="A 的行数，默认 130")
    parser.add_argument("--k", type=int, default=70, help="A 的列数 / B 的行数，默认 70")
    parser.add_argument("--n", type=int, default=50, help="B 的列数，默认 50")

    args = parser.parse_args()

    # 简单校验：三个维度都必须是正数。
    for name, value in (("M", args.m), ("K", args.k), ("N", args.n)):
        if value <= 0:
            print(f"错误：{name} 必须是正数，当前是 {value}")
            return 1

    # 输出目录可能不存在，先创建。
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = build_matmul_model(args.m, args.k, args.n)
    onnx.save(model, args.output)

    print("已生成 ONNX 模型：")
    print(f"  文件 : {args.output}")
    print(f"  算子 : MatMul")
    print(f"  A    : f32 [{args.m}, {args.k}]")
    print(f"  B    : f32 [{args.k}, {args.n}]")
    print(f"  C    : f32 [{args.m}, {args.n}]")
    print()
    print("下一步可以运行：")
    print(f"  python demo/onnx_to_ttir.py {args.output} -o {args.output.with_suffix('.ttir')} --verbose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
