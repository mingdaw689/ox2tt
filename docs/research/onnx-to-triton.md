# ONNX → Triton IR 转换：一手资料调研与工程建议

- 日期：2026-11-09
- 范围：ONNX 模型到 Triton IR（TTIR）的转换器架构选型
- 方法：优先阅读本机源码（`/root/triton`、`/root/onnx-mlir`、`/root/triton-shared`、`/root/triton-linalg`、`/root/FlagTree`、`/root/PaddleTritonOps`、`/root/nkct`、`/root/llvm-project`），再核对官方文档与规范，并检索社区项目。另做了若干最小实测（见 §1.2 与附录 A）。
- 说明：本报告不引用任何既有内部 PoC 或其产物；所有结论仅基于本机一手源码、官方规范与本报告的独立实测。
- 证据标记：
  - **【源码】** 本机源码可直接确认，给出 `路径:行号`
  - **【文档】** 官方文档/规范确认，给出 URL
  - **【实测】** 本报告在 2026-11-09 实际运行验证
  - **【检索】** 基于 2026-11-09 的网络检索，可能遗漏
  - **【推断】** 基于以上事实的工程判断，不是事实陈述
- 版本说明：本报告的 Triton 源码行号以 `/root/triton`（**3.6.0**）为准；本机实际安装的是 **3.8.0**。两者符号名与管线结构基本一致，但行号会偏移（例如 `add_stages` 在 3.8.0 中位于 `backends/nvidia/compiler.py:598`，在 3.6.0 中位于 `third_party/nvidia/backend/compiler.py:556`）。引用时请以符号名为主、行号为辅。

---

## 目录

- [0. 结论摘要](#0-结论摘要)
- [1. “Triton IR”到底是什么](#1-triton-ir到底是什么)
  - [1.1 TTIR / TTGIR / LLIR / PTX / cubin 的层级关系](#11-ttir--ttgir--llir--ptx--cubin-的层级关系)
  - [1.2 如何得到 TTIR](#12-如何得到-ttir)
  - [1.3 TTIR 的结构与硬约束](#13-ttir-的结构与硬约束)
  - [1.4 TTIR 是否稳定，是否适合作为输出契约](#14-ttir-是否稳定是否适合作为输出契约)
- [2. ONNX 侧输入模型](#2-onnx-侧输入模型)
  - [2.1 onnx.proto 的基本结构](#21-onnxproto-的基本结构)
  - [2.2 opset 与版本漂移](#22-opset-与版本漂移)
  - [2.3 initializer 与常量](#23-initializer-与常量)
  - [2.4 shape inference 与动态维度](#24-shape-inference-与动态维度)
  - [2.5 控制流算子 If / Loop / Scan](#25-控制流算子-if--loop--scan)
  - [2.6 对转换器的直接影响](#26-对转换器的直接影响)
- [3. 现有实现盘点](#3-现有实现盘点)
  - [3.1 本地仓库逐个定位](#31-本地仓库逐个定位)
  - [3.2 社区项目现状](#32-社区项目现状)
  - [3.3 盘点结论](#33-盘点结论)
- [4. 可行架构方案对比](#4-可行架构方案对比)
  - [4.1 方案 A：基于 MLIR 的 ONNX → TTIR](#41-方案-a基于-mlir-的-onnx--ttir)
  - [4.2 方案 B：生成 Triton Python DSL，再编译得到 TTIR](#42-方案-b生成-triton-python-dsl再编译得到-ttir)
  - [4.3 方案 C：逐算子手写 Triton kernel + 运行时图调度](#43-方案-c逐算子手写-triton-kernel--运行时图调度)
  - [4.4 方案 D：间接路线（StableHLO/XLA 或 PyTorch/Inductor）](#44-方案-d间接路线stablehloxla-或-pytorchinductor)
  - [4.5 横向对比](#45-横向对比)
- [5. 关键技术难点](#5-关键技术难点)
- [6. MVP 建议](#6-mvp-建议)
- [7. 推荐路线与主要风险](#7-推荐路线与主要风险)
- [8. 参考来源](#8-参考来源)
- [附录 A：本次实测记录](#附录-a本次实测记录)
- [附录 B：缩略语与术语表](#附录-b缩略语与术语表)

---

## 0. 结论摘要

1. **“Triton IR”在工程上通常指 TTIR**：它是 MLIR 的 `tt` dialect（外加 `arith`/`math`/`scf` 等），硬件无关、tile 级。TTIR 之后还有 TTGIR（GPU layout/共享内存/MMA）、LLIR（LLVM dialect）、PTX、cubin。TTIR 是“最合适作为转换器输出”的层级，因为布局与调度尚未绑定到具体 GPU。【源码】`/root/triton/include/triton/Dialect/Triton/IR/TritonDialect.td:7`
2. **TTIR 可以被文本化、被重新解析、被编译**：`triton.compile("x.ttir")` 会从 TTGIR 阶段继续编译；本报告实测把一份 `.ttir` 编到了 sm70 的 cubin。【源码】`/root/triton/python/triton/compiler/compiler.py:87,226,292`【实测】
3. **但 TTIR 不是稳定 ABI**：Triton 官方文档没有给出 TTIR 的版本化/兼容性承诺，`tt` dialect 的 op 集合直接由 `TritonOps.td` 定义并随版本变化；`triton.compile` 甚至不在公开 Python API 文档的索引里。因此 TTIR 适合做“绑定 Triton 版本的编译产物契约”，不适合做“跨版本稳定的公共接口”。【源码】【文档】【推断】
4. **ONNX 侧没有现成的 ONNX → Triton 路径**：onnx-mlir 的 `ONNXToLinalg` 目前只支持 2D×2D MatMul 和受限的 4D Conv；其主力路径是 ONNX → Krnl → affine/LLVM（CPU 导向）。【源码】`/root/onnx-mlir/src/Conversion/ONNXToLinalg/Math/MatMul.cpp:7-14`、`NN/Conv.cpp:9-16`
5. **社区没有成熟的 ONNX → Triton IR 开源项目**：`triton-shared` 与 `triton-linalg` 都是反方向（Triton → Linalg）；`NKCT/CKTI` 是 CUDA → Triton IR；`PaddleTritonOps` 是 Triton 算子库；FlagTree 是多后端 Triton 分支；ONNX Runtime 只是把预编译的 Triton kernel 嵌进 EP，并不做图级转换。最接近的工业级先例是 XLA:GPU 的 Triton fusion emitter（HLO → TTIR）。【检索】【源码】
6. **推荐路线：B-first, A-ready**。先用“Python 图 IR + 生成 Triton DSL + 调 Triton 前端产出 TTIR”的方案拿到正确、可验证的 TTIR，再保留把 codegen 后端替换为 MLIR pass（方案 A）的接口。输出契约建议为“**每个 kernel 一个 `.ttir` 文件 + 一份 manifest.json**”，而不是一个巨大的单文件 TTIR。

---

## 1. “Triton IR”到底是什么

### 1.1 TTIR / TTGIR / LLIR / PTX / cubin 的层级关系

Triton 的编译管线在 NVIDIA backend 里被显式建模为一张 `stages` 表。`add_stages` 按顺序注册 `ttir`、`ttgir`、`llir`、`ptx`、`cubin` 五个阶段：

【源码】`/root/triton/third_party/nvidia/backend/compiler.py:556-565`

```python
stages["ttir"]  = lambda src, metadata: self.make_ttir(...)
stages["ttgir"] = lambda src, metadata: self.make_ttgir(...)
stages["llir"]  = lambda src, metadata: self.make_llir(...)
stages["ptx"]   = lambda src, metadata: self.make_ptx(...)
stages["cubin"] = lambda src, metadata: self.make_cubin(...)
```

| 层级 | 文件扩展名 | 内容 | 硬件相关性 | 生成函数 |
|---|---|---|---|---|
| TTIR | `.ttir` | `tt` dialect + `arith`/`math`/`scf`/`cf`；tile 级张量运算、指针运算、`tt.dot`/`tt.reduce` | 硬件无关 | `make_ttir` |
| TTGIR | `.ttgir` | 给张量分配 layout（`#ttg.blocked`、`#ttg.mma` 等）、共享内存、warp 划分；`ttg.` 系列 op | GPU 相关 | `make_ttgir` |
| LLIR | `.llir` | LLVM dialect（`llvm.` 系列） | 目标相关 | `make_llir` |
| PTX | `.ptx` | NVIDIA PTX 汇编 | NVIDIA 相关 | `make_ptx` |
| cubin | `.cubin` | 可加载的 GPU 二进制 | 架构相关 | `make_cubin` |

- TTIR 阶段跑的是平台无关 pass：inliner、tensor pointer 重写、canonicalize、combine、reorder broadcast、CSE、symbol DCE、loop unroll。【源码】`/root/triton/third_party/nvidia/backend/compiler.py:235-250`
- TTGIR 阶段才引入 `convert_to_ttgpuir`、coalesce、accelerate matmul、pipeline 等 GPU 相关优化。【源码】`/root/triton/third_party/nvidia/backend/compiler.py:252-347`
- 对 ONNX 转换器而言，**TTIR 是正确的输出层级**：如果直接产出 TTGIR，就必须提前决定 layout、`num_warps`、共享内存和具体架构，等于把后端职责揽到自己身上。【推断】

```mermaid
flowchart LR
  P[Python DSL<br/>@triton.jit] -->|ast_to_ttir| T[TTIR<br/>tt dialect]
  L[ONNX 转换器] -->|目标输出| T
  T -->|make_ttir<br/>平台无关 pass| T2[TTIR]
  T2 -->|make_ttgir<br/>layout / shared / MMA| G[TTGIR]
  G -->|make_llir| LIR[LLIR]
  LIR -->|make_ptx| PTX[PTX]
  PTX -->|ptxas| C[cubin]
```

### 1.2 如何得到 TTIR

有四条可操作路径，按“与 Triton 版本的耦合程度”从低到高排列。

#### 路径 1：Python DSL → `triton.compile` → 读 `CompiledKernel.asm["ttir"]`

`compile()` 会把每个阶段的产物放进 cache，并在 `CompiledKernel` 里暴露为 `asm` 字典：

【源码】`/root/triton/python/triton/compiler/compiler.py:226-330`、`:426-430`

```python
import triton
from triton.backends.compiler import GPUTarget

src = triton.compiler.ASTSource(
    fn=my_kernel,
    signature={"x_ptr": "*fp32", "n": "i32"},
    constexprs={"BLOCK": 128},
)
k = triton.compile(src, target=GPUTarget("cuda", 70, 32))
print(k.asm["ttir"])   # 文本形式
```

这是最“官方语义”的方式：TTIR 由 Triton 自己的 AST→IR 前端生成，广播、类型提升、mask、指针运算都由前端处理。【源码】`/root/triton/python/triton/compiler/code_generator.py:1620`（`ast_to_ttir`）

同时可以用环境变量把每个阶段的 IR dump 到磁盘：

- `TRITON_KERNEL_DUMP=1`
- `TRITON_DUMP_DIR=<dir>`

【源码】`/root/triton/python/triton/knobs.py:345,358`；【文档】`/root/triton/README.md:237-238`

#### 路径 2：只跑前端 + `make_ttir`，不需要 ptxas/GPU

如果目标只是产出 TTIR，不必把整条管线跑到 cubin。可以直接调用 `ASTSource.make_ir` 和 backend 的 `make_ttir`：

【源码】`/root/triton/python/triton/compiler/compiler.py:52-85`（`ASTSource.make_ir`）、`/root/triton/third_party/nvidia/backend/compiler.py:235`

```python
from triton._C.libtriton import ir
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend
from triton.backends.compiler import GPUTarget

target = GPUTarget("cuda", 70, 32)
backend = make_backend(target)
options = backend.parse_options({})
ctx = ir.context(); ir.load_dialects(ctx); backend.load_dialects(ctx)

src = ASTSource(fn, signature=..., constexprs=...)
module = src.make_ir(target, options, backend.get_codegen_implementation(options),
                     backend.get_module_map(), ctx)
ttir = backend.make_ttir(module, {}, options, 70)   # 返回 ir.module
text = str(ttir)
```

本报告实测这条路径在 **不依赖 `ptxas`、也没有触发 GPU driver 调用** 的情况下成功产出了 TTIR 文本（见附录 A.1）。【实测】

#### 路径 3：直接读取 `.ttir` 文件并编译

`compile()` 接受字符串路径。如果路径以 `.ttir` 结尾，会走 `IRSource`：

- `IRSource` 用 `ir.parse_mlir_module(path, context)` 解析；
- 取模块里的 entry function（第一个 `public` 的 `tt.func`）；
- `first_stage = stages.index("ttir")`，然后因为 `ir_source` 再加 1，所以**跳过 `make_ttir`**，直接从 TTGIR 开始。

【源码】`/root/triton/python/triton/compiler/compiler.py:87-112,237-238,292-295`

```python
import triton
k = triton.compile("kernel.ttir", target=GPUTarget("cuda", 70, 32))
print(sorted(k.asm.keys()))
# ['cubin', 'llir', 'ptx', 'ttgir', 'ttir']
```

本报告实测：把路径 2 生成的 `add.ttir` 直接编译，成功得到 `ttgir`/`llir`/`ptx`/`cubin`（sm70，cubin 4968 bytes）。【实测】

注意：因为是 IR 输入，TTIR 阶段的优化 pass 不会重跑。也就是说，产出的 `.ttir` 必须自身已经合法、且最好已经规范化；否则应在生成侧先跑 `triton-opt` 的对应 pass，或在编译前手工 `module.verify()`。【推断】

#### 路径 4：手工/程序化构造 TTIR

- 手写 TTIR 文本：可以手写并由 Triton 重新解析（见附录 A 的独立实测），但需要自行处理地址算术、mask、`tt.dot` 精度属性等，且必须满足 `tt` dialect 的 verifier；不适合作为通用生成方式。【源码】`/root/triton/include/triton/Dialect/Triton/IR/TritonOps.td`、`/root/triton/lib/Dialect/Triton/IR/Ops.cpp`【推断】
- C++/MLIR builder 构造 `tt` op：XLA:GPU 的 `fusion_emitter` 与 NKCT 的 `AffineToTriton` pass 都是这样做。【源码】`/root/nkct/src/AffineToTriton.cpp:711,795-837`【文档】[XLA fusion_emitter.cc](https://github.com/openxla/xla/blob/main/xla/backends/gpu/codegen/triton/fusion_emitter.cc)

#### 路径选择建议

| 场景 | 推荐路径 |
|---|---|
| 转换器自身要产出 TTIR，且允许依赖 Triton Python 包 | 路径 2（`make_ir` + `make_ttir`），最快且最稳 |
| 需要同时验证可执行性 | 路径 3（`.ttir` → `triton.compile` → cubin） |
| 转换器必须是自包含 C++ 二进制 | 路径 4（MLIR builder），但工程量大 |
| 只是调试 Triton 自己生成的 kernel | 路径 1 + `TRITON_KERNEL_DUMP` |

### 1.3 TTIR 的结构与硬约束

#### 1.3.1 模块与函数

一份 TTIR 是一个 MLIR `module`，里面是 `tt.func`。入口 kernel 必须是 **public** 的 `tt.func`：

- `tt.func` 定义：`/root/triton/include/triton/Dialect/Triton/IR/TritonOps.td:1161`
- `tt.return` 定义：`:1262`
- “kernel” 的判定是 `funcOp.getVisibility() == Public`：`/root/triton/lib/Dialect/Triton/IR/Utility.cpp:109-111`
- `.ttir` 输入的 entry function 取“第一个 public 的 `tt.func`”：`/root/triton/python/src/ir.cc:680-690`

实测生成的一个最小 TTIR：

```mlir
module {
  tt.func public @add_kernel(%x_ptr: !tt.ptr<f32>, %y_ptr: !tt.ptr<f32>,
                             %out_ptr: !tt.ptr<f32>, %n: i32)
      attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %pid = tt.get_program_id x : i32
    %offs = arith.muli %pid, %c128_i32 : i32
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    ...
    %v = arith.addf %x, %y : tensor<128xf32>
    tt.store %out_ptrs, %v, %mask : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
```

要点：

- 函数参数是 **指针 + 标量**，不是 ONNX 的“张量值”。ONNX 的输入/输出张量在 TTIR 里变成 `!tt.ptr<T>` 参数；输出也是通过指针写回，不是返回值。
- 张量是 **block 类型**（`tensor<128xf32>`），不是任意形状的张量。
- 函数体里可以直接用 `arith`、`math`、`scf` 的 op；TTIR 不是“只有 tt op”。【源码】`TritonDialect.td:31-37`
- 一个模块里可以有多个 `tt.func`，但只有 public 的那个是 entry；非 public 的函数更像 device function，不单独 launch。【源码】`ir.cc:680-690`【推断】

#### 1.3.2 op 集合：没有 conv/pool/softmax

`TritonOps.td` 里定义了 50 个 `tt` op（本报告统计），核心包括：

- 指针/内存：`tt.addptr`、`tt.load`、`tt.store`、`tt.atomic_rmw`、`tt.atomic_cas`
- 形状：`tt.splat`、`tt.expand_dims`、`tt.broadcast`、`tt.reshape`、`tt.trans`、`tt.cat`、`tt.join`、`tt.split`
- 计算：`tt.dot`、`tt.dot_scaled`、`tt.reduce`、`tt.scan`、`tt.gather`、`tt.histogram`、`tt.map_elementwise`
- 程序模型：`tt.get_program_id`、`tt.get_num_programs`、`tt.make_range`
- 其他：`tt.print`、`tt.assert`、tensor descriptor 系列

【源码】`/root/triton/include/triton/Dialect/Triton/IR/TritonOps.td`（op 定义全文）

**没有** `conv`、`pool`、`softmax`、`slice`、`pad`、`resize`、`topk`、`nms` 这类高层 op。这些都必须由转换器展开成 `tt.load` + `tt.dot` / `tt.reduce` / `arith` + mask 的组合。【源码】【推断】

#### 1.3.3 硬约束（会直接决定 ONNX 映射策略）

| 约束 | 事实 | 对 ONNX 转换的影响 |
|---|---|---|
| block 形状每一维必须是 2 的幂 | 前端 `validate_block_shape` 检查 `is_power_of_two(d)`：`/root/triton/python/triton/_utils.py:62-69`；TTGIR 的 LinearLayout 推导也假设 shape 为 2 的幂：`/root/triton/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp:1186-1190` | 28×28、784、1000 这类维度不能直接当 block；必须 pad 到 32×32、1024 等并用 mask |
| 单个 block 元素数上限 1,048,576 | `TRITON_MAX_TENSOR_NUMEL = 1048576`：`/root/triton/python/triton/_utils.py:11` | 大张量必须 tile，不能“整张量一个 block” |
| `tt.dot` 只接受 2D 或 3D | 前端断言：`/root/triton/python/triton/language/semantic.py:1532` | ONNX MatMul 的 N-D/broadcast 形式要先 reshape/循环 |
| `tt.dot` 的 K 有下限 | fp16/bf16 类 K≥16，int8 K≥32；M/N 下限为 1：`/root/triton/third_party/nvidia/backend/compiler.py:19-30`、`semantic.py:1535-1540` | 小 K 的 MatMul 要 pad 或用非 dot 实现 |
| `tt.reduce`/`tt.scan` 的 axis 是属性 | `I32Attr:$axis`：`TritonOps.td:748,781` | 规约轴必须编译期确定；动态轴不支持 |
| grid 轴只有 x/y/z | `program_id` 只接受 0/1/2：`semantic.py:40` | 高维网格要 flatten 到 3 维以内 |
| f32 `tt.dot` 默认 tf32 | `default_dot_input_precision: str = "tf32"`：`/root/triton/third_party/nvidia/backend/compiler.py:123` | 与 ONNX Runtime 的 fp32 结果对拍会差 1e-3 量级；验证时要显式 `inputPrecision=ieee` |

这些约束不是“实现细节”，而是转换器设计的边界条件。尤其是 **2 的幂** 与 **1M 元素上限**：ONNX 的逻辑形状（logical shape）与 Triton 的 block 形状（tile shape）必须解耦，tile 形状是“向上取整到 2 的幂的 padding 形状 + mask”，不是 ONNX 形状本身。【推断】

### 1.4 TTIR 是否稳定，是否适合作为输出契约

#### 事实

- Triton 官方文档有 `tt` dialect 的自动生成页面，列出 op 与类型，但**没有**版本化、兼容性、冻结策略的说明。【文档】[Triton MLIR Dialects](https://triton-lang.org/main/dialects/dialects.html)、[`tt` Dialect](https://triton-lang.org/main/dialects/TritonDialect.html)、[TritonOps](https://triton-lang.org/main/dialects/TritonOps.html)
- `tt` op 直接定义在 `TritonOps.td`，随 Triton 版本增删改；不同版本的 op 集合与属性可能不同。【源码】
- 公开 Python API 文档只索引 `triton.jit`、`autotune`、`heuristics`、`Config` 等；`triton.compile`、`ASTSource`、`IRSource` 没有出现在该索引中。【文档】[triton — Triton documentation](https://triton-lang.org/main/python-api/triton.html)
- 但 `triton.compile` 确实支持 IR 文件输入，并且 `TRITON_KERNEL_OVERRIDE` 明确允许用户替换 `.ttir/.ttgir/.llir/.ptx`，说明 IR 文本是“事实上可用”的，而不是纯内部格式。【源码】`compiler.py:87-112`、`README.md:237-262`
- Triton 的测试（`test/**/*.mlir`、`triton-opt`）大量使用 TTIR/TTGIR 文本，说明语法相对稳定，但这是测试基础设施的稳定性，不是对外 ABI 承诺。【源码】`/root/triton/test/`、`/root/triton/bin/triton-opt.cpp`

#### 判断

**TTIR 适合作为“版本绑定的编译产物契约”，不适合作为“跨版本稳定的公共接口”。**【推断】

具体建议：

1. **固定 Triton 版本**（本机已安装 3.8.0；源码树 `/root/triton` 是 3.6.0，两者不要混用）。把 `triton.__version__` 写进产物元数据。
2. 产物中记录 `target`（如 `cuda:70`）、`num_warps`、`num_stages`、`llvm-hash`。
3. 在消费端做**启动时校验**：用 `ir.parse_mlir_module` 重新解析，并跑 `module.verify()`；版本不匹配直接报错，而不是“尽力而为”。
4. 如果项目需要对外承诺稳定 IR，则应该定义**自己的稳定图 IR + manifest**，把 TTIR 当作一个 backend target；或者只承诺“同一 Triton 版本内可复现”。

#### 输出契约建议

不要输出“一个巨大的 TTIR 文件”。TTIR 没有图/launch 的概念，`tt.func` 是 kernel 级单位；多个 public 函数会产生 entry 选择歧义（`get_entry_func_name` 取第一个）。【源码】`ir.cc:680-690`【推断】

建议契约：

```
out/
  model.manifest.json
  kernels/
    mm_0.ttir
    add_bias_relu_1.ttir
    softmax_2.ttir
```

`manifest.json` 至少包含：

```json
{
  "schema_version": "0.1",
  "onnx_model_hash": "sha256:...",
  "onnx_opset": 17,
  "triton_version": "3.8.0",
  "target": {"backend": "cuda", "arch": 70, "warp_size": 32},
  "kernels": [
    {
      "name": "mm_0",
      "ttir": "kernels/mm_0.ttir",
      "signature": [{"name": "a_ptr", "type": "*fp32"}, ...],
      "constexprs": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
      "grid": ["cdiv(M, BLOCK_M)", "cdiv(N, BLOCK_N)", "1"],
      "num_warps": 4,
      "num_stages": 3,
      "inputs": ["a", "b"],
      "outputs": ["y"],
      "initializers": [],
      "deps": []
    }
  ]
}
```

这样即使 TTIR 本身不含 grid/参数绑定信息，产物仍然是可执行的。

---

## 2. ONNX 侧输入模型

### 2.1 onnx.proto 的基本结构

ONNX 的语法由 protobuf 定义，语义由 `docs/IR.md` 定义。【文档】[onnx.proto3](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3)、[IR.md](https://github.com/onnx/onnx/blob/main/docs/IR.md)

| 消息 | 本地行号 | 作用 |
|---|---|---|
| `ModelProto` | `third_party/onnx/onnx/onnx.proto3:446` | 顶层模型：`ir_version`、`opset_import`、`graph`、`functions`、metadata |
| `GraphProto` | `:561` | `node[]`、`initializer[]`、`input[]`、`output[]`、`value_info[]` |
| `NodeProto` | `:220` | `op_type`、`domain`、`input[]`、`output[]`、`attribute[]` |
| `TensorProto` | `:604` | 权重/常量：dtype、dims、raw_data 等 |
| `AttributeProto` | `:134` | `f`/`i`/`s`/`t`/`g`（graph）/`graphs` 等 |
| `ValueInfoProto` | `:201` | 值名 + 类型（含 shape） |
| `OperatorSetIdProto` | `:913` | `(domain, version)` |

关键语义：

- Graph 是**拓扑排序、无环、SSA** 的节点列表。`IR.md:222,257,315` 明确要求 SSA 与拓扑序。
- 节点输入可以引用 graph input、initializer 或前序节点输出；节点输出引入新名字。`IR.md:278-315`
- 属性是编译期常量；属性值可以是 Graph（控制流子图）。`IR.md:495-497`
- initializer 既可以作为默认输入（名字同时出现在 input 列表），也可以是纯常量。`IR.md:303`

### 2.2 opset 与版本漂移

ONNX 的算子语义按 opset 版本演进，同一个 `op_type` 在不同 opset 下可能有不同的输入/属性/默认值。典型例子：

- `Squeeze`/`Unsqueeze`：axes 从属性变成输入（opset 13）。
- `Split`：split 从属性变成输入（opset 13）。
- `Softmax`：默认 axis 从 1 变成 -1（opset 13）。
- `Clip`：min/max 从属性变成输入（opset 11）。
- `Pad`：pads 从属性变成输入（opset 11）。
- `Resize`：10/11/13/18/19 多次改语义。

【文档】[ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md)；本机同一份文档在 `/root/onnx-mlir/third_party/onnx/docs/Operators.md`

工程建议：

1. 读取 `ModelProto.opset_import`，把目标 opset 固定为一个值（例如 17 或 onnx-mlir 支持的 22）。
2. 用 `onnx.version_converter.convert_version(model, target_opset)` 做归一化；不支持的版本直接拒绝。
3. 用 `onnx.inliner.inline_local_functions(model)` 把模型本地函数展开，避免转换器还要处理 `FunctionProto`。【实测：onnx 1.21.0 提供 `version_converter`、`inliner`、`shape_inference`】

【推断】把 opset 归一化放在最前面，可以显著减少算子 lowering 的分支数量；否则每个 op 都要写多版本 pattern。

### 2.3 initializer 与常量

- initializer 是 `TensorProto`，可以是数 MB 的权重。
- 不建议把权重直接嵌进 TTIR 的 `arith.constant dense<...>`：
  - TTIR 文本会膨胀到不可维护；
  - Triton 会把常量放进编译单元，编译时间和寄存器/常量内存压力都会变差；
  - 权重更新/共享也不方便。
- 建议把 initializer 导出为独立的二进制 blob（或 `numpy` `.npy`），在 manifest 里记录 `name → blob offset/dtype/shape`，运行时作为 device buffer 上传，并以 `!tt.ptr<T>` 参数传入 kernel。【推断】

### 2.4 shape inference 与动态维度

ONNX 的 `ValueInfoProto` 里，shape 可以是：

- 完全静态：`dim_value = 100`
- 符号维度：`dim_param = "batch"`
- 未知维度：两者都不设
- 未知 rank：整个 shape 缺失（合法）

【文档】`IR.md:444,474,491`

`onnx.shape_inference.infer_shapes` 可以补全 `value_info`，但**不保证完整**：

> “Shape inference is not guaranteed to be complete. In particular, some dynamic behaviors block the flow of shape inference...”
> `ShapeInference.md:56-62`

对转换器的影响：

1. **rank 必须静态**。Triton 的 block 类型需要确定的维数；rank 未知的 ONNX 值无法映射。MVP 应直接拒绝 unknown-rank。
2. **dim 可以动态**。把动态 dim 作为 `i32` 标量参数传入 kernel，host 侧用实际 shape 计算 grid；kernel 内部用 `offs < dim` 生成 mask。【源码】Triton 的 `tt.splat`/`arith.cmpi` 组合即可表达。
3. **动态 shape 的 reshape/transpose 很难**。`tt.reshape` 的结果类型是静态 block 形状；如果目标形状依赖运行时值，通常无法直接表达。MVP 只支持“目标形状静态可推导”的 reshape。
4. **数据依赖形状**（NonZero、Unique、动态 Slice、动态 TopK）不要进 MVP。

### 2.5 控制流算子 If / Loop / Scan

| 算子 | 语义 | 本地文档 | 对转换器的挑战 |
|---|---|---|---|
| `If` | 两个子图，按 `cond` 选择 | `Operators.md:15826` | 动态 cond 需要 `scf.if` 或 host 分支；两个分支输出 shape 必须一致 |
| `Loop` | 通用循环，带 loop-carried deps 与 scan outputs | `Operators.md:17793` | 循环体可能包含需要独立 kernel 的算子；难以塞进单个 `tt.func` |
| `Scan` | 扫描/累积，带状态与 scan inputs | `Operators.md:31921` | 只有结合律成立的扫描才能用 `tt.scan`；一般 body 需要 host 循环 |

关键判断：

- **常量 cond 的 If** 可以在转换期常量折叠，直接选一个分支。
- **动态 cond 的 If**：如果两个分支都是 tile 可表达的 elementwise/小算子，可以生成一个 kernel，内部用 `scf.if` + `tt.return`/`scf.yield`；如果分支里有 matmul/conv，通常只能拆成两个 kernel，host 侧按 cond 选择。
- **Loop/Scan**：Triton kernel 是 SPMD 的，不能在 kernel 内 launch 另一个 kernel。通用 Loop 只能：
  1. 在 host 侧展开/调度（多 kernel + 循环）；
  2. 或者识别为可融合的局部循环（如 RNN 的 time loop），生成一个 persistent kernel（很难）。
- MVP 建议：**不支持动态 Loop/Scan**；常量 trip count 的小循环可以展开；RNN/beam search 类模型明确排除。【推断】

### 2.6 对转换器的直接影响

| ONNX 特性 | 转换器必须做什么 | MVP 决策 |
|---|---|---|
| Graph SSA + 拓扑序 | 建立值编号、use-def、拓扑调度 | 必须 |
| opset 漂移 | 版本归一化 / 多版本 pattern | 固定目标 opset |
| initializer | 常量折叠 + 权重外置 | 权重外置 |
| 静态 shape | 直接映射为 tile shape（pad 到 2 的幂） | 支持 |
| 动态 dim | 标量参数 + mask + host grid | 支持 batch 维 |
| 未知 rank | 无法映射 | 拒绝 |
| 控制流 | scf / host 编排 | 仅常量 If |
| 可选/可变输入 | 缺失输入的处理、默认值 | 按 op 白名单 |
| 自定义 domain | 无法映射 | 拒绝 |

---

## 3. 现有实现盘点

### 3.1 本地仓库逐个定位

| 仓库 | 方向/定位 | 关键事实 | 维护状态 | 对本项目可用性 |
|---|---|---|---|---|
| `/root/triton` | Triton 编译器本体 | `tt` dialect、AST→TTIR、TTGIR/LLIR/PTX/cubin、`triton.compile`、IRSource | 活跃（上游 20.5k stars） | **核心依赖**：TTIR 定义与编译后端 |
| `/root/onnx-mlir` | ONNX → MLIR → 可执行 | ONNX dialect、shape inference、ONNXToKrnl（默认 CPU 路径的主力）、ONNXToLinalg（仅 MatMul/Conv）、ONNXToTOSA/StableHLO | 活跃（1.1k stars，opset 22） | **前端参考/可选依赖**；其 linalg 路径不足以支撑通用转换 |
| `/root/triton-shared` | Triton → Linalg（反方向） | 把 Triton IR 降到 Linalg/Tensor/Memref，供 CPU/其他后端 | **已归档/停止维护**（README 第一行；GitHub `archived=true`） | 不可直接复用；可参考其指针分析、mask 分析 |
| `/root/triton-linalg` | Triton → Linalg（反方向） | 面向寒武纪后端；`tt.reduce→linalg.reduce`、`tt.dot→linalg.matmul` | 2025-02 后基本停更（229 stars） | 不可直接复用；反方向 |
| `/root/FlagTree` | 多后端 Triton 分支 | FlagOS 维护，支持 NVIDIA/AMD/寒武纪/昇腾等十余个后端，Triton 3.6 | 活跃（385 stars） | 若要做多硬件后端，可作为 Triton 发行版候选；不解决 ONNX 前端 |
| `/root/PaddleTritonOps` | Triton 算子库 | 用 Triton 实现 Paddle 算子（softmax、matmul、conv2d、flash attention 等），统一 runtime/backend 包装 | 内部仓库，活跃 | **方案 C 的现成素材**；不是 IR 转换器 |
| `/root/nkct` | CUDA → Triton IR | Clang 解析 CUDA → `nk` dialect → `tt`/TTGIR；`AffineToTriton` 用 MLIR `DialectConversion` 构造 `tt.load/store/reduce` | 内部仓库；对应论文 CKTI | **方案 A 的可行性证据**；输入不是 ONNX |
| `/root/llvm-project` | MLIR 基础设施 | `ConversionTarget`、`TypeConverter`、`RewritePattern`、`applyPartialConversion` | 活跃 | 方案 A 的基础设施 |

补充事实：

- `triton-shared` 的 README 明确写着 “This repository is no longer maintained.”，并说明其目标是 Triton IR → 中间层 → 硬件特定 IR，不是 ONNX 前端。它的主要 pass 包括 `TritonToStructured`、`TritonToUnstructured`、`TritonToLinalg`、`TritonPtrToMemref`、`StructuredToMemref`、`TritonArithToLinalg`；核心分析是 pointer analysis、use analysis、mask analysis。【源码】`/root/triton-shared/README.md:1-12,83-101`、`/root/triton-shared/include/triton-shared/Conversion/`
- `triton-linalg` 的 README 说明它把 Triton dialect 转成 Linalg，并支持寒武纪后端；方向同样是 Triton → Linalg。它包含 AxisInfo analysis、MaskTracker、PointerOffsetTracker，以及 `aux`/`linalg_ext` 两个扩展 dialect。【源码】`/root/triton-linalg/README.md:1-10,60-95`
- `FlagTree` 是 Triton 的多后端 fork，不包含 ONNX 前端。【源码】`/root/FlagTree/README.md`（多后端表）
- `PaddleTritonOps` 的 `OperatorList.md` 列出了 150+ 个条目（含不同版本/变体），是“逐算子 kernel 库”的典型形态。【源码】`/root/PaddleTritonOps/OperatorList.md`
- `nkct` 的 `AffineToTriton.cpp` 用 `rewriter.create<triton::LoadOp/StoreOp/ReduceOp/GetProgramIdOp>` 构造 TTIR，并用 `ConversionTarget` 把 `func/memref` 标为 illegal。【源码】`/root/nkct/src/AffineToTriton.cpp:711,795-837,1100-1140`

### 3.2 社区项目现状

检索结论（截至 2026-11-09）：**没有找到成熟的、公开的 “ONNX → Triton IR” 转换器**。【检索】

最接近的项目都不是这个方向：

| 项目 | 实际方向 | 成熟度/状态 | 为什么不是答案 |
|---|---|---|---|
| [microsoft/triton-shared](https://github.com/microsoft/triton-shared) | Triton → Linalg | 348 stars，已归档 | 反方向 |
| [Cambricon/triton-linalg](https://github.com/Cambricon/triton-linalg) | Triton → Linalg | 229 stars，2025-02 后停更 | 反方向 |
| [NKCT / CKTI](https://doi.org/10.1145/3797905.3800551) | CUDA → Triton IR | 论文 + 内部实现 | 输入是 CUDA，不是 ONNX |
| [XLA:GPU Triton emitter](https://github.com/openxla/xla/blob/main/xla/backends/gpu/codegen/triton/fusion_emitter.cc) | HLO fusion → TTIR | 工业级，但绑定 XLA | 不是 ONNX 前端；依赖 XLA 生态 |
| [ONNX Runtime Triton kernel](https://github.com/microsoft/onnxruntime/blob/main/docs/ORT_Use_Triton_Kernel.md) | 预编译 Triton kernel 嵌入 ORT EP | 仅 softmax，主要 ROCm 路径 | 不做图级转换 |
| [IBM/onnxmlir-triton-backend](https://github.com/IBM/onnxmlir-triton-backend) | 把 onnx-mlir 编译出的 `.so` 部署到 Triton Inference Server | 5 stars，2024 后不活跃 | 是 serving 后端，不是 IR 转换 |
| [KernelLens](https://zenodo.org/records/23194523) | Triton kernel → ONNX/TensorRT 插件 | 新项目 | 反方向 |
| 中国专利 CN119336336B | ONNX → MLIR/Triton codegen | 专利，无开源实现 | 不可直接用 |

另外，`Avi197/pytorch-to-onnx-to-triton` 之类的仓库名字容易误导，实际是 Triton Inference Server 的部署示例，不是编译器。【检索】

### 3.3 盘点结论

1. **“ONNX → Triton IR” 是一个真实存在的空白**。现有工作要么是 Triton → Linalg（供 CPU/其他后端），要么是 CUDA → Triton（NKCT/CKTI），要么是 HLO → TTIR（XLA），要么是把 Triton kernel 塞进 ONNX Runtime。
2. **onnx-mlir 不能直接复用为 Triton 前端**：它的 ONNXToLinalg 只有两个 pattern（MatMul 2D、Conv 4D 且限制很多），主力是 ONNXToKrnl（CPU 导向）。【源码】`MatMul.cpp:7-14`、`Conv.cpp:9-16`、`ConvertONNXToLinalg.cpp:54-57`
3. **最接近的工业先例是 XLA:GPU**：它证明了“在 MLIR 里程序化构造 TTIR 并交给 Triton 后端”在产品中可行。NKCT/CKTI 则证明了用 MLIR `DialectConversion` 构造 `tt` op 的可行性，但输入是 CUDA 而非 ONNX。
4. **TTIR 作为编译输入是可验证的**：本报告独立实测了 `.ttir → TTGIR → LLIR → PTX → cubin`（见附录 A），因此“转换器输出 TTIR”在技术上成立；但 TTIR 的版本稳定性风险仍需单独管理。
5. **PaddleTritonOps 证明了方案 C 的可行性**，但它不是转换器；它更适合作为“某些难算子的手写 kernel 后备库”。

---

## 4. 可行架构方案对比

### 4.1 方案 A：基于 MLIR 的 ONNX → TTIR

#### 数据流

```mermaid
flowchart LR
  O[.onnx] --> P[ONNX 解析<br/>onnx-mlir ONNX dialect 或自写 importer]
  P --> S[Shape/Type 分析<br/>ONNX shape inference]
  S --> L{中间层选择}
  L -->|A1 直接| T[tt dialect<br/>DialectConversion]
  L -->|A2 经 linalg| LA[linalg]
  LA --> T
  T --> P2[TTIR 文本<br/>MLIR printer]
```

两种子方案：

- **A1：ONNX dialect → `tt` dialect 直接转换**。用 `ConversionTarget` 把 `tt` 标为 legal、`onnx` 标为 illegal，用 `OpConversionPattern` 逐算子构造 `tt.load/store/dot/reduce`。
- **A2：ONNX → linalg → `tt`**。先补 onnx-mlir 的 ONNXToLinalg（目前只有 2 个 op），再写 linalg→tt 的通用 pass；这条路线需要同时维护 ONNX→linalg 与 linalg→tt 两套 lowering，前期成本较高。【推断】

#### 需要的组件

- ONNX 解析与 ONNX dialect（可复用 onnx-mlir 的 `src/Dialect/ONNX`）
- ONNX shape/type inference
- kernel 划分 pass（把整图切成多个 `tt.func`）
- 每个 ONNX op 的 lowering pattern（或 linalg 生成 pattern + 通用 linalg→tt pattern）
- `TypeConverter`：`tensor<...>` → `!tt.ptr<T>` 参数；输出改写入参
- `ConversionTarget` / `RewritePatternSet` / `applyPartialConversion`
- TTIR printer / verifier

#### 优点

- 真正的编译器架构：有 verifier、diagnostics、pass pipeline、lit 测试。
- 直接产出 MLIR 文本，不依赖 Python 运行时。
- 可以复用 ONNX dialect、shape inference、linalg 的 tiling/fusion 基础设施。
- 与 XLA/NKCT 的工业实践一致，长期可维护性最好（如果版本对齐解决）。
- 可以把“ONNX 前端”和“TTIR 后端”解耦，未来接 TOSA/StableHLO/torch-mlir 都受益。

#### 缺点

- **LLVM/MLIR 版本耦合是最大风险**：
  - onnx-mlir 固定 LLVM `1053047a4be7d1fece3adaf5e7597f838058c947`（`/root/onnx-mlir/docs/BuildOnLinuxOSX.md:18`）；
  - Triton 固定 LLVM `27d654c4c4e6eb7c19e46af20500200e793da7c7`（`/root/triton/cmake/llvm-hash.txt:1`）；
  - 两者要在同一个进程里链接，必须统一到同一份 LLVM/MLIR 构建。重建 onnx-mlir 或重建 Triton 都是大工程。
  - 退路是拆成两个进程、用 MLIR 文本通信（onnx-mlir 进程输出 `.mlir`，另一个链接 Triton 的进程读入并转 TTIR），代价是跨进程解析/打印和调试成本；另一种做法是把 Triton 源码对齐到 onnx-mlir 的 LLVM，但这意味着要维护一份 Triton 构建。【推断】
- onnx-mlir 的 ONNXToLinalg 只有 MatMul/Conv 两个 pattern，A2 需要先补大量 ONNX→linalg lowering。
- A1 则需要自己实现每个 op 的 tile 化语义（广播、mask、类型提升、地址算术），等于重写 Triton 前端的一部分。
- C++ 迭代慢、构建时间长；对动态 shape 的表达也更繁琐。
- 需要自己保证 TTIR verifier 通过（尤其是 `tt.dot` 的精度属性、`tt.reduce` 的 region）。

#### 契合度与工作量

- 与“输出 Triton IR”目标契合度：**最高**（直接构造 TTIR）。
- MVP（MatMul + elementwise + ReLU/Softmax）：约 **2–3 人月**（含工具链对齐）。【推断】
- CNN（Conv/Pool/BN/Reshape）：再加 **4–6 人月**。【推断】
- 广泛算子覆盖：12 人月以上。【推断】
- 可维护性：高（如果版本对齐稳定）；版本升级成本高。
- 动态 shape：可做，但要手写运行时维度参数与 mask 逻辑。
- 控制流：`scf.if/for` 可表达一部分；通用 Loop/Scan 仍要 host 编排。

### 4.2 方案 B：生成 Triton Python DSL，再编译得到 TTIR

#### 数据流

```mermaid
flowchart LR
  O[.onnx] --> I[onnx importer<br/>check / infer_shapes / version_converter / inline]
  I --> G[Graph IR<br/>SSA + 类型 + 符号维度]
  G --> P[Partitioner / Fusion]
  P --> K[KernelPlan<br/>tile shape + grid + args]
  K --> E[Triton DSL codegen<br/>每个 kernel 一个 @triton.jit]
  E --> C[triton.compile<br/>ASTSource.make_ir + make_ttir]
  C --> T[.ttir]
  T --> M[manifest.json]
```

#### 需要的组件

- `onnx` Python 包：解析、checker、shape inference、version converter、inliner
- 自己的 Graph IR：Value/Node/TensorType/opset/initializer
- 常量折叠、死代码消除、拓扑排序
- shape/type 分析与符号维度传播
- kernel 划分与融合（anchor op + elementwise 贪心融合）
- 每个 ONNX op 的 Triton DSL emitter（一个函数生成一段 Python）
- 一个小的源码模板/拼接层（注意变量命名、缩进、类型）
- TTIR 提取器：
  - 轻量：`ASTSource.make_ir` + `backend.make_ttir`（无需 ptxas/GPU）
  - 完整：`triton.compile` 后读 `kernel.asm["ttir"]`（同时得到 ttgir/ptx/cubin）
- manifest writer
- 数值验证 harness（vs onnxruntime）

#### 优点

- **最快拿到“正确且合法”的 TTIR**：广播、类型提升、mask、地址算术都由 Triton 前端处理，转换器不需要重新实现这些语义。
- 不依赖 LLVM/MLIR 版本对齐；用 pip 安装的 Triton 即可。
- 生成的 Python 源码可读、可调试、可手工修改；每个 op 的 lowering 是一个小函数，迭代快。
- 动态 shape 可以直接用标量参数 + mask 表达，和 Triton 原生写法一致。
- 同一套流程可以顺带产出 TTGIR/PTX/cubin，方便做执行验证。
- 可以用 `triton.compile` 的 cache，重复编译很快。

#### 缺点

- **源码生成是字符串拼接**，需要严格的命名/转义/类型管理；生成器本身容易出 bug。
- 需要在转换时执行 Python/Triton 前端；每次编译是秒级，kernel 多时慢（可缓存、可并行）。
- 产出 TTIR 依赖 Triton 版本；升级 Triton 可能改变 TTIR 文本。
- 融合范围受“一个 Python kernel 能表达什么”限制；跨 matmul 的复杂融合仍然难。
- 控制流（Loop/Scan）仍然需要 host 编排或专门 kernel。
- 某些 op（动态 reshape、数据依赖形状）无法直接表达。

#### 契合度与工作量

- 与“输出 Triton IR”目标契合度：**高**（输出的是 Triton 自己生成的 canonical TTIR）。
- MVP（MatMul + elementwise + ReLU/GELU + Softmax + MLP）：约 **3–5 人周**。【推断】
- CNN（Conv/Pool/BN/Reshape/Transpose）：再加 **6–10 人周**。【推断】
- 可维护性：中高；Triton 版本升级需要回归测试，但比 C++ ABI 对齐轻。
- 动态 shape：好（tile 级）；rank 必须静态。
- 控制流：常量 If 容易；动态 If 可生成 `tl.where`/`scf.if`；通用 Loop/Scan 难。

### 4.3 方案 C：逐算子手写 Triton kernel + 运行时图调度

#### 数据流

```
.onnx → 图解析 → 逐 op 分发到预写 kernel（matmul/softmax/conv/...）
      → host 侧按拓扑序 launch → 中间 buffer 管理
```

#### 需要的组件

- ONNX 图解析与调度器
- 每个 op family 的手写 Triton kernel（dtype × rank × 属性 × 布局的组合）
- autotune / 启发式选择 kernel 变体
- 中间 buffer 分配与复用
- 可选的 per-kernel TTIR dump（作为副产物）

#### 优点

- 单 op 性能最容易调优；可以针对常见 shape 写专用 kernel。
- 工程上最直接，PaddleTritonOps/ORT 都是这个形态。
- 每个 kernel 可以独立测试，风险局部化。

#### 缺点

- **不是 IR 转换器**：输出是“kernel 库 + 调度计划”，TTIR 只是每个 kernel 的副产物。
- 算子组合爆炸：dtype × rank × layout × 属性（transA/transB/group/dilation/pads...）。
- 没有跨算子融合，MLP/CNN 的性能会明显落后。
- 动态 shape 需要预编译多个 BLOCK_SIZE 变体（ORT 的 softmax 就是编译 1024/2048/4096/8192/16384 五档）。【文档】[ORT Use Triton Kernel](https://github.com/microsoft/onnxruntime/blob/main/docs/ORT_Use_Triton_Kernel.md)
- 与“输出 Triton IR”目标契合度低。

#### 契合度与工作量

- 契合度：**低**（除非项目真正目标是“在 Triton 上跑 ONNX”而不是“输出 IR”）。
- 每个 op family 约 1–2 人周；广泛覆盖的总工作量与方案 A/B 相当甚至更大。【推断】
- 可维护性：中；kernel 多、测试矩阵大。
- 动态 shape：靠多档 BLOCK_SIZE/autotune，能覆盖常见 batch，但不是通用解。
- 控制流：host 侧调度，能处理一部分 Loop。

### 4.4 方案 D：间接路线（StableHLO/XLA 或 PyTorch/Inductor）

两条间接路线，均不推荐作为主方案：

1. **ONNX → StableHLO → XLA:GPU → TTIR**
   - onnx-mlir 有 `src/Conversion/ONNXToStablehlo`，XLA:GPU 有 Triton fusion emitter。
   - 但 XLA 的 emitter 面向 HLO fusion，不是通用的“ONNX 图 → TTIR”；需要引入 XLA 巨型依赖，且中间语义转换不可控。
   - 适合“验证可行性”，不适合作为产品主线。

2. **ONNX → PyTorch（onnx2torch）→ torch.compile/Inductor → Triton**
   - 依赖 torch，转换不一定语义等价（自定义 op、动态 shape、控制流）。
   - 产出的是 Inductor 生成的 Triton kernel，不是“ONNX 模型对应的 TTIR”。
   - 只适合作为对照/应急。

【推断】方案 D 的工程风险高于收益，除非团队已经有 XLA 基础设施。

### 4.5 横向对比

| 维度 | A：MLIR 直接构造 | B：生成 Triton DSL | C：算子库 + 调度 | D：间接路线 |
|---|---|---|---|---|
| 输出是否为真正的 TTIR | 是 | 是（Triton 前端生成） | 仅 per-kernel 副产物 | 是，但不可控 |
| 数据流 | ONNX→MLIR→tt | ONNX→GraphIR→Python→TTIR | ONNX→dispatcher→kernel | ONNX→HLO/torch→TTIR |
| 核心依赖 | onnx-mlir + Triton C++ + 同一 LLVM | onnx + triton（pip） | triton | XLA 或 torch |
| 版本对齐风险 | **高** | 低 | 低 | 高 |
| 实现难度 | 高 | 中 | 中 | 高 |
| 迭代速度 | 慢 | 快 | 中 | 慢 |
| 融合能力 | 高（可做通用 pass） | 中（kernel 内融合） | 低（无跨 op 融合） | 取决于 XLA |
| 动态 shape | 中（手写） | 好（tile + mask） | 中（多档变体） | 取决于上游 |
| 控制流 | 中（scf + host） | 中（scf + host） | 中（host） | 取决于上游 |
| 输出契约稳定性 | 取决于 Triton 版本 | 取决于 Triton 版本 | 不适用 | 取决于 Triton 版本 |
| 与目标契合度 | 最高 | 高 | 低 | 低 |
| MVP 预估 | 2–3 人月 | 3–5 人周 | 4–6 人周（10 个 op） | 4–8 人周（打通链路） |
| 长期可维护性 | 高（若版本稳定） | 中高 | 中 | 低 |

> 工作量为**【推断】**，假设 1–2 名熟悉编译器的工程师，且已有 ONNX/Triton 环境。

---

## 5. 关键技术难点

### 5.1 算子语义映射：TTIR 的表达能力缺口

**问题**：TTIR 的 `tt` op 数量有限（本机 3.6.0 为 50 个），没有 conv/pool/softmax/slice/pad/resize/topk/nms。ONNX 的 200+ 算子必须分解。

**建议**：

- 建立一张“ONNX op → TTIR 原语组合”的 lowering 表，按优先级实现：
  - 一级：MatMul/Gemm → `tt.dot`；Add/Mul/Sub/Div/Relu/GELU/Sigmoid/Tanh → `arith`/`math`；Softmax → `tt.reduce` 序列；Reshape/Transpose → `tt.reshape`/`tt.trans`。
  - 二级：ReduceMean/Sum/Max/Min/Prod → `tt.reduce`；BatchNorm（inference）→ elementwise；Conv → im2col + `tt.dot`；MaxPool/AveragePool → 滑窗 + `tt.reduce`。
  - 三级：Gather → `tt.gather`；ScatterAdd → `tt.atomic_rmw`；CumSum → `tt.scan`。
  - 暂不支持：NonZero、Unique、TopK、NMS、Resize（任意插值）、动态控制流。
- 每个 lowering 都要有单算子测试与数值对拍。

### 5.2 2 的幂与 1M 元素上限

**问题**：Triton block 的每一维必须是 2 的幂，且总元素数 ≤ 1,048,576。前端由 `validate_block_shape` 强制；后端 layout 推导也大量假设 2 的幂。【源码】`_utils.py:62-69,11`、`LinearLayoutConversions.cpp:1186-1190`

ONNX 的逻辑形状经常不是 2 的幂：28×28、784、1000、300、7×7。

**建议**：

- 区分 **logical shape**（ONNX 语义，运行时真实边界）与 **tile shape**（Triton block，编译期 2 的幂）。
- 对每个维度取 `next_pow2(min(dim, BLOCK))`；如果 dim 大于 BLOCK，则用 grid + mask 分块。
- 所有 load/store 带 mask；归约的 mask 填充值要按算子语义选择（sum→0，max→-inf，min→+inf，prod→1）。
- Flatten/Reshape 不要盲目用 `tt.reshape`：如果 784 不是 2 的幂，应该把 buffer 当作一维线性内存，用新的 tile shape（如 1024）加 mask 读取，逻辑上等价于 reshape。
- 大张量必须 tile；不要让一个 block 超过 1M 元素。

### 5.3 kernel 粒度与融合

**问题**：ONNX 是粗粒度图，TTIR 是 kernel 级；一个 ONNX 模型需要切成多个 `tt.func`。切得太细 → launch 开销大、中间张量多；切得太粗 → 单 kernel 无法表达。

**建议**（按 MVP 递进）：

1. **Level 0**：一个 ONNX 节点一个 kernel（shape-only 算子除外）。正确性优先，性能最差。
2. **Level 1**：elementwise 链融合进消费者；matmul/conv 的 epilogue 融合（bias、relu、gelu）。这是 MLP/CNN 的主要收益来源。
3. **Level 2**：softmax/layernorm 的整段融合（reduce + elementwise）。
4. **Level 3**：跨 matmul 融合（attention 等）暂不做。

融合合法性：

- 生产者只有一个消费者（或复制计算可接受）；
- 迭代空间一致（或可广播）；
- 不跨越 reduction 边界（除非专门处理）；
- 不引入跨 block 依赖；
- 中间结果不需要被图输出。

### 5.4 grid / block 映射

**问题**：TTIR 本身没有 grid，grid 是 launch 参数；`tt.get_program_id` 只有 x/y/z 三个轴。

**建议**：

- 在 `KernelPlan` 里显式记录 grid 表达式，写入 manifest。
- 常见映射：
  - elementwise：`grid = (cdiv(numel, BLOCK), 1, 1)`
  - matmul：`grid = (cdiv(M, BM) * cdiv(N, BN), 1, 1)` 或 2D `(cdiv(M,BM), cdiv(N,BN), 1)`
  - reduction：一个 program 处理一个输出行/切片
  - conv：`grid = (cdiv(N*P*Q, BLOCK), cdiv(K, BLOCK_K), 1)` 或 3D
- 高维 grid 要 flatten 到 3 维以内。
- 记录 `num_warps`/`num_stages` 到 manifest；它们不是 TTIR 的一部分，但影响可执行性。

### 5.5 动态 shape

**问题**：Triton 的 block 形状是编译期常量；ONNX 的动态 dim 是运行时值。

**建议**：

- rank 必须静态；动态 dim 作为 `i32` 标量参数。
- host 侧用实际 shape 计算 grid；kernel 内用 `offs < dim` 生成 mask。
- K 维动态：用 `scf.for` 的动态上界 + 每次 `tt.dot` 累加；K 不是 2 的幂时最后一次迭代 mask。
- M/N 动态：tile 固定，grid 运行时算，边界 mask。
- 动态 reshape：MVP 不支持；可以退化为“按线性内存重新解释 + 新 tile shape”，但要求目标形状可静态推导。
- 动态控制流：MVP 不支持；常量 If 折叠。

### 5.6 layout / stride

**问题**：ONNX 张量在语义上是 row-major，但 Transpose/Reshape/Slice/Concat 会产生非连续视图。Triton kernel 接收的是裸指针 + 手写索引。

**建议**：

- 内部 Graph IR 为每个张量记录 `shape`、`strides`、`offset`、`is_contiguous`。
- MVP 只支持 contiguous 张量；遇到非连续视图，插入一个 copy kernel（materialize）。
- 后续版本支持任意 stride：kernel 参数传 stride，索引算术用 stride 乘加；性能可能下降。
- `tt.trans` 是逻辑转置，不改变内存布局；如果后续 kernel 期待连续内存，需要显式 copy 或调整索引。

### 5.7 reduction

**问题**：`tt.reduce` 一次只能沿一个静态 axis 归约；多轴归约、非末轴归约、NaN 语义都要处理。

**建议**：

- 单轴归约直接映射；多轴归约拆成多次 `tt.reduce` 或先 reshape 合并被归约的维度（如果维度乘积是 2 的幂）。
- 非末轴归约可能触发布局转换，性能差；考虑先 `tt.trans` 把归约轴移到最后。
- 注意 `max` 的 NaN 语义：ONNX 的 ReduceMax/MaxPool 对 NaN 的处理与 `arith.maxnumf`/`maximumf` 不同，必须按 ONNX 规范选对 op。
- 对 softmax/layernorm，用两趟 reduce（max/sum）或 Welford；数值稳定性优先。

### 5.8 conv / pool

**问题**：TTIR 没有 conv/pool；ONNX Conv 的语义组合很多（pads、strides、dilations、group、auto_pad、bias）。

**建议**：

- **im2col + matmul**：把 Conv 变成 `(N*P*Q, C*KH*KW) × (C*KH*KW, K)` 的 GEMM；实现简单，复用 matmul 的 tile/流水线；缺点是 im2col buffer 内存膨胀。
- **implicit GEMM**：在 kernel 内按索引现算 im2col，避免物化；性能更好，实现更复杂。
- **直接卷积**：小 kernel（LeNet 的 5×5）可以接受；用嵌套循环 + `tt.load` + FMA。
- Pooling：滑窗 load + `tt.reduce`；stride/padding 用 mask；MaxPool 注意 NaN。
- MVP 先做 im2col + matmul，保证正确性；性能优化留到后续。

### 5.9 reshape / transpose

**问题**：`tt.reshape` 要求元素数相同，且 block 形状必须是 2 的幂；ONNX 的 reshape 目标形状经常不是 2 的幂。

**建议**：

- Transpose：`tt.trans` 直接支持静态 permutation。
- Reshape/Flatten/Squeeze/Unsqueeze：
  - 如果源/目标维度都是 2 的幂，可以用 `tt.reshape`。
  - 否则不要物化 reshape；把 buffer 当线性内存，用新的 tile shape + 索引映射读取。
  - 对于 `Squeeze/Unsqueeze` 这类只改 rank 的，优先用 `tt.expand_dims` 或索引算术，不产生数据移动。
- 需要跨 kernel 的 reshape，可以在 manifest 里记录“同一 buffer 的两种 view”，避免 copy。

### 5.10 控制流

**问题**：If/Loop/Scan 的子图可能包含任意算子；Triton kernel 不能 launch 子 kernel。

**建议**：

- 常量 cond 的 If：转换期折叠。
- 动态 cond、分支都是 elementwise：生成一个 kernel，用 `scf.if` 或 `tl.where`。
- 动态 cond、分支含 matmul/conv：拆成两个 kernel，host 侧按 cond 调度。
- Loop：常量 trip count 且 body 可 tile 化 → 展开/生成一个 kernel；否则 host 循环。
- Scan：只有结合律成立的扫描用 `tt.scan`；一般 body 用 host 循环。
- MVP 明确不支持动态 Loop/Scan，在文档和错误信息里写清楚。

### 5.11 数据类型与精度

**问题**：ONNX 支持 f32/f16/bf16/f64/i8/i16/i32/i64/bool/fp8 等；Triton 的支持矩阵不同，且默认精度策略会影响数值。

**建议**：

- 一级支持：f32、f16、bf16、i8、i32、i64、bool。
- f64：Triton 支持但性能差；MVP 可拒绝或软件模拟。
- fp8：后续版本。
- **f32 matmul 默认 tf32**：与 ONNX Runtime 对拍时必须设 `input_precision="ieee"`（或 `TRITON_F32_DEFAULT=ieee`），否则误差可达 1e-3。【源码】`/root/triton/third_party/nvidia/backend/compiler.py:118`
- `math.exp` 等超越函数可能是近似实现；对严格数值要求的模型，使用 libdevice 的精确版本或放宽容差。
- int64 索引：Triton 的指针偏移常用 i32；在安全范围内 cast 到 i32，超出则用 i64，并在文档中说明限制。

### 5.12 边界 / padding / NaN

**问题**：mask 的填充值、padding 的常量、NaN 传播都会影响正确性。

**建议**：

- 为每种归约定义 neutral element：sum→0，prod→1，max→-inf，min→+inf，logical and→true，or→false。
- Conv/Pool 的 padding 用 mask + 常量填充；注意 `auto_pad=SAME_UPPER/SAME_LOWER` 的偏移计算。
- NaN：明确 ONNX 算子的 NaN 语义；不要默认 `maxnumf`。
- 测试用例要包含 NaN/Inf/极大值/全零输入。

### 5.13 initializer / 常量

**建议**：

- 权重外置为 blob；kernel 参数传指针。
- 小常量（如 axis、epsilon、alpha/beta）可以作为 constexpr 或标量参数。
- 常量折叠：能在转换期算出的子图（Shape/ConstantOfShape/简单算术）直接折叠，减少 kernel 数量。
- 注意 ONNX 的 `Constant` 节点可能是大张量；不要把它变成 TTIR 常量。

### 5.14 多 kernel 编排

**建议**：

- 生成 DAG 的拓扑序；每个 kernel 记录输入/输出 buffer。
- 做 liveness 分析，复用中间 buffer。
- 输出 manifest；运行时按 manifest 加载 `.ttir`/cubin、分配显存、依次 launch。
- 对动态 shape，manifest 里记录 grid 的符号表达式（如 `cdiv(batch, 64)`），运行时求值。

### 5.15 验证与性能

**建议**：

- 正确性：与 onnxruntime 对拍（见 §6.4）。
- 编译：每个 `.ttir` 都用 `ir.parse_mlir_module` + `triton.compile` 验证。
- 性能：先保证正确，再用 `triton.testing.do_bench` 或 CUDA event 测；与 ORT 的对应 op 比较。
- 回归：每次 Triton 升级都跑完整的 IR golden + 数值测试。

---

## 6. MVP 建议

### 6.1 先支持哪些算子和模型

**模型范围**：

1. **单 MatMul**：最小闭环，验证 ONNX 解析 → TTIR → cubin → 数值对拍。
2. **2–3 层 MLP**：`Gemm` + `Add` + `Relu`/`Gelu`，验证 elementwise 融合与多 kernel 编排。
3. **Softmax / LayerNorm 子图**：验证 reduction 与数值稳定性。
4. **LeNet 风格的 CNN**：`Conv` + `MaxPool` + `BatchNorm`（inference）+ `Flatten` + `Gemm`，验证 conv/pool/reshape。

**算子白名单（MVP）**：

- 计算：`MatMul`、`Gemm`、`Add`、`Sub`、`Mul`、`Div`、`Relu`、`Gelu`、`Sigmoid`、`Tanh`、`Softmax`、`ReduceMean`、`ReduceSum`、`ReduceMax`
- 形状：`Reshape`、`Flatten`、`Squeeze`、`Unsqueeze`、`Transpose`、`Concat`（末维）
- CNN：`Conv`（2D，group=1 起步）、`MaxPool`、`AveragePool`、`GlobalAveragePool`、`BatchNormalization`（inference）
- 其他：`Cast`、`Clip`、`Where`、`Constant`、`Identity`

**明确不支持（MVP）**：动态 rank、动态 Loop/Scan、NonZero、Unique、TopK、NMS、任意 Resize、动态 reshape、量化算子。

### 6.2 建议的模块划分

```text
ox2tt/
  ox2tt/
    ir.py                  # GraphIR / Value / Node / TensorType / Layout
    importers/
      onnx_importer.py     # onnx → GraphIR（check/infer_shapes/version_converter/inline）
    analysis/
      shape.py             # 符号维度、广播、rank 推断
      constfold.py         # 常量折叠 / DCE
      partition.py         # kernel 划分 + elementwise 融合
      liveness.py          # 中间 buffer 复用
    lowering/
      registry.py          # ONNX op → emitter 的注册表
      elementwise.py
      matmul.py
      reduction.py
      conv.py
      shape_ops.py
    codegen/
      python_emitter.py    # KernelPlan → Triton Python 源码
      ttir_extract.py      # ASTSource.make_ir + make_ttir，或 triton.compile
      manifest.py
    runtime/
      launcher.py          # 可选：按 manifest 启动 kernel
    cli.py
  tests/
    ops/                   # 单算子数值测试
    models/                # MLP / LeNet 端到端
    ir_golden/             # TTIR FileCheck
```

**关键接口（建议）**：

```python
# importers/onnx_importer.py
def load(path: str, target_opset: int = 17) -> GraphIR: ...

# analysis/partition.py
def partition(graph: GraphIR) -> list[KernelPlan]: ...

# codegen/python_emitter.py
class TritonSourceEmitter:
    def emit(self, plan: KernelPlan) -> KernelArtifact: ...

# codegen/ttir_extract.py
def extract_ttir(source_path: str, signature: dict, constexprs: dict,
                 target: GPUTarget, num_warps: int = 4) -> str: ...

# cli.py
# ox2tt convert model.onnx -o out/ --target sm70 --emit ttir
# ox2tt verify out/ --model model.onnx --against onnxruntime
```

### 6.3 关键实现入口

- Triton 前端入口：`ASTSource`（`/root/triton/python/triton/compiler/compiler.py:52`）、`ast_to_ttir`（`/root/triton/python/triton/compiler/code_generator.py:1620`）。
- TTIR-only 提取：`ASTSource.make_ir` + `CUDABackend.make_ttir`（`/root/triton/third_party/nvidia/backend/compiler.py:235`）。
- TTIR 校验：`ir.parse_mlir_module` + `module.verify()`（`/root/triton/python/src/ir.cc:764`）。
- TTIR → cubin：`triton.compile(path, target=..., options=...)`（`compiler.py:226`）。
- dump：`TRITON_KERNEL_DUMP`/`TRITON_DUMP_DIR`（`knobs.py:345,358`）。
- 产物整洁：`TRITON_DISABLE_LINE_INFO=1` 会去掉 module 里的 location 信息（`knobs.py:365`）；否则 TTIR 会带大量 `#loc`，不利于 golden diff。

### 6.4 数值正确性验证

**基准**：onnxruntime（本机 1.23.2）作为 golden reference。对每个测试：

1. 用固定 seed 生成输入（numpy）。
2. ORT `InferenceSession` 跑 CPU（或 CUDA EP）得到参考输出。
3. Triton 路径跑同一输入，得到输出。
4. 比较：
   - `max_abs_err`、`max_rel_err`、`allclose`；
   - NaN/Inf 位置一致；
   - 对 fp32：`rtol=1e-5, atol=1e-5`（matmul 可用 1e-4，取决于 K）；
   - 对 fp16/bf16：`rtol=1e-2, atol=1e-2`；
   - f32 matmul 必须 `input_precision="ieee"`，否则 tf32 会造成 1e-3 量级差异。
5. 输入分布覆盖：全零、全一、随机正态、大幅值、含 NaN/Inf。
6. 每个 op 至少 3 组 shape；每个模型至少 5 个 seed。

**分层验证**：

- 单算子：用 `onnx.helper` 构造单节点模型。
- 子图：手工构造 2–5 个节点的组合。
- 整模型：MLP、LeNet。
- IR 级：对每个生成的 `.ttir` 做 parse + verify；对关键 op 做 FileCheck golden。
- 执行级：用 Triton 的 launcher 或 cuda-python driver API 启动；本机无 torch，可直接用 cuda-python 的 driver API 管理 primary context 并 launch。【推断】

### 6.5 里程碑

| 里程碑 | 目标 | 验收标准 |
|---|---|---|
| M0 | 单 MatMul 闭环 | `.onnx` → 一个 `.ttir` → `triton.compile` → sm70 cubin；与 ORT 误差 ≤ 1e-4（ieee） |
| M1 | elementwise + 融合 + MLP | 2–3 层 MLP 端到端；bias/relu 融合进 matmul epilogue；误差 ≤ 1e-5 |
| M2 | reduction + Softmax/LayerNorm | softmax/layernorm 单 kernel；多 seed 通过；NaN 语义正确 |
| M3 | CNN（LeNet） | Conv（im2col）+ MaxPool + BN + Flatten + Gemm；端到端误差 ≤ 1e-4 |
| M4 | 动态 batch + manifest + runtime | 同一 TTIR 支持多个 batch；manifest 可驱动 launch；多 shape 回归 |
| M5 | 工程化 | opset 归一化、错误诊断、IR golden、CI、文档 |

> 时间估算：M0 约 1–2 周；M1 约 2–3 周；M2 约 2 周；M3 约 3–5 周；M4 约 2–3 周；M5 持续。【推断】

### 6.6 验收标准（建议）

- 所有生成的 `.ttir` 都能被 `ir.parse_mlir_module` 解析并通过 `module.verify()`。
- 所有 `.ttir` 都能在目标 GPU 上编译为 cubin 并 launch。
- 白名单算子的单算子测试全部通过；MLP/LeNet 端到端通过。
- 失败时有明确诊断：不支持的 op、动态 rank、非 2 的幂且无法 pad 的形状、控制流等。
- 产物包含 manifest，且 manifest 能驱动一次完整的推理。

---

## 7. 推荐路线与主要风险

### 7.1 推荐

**采用“B-first, A-ready”的路线：**

1. **v0.x 用方案 B**：ONNX → 自有 Graph IR → kernel 划分/融合 → 生成 Triton Python DSL → 用 Triton 前端产出 TTIR。
2. **输出契约**：每个 kernel 一个 `.ttir` + `manifest.json`；固定 Triton 版本；记录 target/num_warps/num_stages。
3. **保留方案 A 的接口**：把 `KernelPlan → TTIR` 做成一个可替换的 `TtirBackend` 接口。未来如果需要“无 Python 运行时”或更强的跨 op 融合，再实现 MLIR pass 后端；Graph IR、partition、lowering 表、测试全部复用。
4. **不要一开始绑定 onnx-mlir**：ONNX 解析/形状推断用 `onnx` Python 包即可；onnx-mlir 作为参考实现和未来的 A 前端候选，而不是 v0 的硬依赖。
5. **固定 Triton 3.8.0**（本机已安装），不要混用 `/root/triton` 的 3.6.0 源码；如果要从源码构建，先统一版本再谈。

### 7.2 为什么推荐 B

- **最快拿到正确 TTIR**：Triton 前端已经实现了广播、类型提升、mask、指针算术、`tt.dot` 的精度属性等；方案 A 需要重新实现这些语义。
- **避免 LLVM/MLIR 版本地狱**：本机 onnx-mlir 与 Triton 固定的是不同 LLVM 提交；方案 A 的第一道坎就是统一 LLVM，而不是 ONNX 转换本身。
- **仍然输出真正的 TTIR**：不是“伪 IR”，而是 Triton 自己生成的 canonical TTIR；本报告已实测该 TTIR 可被 `triton.compile` 重新编译到 cubin。
- **工程可迭代**：每个 op 的 lowering 是一个小函数，测试和调试成本低；适合 MVP 和快速扩展。
- **保留了升级到 A 的路径**：Graph IR、partition、lowering 表与后端解耦；A 只替换“KernelPlan → TTIR”这一段。

### 7.3 为什么不是 A 或 C

- **A 不是不成立，而是“先难后易”**：XLA/NKCT 证明了可行，但需要统一的 LLVM/Triton 栈和大量 C++ lowering；对“从 0 到 1 输出 TTIR”的目标，前期风险过高。
- **C 不满足目标**：它是“Triton 算子库 + 调度器”，不是 IR 转换器；除非产品目标其实是“在 Triton 上跑 ONNX”，否则不应作为主线。

### 7.4 风险清单

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| TTIR 随 Triton 版本漂移 | 高 | 高 | 固定版本；manifest 记录版本；CI 做跨版本回归；必要时锁定到某个 Triton release |
| 2 的幂/1M 元素约束导致大量 padding，性能差 | 高 | 中 | logical shape 与 tile shape 解耦；按 op 选 tile；benchmark 驱动调参 |
| Conv/Pool 性能远低于 cuDNN | 高 | 中 | MVP 只保证正确；后续 im2col/implicit GEMM/直接卷积对比选优 |
| 动态 shape 支持不完整 | 高 | 中 | rank 静态；动态 dim 用标量 + mask；动态 reshape/控制流明确拒绝 |
| 控制流（Loop/Scan）无法通用 | 中 | 高 | 只支持常量 If；Loop/Scan 走 host 编排；文档写清限制 |
| 数值不一致（tf32、exp 近似、归约顺序） | 中 | 中 | 验证时 `input_precision=ieee`；按 dtype 设容差；NaN/Inf 用例 |
| 融合过度导致 kernel 无法表达/寄存器压力 | 中 | 中 | 融合规则保守；先 Level 0/1；有回退路径 |
| 生成 Python 源码的注入/命名/缩进 bug | 中 | 中 | 用 AST 或严格模板；生成后 `compile()` 语法检查；单测 |
| 输出契约不被下游接受 | 中 | 高 | 尽早和消费方确认；提供 manifest + 可执行验证；明确版本绑定 |
| onnx-mlir 依赖导致构建复杂（若选 A） | 高 | 高 | v0 不依赖；A 阶段再评估 |

### 7.5 需要用户决策的点

1. **输出消费者是谁**：TTIR 是给内部 pipeline 用，还是要交给外部工具/团队？如果是后者，必须定义稳定契约与版本策略。
2. **目标硬件**：只支持 NVIDIA sm70（本机 V100）？还是 sm80+/AMD？这会影响是否选择 FlagTree 多后端，以及是否需要在 TTIR 阶段就考虑布局。
3. **Triton 版本策略**：固定 pip 3.8.0，还是从源码构建特定 commit？是否接受“TTIR 随版本变化”？
4. **转换时是否允许依赖 Python**：方案 B 需要 Python + Triton 包；如果产品要求自包含 C++ 二进制，则必须走 A。
5. **动态 shape 的范围**：只需要动态 batch，还是任意维度动态？是否需要动态 reshape/控制流？
6. **opset 范围**：固定到 17/22，还是需要兼容更老/更新的模型？
7. **交付物**：只要 `.ttir`，还是需要 manifest + 可执行 runtime + cubin？
8. **性能目标**：MVP 只要求正确，还是从一开始就要对标 ONNX Runtime/cuDNN？
9. **依赖/license**：onnx-mlir（Apache-2.0）、Triton（MIT）、onnx（Apache-2.0）是否都可接受？

---

## 8. 参考来源

### 8.1 本地源码

Triton：

- `/root/triton/include/triton/Dialect/Triton/IR/TritonDialect.td:7`（dialect 名 `tt`）、`:31-37`（dependent dialects）
- `/root/triton/include/triton/Dialect/Triton/IR/TritonOps.td:741,748,774,781,1161,1262`（Reduce/Scan/Func/Return）
- `/root/triton/lib/Dialect/Triton/IR/Utility.cpp:109-111`（`isKernel` = public）
- `/root/triton/lib/Dialect/Triton/IR/Ops.cpp:218,480,575,626,877`（Trans/Reduce/Scan/Reshape verifier）
- `/root/triton/python/triton/compiler/compiler.py:52,87,92,108,111,226,237,292,426`
- `/root/triton/python/triton/compiler/code_generator.py:1620`
- `/root/triton/third_party/nvidia/backend/compiler.py:19,118,235,252,349,447,473,556-565`
- `/root/triton/python/triton/knobs.py:345,358`
- `/root/triton/python/triton/_utils.py:11,62-69`
- `/root/triton/python/triton/language/semantic.py:589,1532,1535-1540`
- `/root/triton/python/src/ir.cc:680-690,764`
- `/root/triton/README.md:237-262`
- `/root/triton/cmake/llvm-hash.txt:1`

onnx-mlir / ONNX：

- `/root/onnx-mlir/third_party/onnx/onnx/onnx.proto3:134,201,220,446,561,604,913`
- `/root/onnx-mlir/third_party/onnx/docs/IR.md:222,227,257,278,303,309,315,444,474,491,495-497`
- `/root/onnx-mlir/third_party/onnx/docs/Operators.md:15826,17793,31921`
- `/root/onnx-mlir/third_party/onnx/docs/ShapeInference.md:56-62`
- `/root/onnx-mlir/src/Conversion/ONNXToLinalg/Math/MatMul.cpp:7-14`
- `/root/onnx-mlir/src/Conversion/ONNXToLinalg/NN/Conv.cpp:9-16`
- `/root/onnx-mlir/src/Conversion/ONNXToLinalg/ConvertONNXToLinalg.cpp:41,54-57`
- `/root/onnx-mlir/docs/SupportedONNXOps-cpu.md:6`
- `/root/onnx-mlir/docs/BuildOnLinuxOSX.md:18`

其他仓库：

- `/root/triton-shared/README.md:1-12`
- `/root/triton-linalg/README.md:1-10`
- `/root/FlagTree/README.md`（多后端表）
- `/root/PaddleTritonOps/README_cn.md`、`/root/PaddleTritonOps/OperatorList.md`
- `/root/nkct/CLAUDE.md`、`/root/nkct/src/AffineToTriton.cpp:711,795-837,1100-1140`
- `/root/llvm-project/mlir/include/mlir/Transforms/DialectConversion.h:41,1078,1517`
- `/root/llvm-project/mlir/include/mlir/IR/PatternMatch.h:238,822`

### 8.2 官方文档与规范

- ONNX IR 规范：https://onnx.ai/onnx/repo-docs/IR.html
- ONNX IR.md（GitHub）：https://github.com/onnx/onnx/blob/main/docs/IR.md
- ONNX Operators：https://github.com/onnx/onnx/blob/main/docs/Operators.md
- ONNX Shape Inference：https://github.com/onnx/onnx/blob/main/docs/ShapeInference.md
- onnx.proto3：https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3
- Triton MLIR Dialects：https://triton-lang.org/main/dialects/dialects.html
- `tt` Dialect：https://triton-lang.org/main/dialects/TritonDialect.html
- TritonOps：https://triton-lang.org/main/dialects/TritonOps.html
- Triton Python API：https://triton-lang.org/main/python-api/triton.html
- Triton GitHub README（dump/override）：https://github.com/triton-lang/triton/blob/main/README.md
- TritonOps.td：https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonOps.td
- MLIR Dialect Conversion：https://mlir.llvm.org/docs/DialectConversion/
- MLIR Defining Dialects：https://mlir.llvm.org/docs/DefiningDialects/
- MLIR Rewrite Patterns：https://mlir.llvm.org/docs/RewritePattern/

### 8.3 社区项目与相关工作

- microsoft/triton-shared：https://github.com/microsoft/triton-shared
- Cambricon/triton-linalg：https://github.com/Cambricon/triton-linalg
- flagos-ai/FlagTree：https://github.com/flagos-ai/FlagTree
- onnx/onnx-mlir：https://github.com/onnx/onnx-mlir
- triton-lang/triton：https://github.com/triton-lang/triton
- XLA Triton fusion emitter：https://github.com/openxla/xla/blob/main/xla/backends/gpu/codegen/triton/fusion_emitter.cc
- CKTI 论文：https://doi.org/10.1145/3797905.3800551
- ONNX Runtime Triton kernel 文档：https://github.com/microsoft/onnxruntime/blob/main/docs/ORT_Use_Triton_Kernel.md
- ONNX Runtime integrate triton PR：https://github.com/microsoft/onnxruntime/pull/15862
- IBM/onnxmlir-triton-backend：https://github.com/IBM/onnxmlir-triton-backend
- KernelLens：https://zenodo.org/records/23194523
- onnx2torch：https://github.com/ENOT-AutoDL/onnx2torch

---

## 附录 A：本次实测记录

环境：Tesla V100S-PCIE-32GB（sm70），Triton 3.8.0（`/root/miniconda3/envs/onnxenv`），onnxruntime 1.23.2，onnx 1.21.0，CUDA 12.6 的 `ptxas`。无 torch。

### A.1 仅用前端 + `make_ttir` 产出 TTIR（不跑 ptxas、不启动 GPU）

输入：一个 3 参数的 elementwise add kernel。

调用链：

```python
target = GPUTarget("cuda", 70, 32)
backend = make_backend(target)
options = backend.parse_options({})
ctx = ir.context(); ir.load_dialects(ctx); backend.load_dialects(ctx)
src = ASTSource(fn, signature={...}, constexprs={"BLOCK": 128})
module = src.make_ir(target, options, backend.get_codegen_implementation(options),
                     backend.get_module_map(), ctx)
ttir = backend.make_ttir(module, {}, options, 70)
```

结果：成功输出 `tt.func public @add_kernel` 的 TTIR 文本，包含 `tt.get_program_id`、`tt.make_range`、`tt.splat`、`tt.addptr`、`tt.load`、`arith.addf`、`tt.store`。没有调用 `ptxas`，也没有 GPU launch。

### A.2 TTIR 文本重新解析

用 `ir.parse_mlir_module('add.ttir', ctx)` 解析 A.1 的输出：

```
entry: add_kernel
sig: ['*f32', '*f32', '*f32', 'i32']
```

说明 TTIR 文本可以被 Triton 的 IRSource 正常识别，entry function 与签名都正确。

### A.3 TTIR → TTGIR → LLIR → PTX → cubin

```python
k = triton.compile('/tmp/ttir_probe/add.ttir',
                   target=GPUTarget("cuda", 70, 32), options={})
```

结果：

```
asm keys: ['cubin', 'llir', 'ptx', 'ttgir', 'ttir']
name: add_kernel
num_warps: 4
cubin bytes: 4968
```

说明：`.ttir` 可以作为稳定的“编译输入”，Triton 会从 TTGIR 阶段继续编译，不需要重新跑 TTIR 阶段的优化 pass。这与 `compiler.py:292-295` 的 `first_stage` 逻辑一致。

### A.4 复现实验的注意事项

- `@triton.jit` 必须定义在真实的 `.py` 文件中；用 `exec`/heredoc 动态定义会触发 `inspect.getsourcelines` 失败（`ValueError: @jit functions should be defined in a Python file`）。生成器必须把源码写到文件再 import。
- `triton.compile` 的 `target` 参数可以避免依赖 `driver.active`；但完整的 cubin 编译仍需要 `ptxas` 和 NVPTX 后端。
- 如果只产出 TTIR，用 A.1 的路径可以避免 `ptxas`；如果要验证可执行性，用 A.3。

---

## 附录 B：缩略语与术语表

| 术语 | 含义 |
|---|---|
| ONNX | Open Neural Network Exchange，模型交换格式 |
| opset | ONNX 算子集版本，以 `(domain, version)` 标识 |
| initializer | ONNX 图中的常量张量（通常是权重） |
| SSA | Static Single Assignment，ONNX 图与 MLIR 都要求 |
| MLIR | Multi-Level Intermediate Representation，LLVM 的编译器基础设施 |
| Dialect | MLIR 中的一组 op/type/attribute 定义 |
| TTIR | Triton IR，`tt` dialect，硬件无关的 tile 级 IR |
| TTGIR | Triton GPU IR，带 layout/共享内存/MMA 的 GPU 相关 IR |
| LLIR | LLVM dialect IR |
| PTX | NVIDIA 的虚拟指令集 |
| cubin | NVIDIA GPU 可加载二进制 |
| block / tile | Triton 的一次 program 处理的张量块 |
| grid | kernel launch 的 program 数量与维度 |
| mask | 边界屏蔽，避免越界访问 |
| layout | TTGIR 中张量元素在线程/寄存器/共享内存间的分布 |
| `num_warps` | 每个 program 使用的 warp 数 |
| `num_stages` | 软件流水线的 stage 数 |
| constexpr | Triton kernel 的编译期常量参数 |
| `tt.func public` | TTIR 中的 kernel 入口函数 |
| manifest | 本报告建议的产物清单，记录 kernel/grid/参数/依赖 |
