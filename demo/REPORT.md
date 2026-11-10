# ONNX → TTIR 转换 demo 技术报告

- 日期：2026-11-10
- 主程序：`demo/onnx_to_ttir.py`
- 示例模型生成器：`demo/make_sample_model.py`
- 本文档目的：把这个 demo 的流程、每一步的输入输出、实际运行结果讲清楚

---

## 1. 一句话总结

这个 demo 读取一个**真实的 `.onnx` 文件**，确认它只包含一个二维 MatMul，然后自动生成 Triton kernel 源码，再调用 Triton 前端，最后写出一个 `.ttir` 文件。

它覆盖的完整流程是：

```text
读取 .onnx
  → 检查模型、取出矩阵形状
  → 计算分块和启动网格
  → 生成 Triton Python 源码
  → 调用 Triton 前端
  → 写出并检查 TTIR
```

---

## 2. 怎么运行

### 2.1 生成示例模型

```bash
cd /root/ox2tt
python demo/make_sample_model.py
```

默认生成：

```text
demo/out/sample_matmul.onnx
```

模型内容：

```text
A: f32 [130, 70]
B: f32 [70, 50]
C: f32 [130, 50]

C = A @ B
```

### 2.2 转换成 TTIR

```bash
python demo/onnx_to_ttir.py demo/out/sample_matmul.onnx \
  -o demo/out/sample_matmul.ttir --verbose
```

输出：

```text
demo/out/sample_matmul.ttir
```

如果加上 `--emit-intermediates`，还会额外保存：

```text
demo/out/sample_matmul.graph.json   从 ONNX 读出来的图信息
demo/out/sample_matmul.plan.json    分块、网格、参数类型等执行计划
demo/out/sample_matmul.kernel.py    自动生成的 Triton Python 源码
```

### 2.3 环境

| 项目 | 版本 / 说明 |
|---|---|
| Python | 3.10.21 |
| Triton | 3.8.0 |
| onnx | 1.21.0 |
| GPU | Tesla V100S，计算能力 7.0（sm70） |

不需要 torch。这个 demo 只跑 Triton 前端，不需要启动 GPU。

---

## 3. 支持的输入范围

这个 demo 故意只支持一小类模型：

| 项目 | 支持情况 |
|---|---|
| 算子数量 | 只能有 1 个节点 |
| 算子类型 | 只能是 `MatMul` |
| 矩阵维度 | 只能是二维 |
| 形状 | 必须是静态的，不能有动态维度 |
| 数据类型 | 只能是 f32 |
| 输出 | 只输出 TTIR，不生成 cubin，不执行 |

不满足这些条件时，程序会给出明确的错误信息，而不是悄悄算错。

例如，输入一个只有 `Relu` 的模型，会得到：

```text
错误：这个 demo 只支持 MatMul，但模型里的算子是 Relu。
```

---

## 4. 整体流程

```text
┌──────────────────────────────────────────────┐
│ 第 1 步：读取并检查 ONNX                      │
│ 输入：.onnx 文件                              │
│ 输出：A、B、C 的名字、类型、形状、opset 版本   │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│ 第 2 步：计算执行计划                         │
│ 输入：矩阵形状                                │
│ 输出：分块大小、启动网格、参数类型             │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│ 第 3 步：生成 Triton Python 源码              │
│ 输入：执行计划                                │
│ 输出：一个真实的 .py 文件                     │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│ 第 4 步：调用 Triton 前端                     │
│ 输入：Triton Python 源码                      │
│ 输出：TTIR 文本                               │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│ 第 5 步：验证并写出                           │
│ 输入：TTIR 文本                               │
│ 输出：通过合法性检查的 .ttir 文件              │
└──────────────────────────────────────────────┘
```

---

## 5. 每一步在做什么

### 5.1 第 1 步：读取并检查 ONNX

对应代码：`load_and_validate_model`

**输入**

一个真实的 `.onnx` 文件。

**做了什么**

1. 用 `onnx.load` 读文件。
2. 用 `onnx.checker.check_model` 检查模型结构是否合法。
3. 让 ONNX 自带的形状推断跑一遍，尽量补齐中间结果的形状。
4. 检查图里是不是只有一个节点。
5. 检查这个节点是不是 `MatMul`。
6. 检查两个输入是不是二维、静态、f32。
7. 检查形状能不能相乘：`A[M,K] @ B[K,N] = C[M,N]`。

**实际输出**

```text
算子: MatMul（opset 17）
A   : f32 [130, 70]
B   : f32 [70, 50]
C   : f32 [130, 50]
```

**这一步为什么需要**

转换器首先要保证“输入是它能处理的模型”。如果模型里有多个算子、动态形状或者别的数据类型，应该在进入后面步骤之前就明确拒绝，而不是生成一个错误的 kernel。

---

### 5.2 第 2 步：计算执行计划

对应代码：`build_kernel_plan`

**输入**

矩阵形状 `M=130, K=70, N=50`。

**做了什么**

决定三件事：

#### ① 分块大小

```text
BLOCK_M = 64
BLOCK_N = 32
BLOCK_K = 32
```

Triton 要求分块的每一维是 2 的幂，所以用 64、32、32。

#### ② 启动网格

行方向：

```text
ceil(130 / 64) = 3 个块
```

列方向：

```text
ceil(50 / 32) = 2 个块
```

所以 `grid = (3, 2, 1)`，一共 6 个 program。

沿 K 方向需要：

```text
ceil(70 / 32) = 3 轮
```

#### ③ 参数类型和编译期常量

`signature` 记录 kernel 每个“运行时参数”的类型：

```json
{
  "a_ptr": "*fp32",
  "b_ptr": "*fp32",
  "c_ptr": "*fp32",
  "M": "i32",
  "N": "i32",
  "K": "i32",
  "stride_am": "i32",
  "stride_ak": "i32",
  "stride_bk": "i32",
  "stride_bn": "i32",
  "stride_cm": "i32",
  "stride_cn": "i32"
}
```

- `*fp32`：指向 f32 数据的指针，也就是一块显存的起始地址
- `i32`：32 位整数
- `stride_xx`：移动一行或一列时，地址要加多少

`constexprs` 记录编译期常量：

```json
{
  "BLOCK_M": 64,
  "BLOCK_N": 32,
  "BLOCK_K": 32
}
```

**实际输出**

如果加了 `--emit-intermediates`，这一步会写成：

```text
demo/out/sample_matmul.plan.json
```

**这一步为什么需要**

图只说明“要算一个矩阵乘法”；执行计划说明“具体怎么分块、开多少个 program、每个参数是什么类型”。生成源码时只需要看执行计划。

---

### 5.3 第 3 步：生成 Triton Python 源码

对应代码：`render_kernel_source` 和 `write_kernel_source`

**输入**

执行计划。

**做了什么**

按照模板生成一个真正的 Triton kernel 源文件。生成后会先用 Python 的 `compile()` 做语法检查，再写文件。

如果加了 `--emit-intermediates`，文件保存在：

```text
demo/out/sample_matmul.kernel.py
```

如果没有加，源码会写到临时目录，生成 TTIR 后自动删除。

**生成代码的核心部分**

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
```

**逐段说明**

1. 找到自己负责的块

```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
```

2. 计算这一块在真实矩阵里的行号和列号

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
```

3. 构造 A、B 的地址网格

```python
a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
```

这两行把二维下标变成显存地址。

4. 沿 K 方向分块累加

```python
for k in range(0, tl.cdiv(K, BLOCK_K)):
    ...
    acc = tl.dot(a, b, acc, input_precision="ieee")
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

`tl.dot` 就是矩阵乘加。

5. 边界遮罩

```python
a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
```

因为 130 不是 64 的整数倍、50 不是 32 的整数倍，最后一块会超出真实矩阵范围。遮罩用来挡住越界位置。

**这一步为什么需要**

Triton 前端已经帮我们处理了广播、类型提升、指针算术、mask、`tt.dot` 属性等细节。生成 Python 源码比直接拼 TTIR 文本更稳，也更容易检查。

**需要注意**

`@triton.jit` 函数必须写在真实的 `.py` 文件里。Triton 要通过 `inspect` 读取函数的源码行；动态 `exec` 出来的函数没有源码文件，会报错。

---

### 5.4 第 4 步：调用 Triton 前端生成 TTIR

对应代码：`extract_ttir`

**输入**

第 3 步生成的 Triton Python 源码。

**做了什么**

```python
source = ASTSource(fn=kernel_fn, signature=plan.signature, constexprs=plan.constexprs)
module = source.make_ir(...)
ttir_module = backend.make_ttir(module, {}, options, target.arch)
ttir_text = str(ttir_module)
```

这条路径只跑 Triton 前端，不需要 ptxas，也不需要启动 GPU。

**TTIR 是什么**

TTIR 是 Triton 的第一层中间表示。它描述“每个 program 对一块数据做什么”，但还没有绑定共享内存、warp 分工等具体 GPU 细节。

可以把它理解成：

> **Python 源码的结构化中间版本：去掉了 Python 语法，但还没变成机器码。**

**实际输出的关键片段**

```mlir
tt.func public @matmul_kernel(
    %a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>,
    %M: i32, %N: i32, %K: i32, ...) {
```

对应 kernel 的参数。

```mlir
%0 = tt.get_program_id x : i32
%1 = tt.get_program_id y : i32
```

对应 `tl.program_id(0)` 和 `tl.program_id(1)`。

```mlir
%acc:3 = scf.for %k = %c0_i32 to %33 step %c1_i32 iter_args(...) {
```

这是沿 K 方向的循环。

```mlir
%66 = tt.load %a_ptrs, %58, %cst_0 : tensor<64x32x!tt.ptr<f32>>
%67 = tt.load %b_ptrs, %65, %cst : tensor<32x32x!tt.ptr<f32>>
%68 = tt.dot %66, %67, %acc_1 : tensor<64x32xf32> * tensor<32x32xf32> -> tensor<64x32xf32>
```

这三行是核心：读入 A 的小块、读入 B 的小块、做矩阵乘加。

```mlir
tt.store %42, %acc#2, %49 : tensor<64x32x!tt.ptr<f32>>
```

把结果写回 C。

---

### 5.5 第 5 步：验证并写出

对应代码：`extract_ttir` 的后半段

**做了什么**

1. 把 TTIR 文本写到目标文件。
2. 把文本重新解析一遍。
3. 调用 `verify()` 检查 IR 是否合法。

```python
parsed = ir.parse_mlir_module(str(ttir_path), context)
parsed.verify()
```

**实际输出**

```text
TTIR: 108 行，6901 个字符，已通过合法性检查
```

**这一步为什么需要**

能打印出文本，不代表文本一定合法。重新解析并调用 `verify()`，可以确认生成的 TTIR 能被 Triton 重新读入，是后续继续编译的基础。

---

## 6. 实际运行结果

### 6.1 默认分块

命令：

```bash
python demo/onnx_to_ttir.py demo/out/sample_matmul.onnx \
  -o demo/out/sample_matmul.ttir --verbose
```

结果：

| 项目 | 实际值 |
|---|---|
| 输入模型 | `demo/out/sample_matmul.onnx` |
| 输出 | `demo/out/sample_matmul.ttir` |
| MatMul | A[130,70] @ B[70,50] -> C[130,50] |
| 分块 | BLOCK_M=64，BLOCK_N=32，BLOCK_K=32 |
| 网格 | (3, 2, 1)，共 6 个 program |
| TTIR | 108 行，6901 个字符 |
| 合法性检查 | 通过 |
| 耗时 | 约 0.09–0.12 秒 |

### 6.2 换一组分块

命令：

```bash
python demo/onnx_to_ttir.py demo/out/sample_matmul.onnx \
  -o /tmp/custom.ttir --block-m 32 --block-n 32 --block-k 16
```

结果：

| 项目 | 实际值 |
|---|---|
| 分块 | BLOCK_M=32，BLOCK_N=32，BLOCK_K=16 |
| 网格 | (5, 2, 1) |
| TTIR | 109 行，6979 个字符 |
| 合法性检查 | 通过 |

这说明分块大小不是写死的，可以通过命令行调整，TTIR 也会随之变化。

---

## 7. 代码结构

```text
demo/
├── onnx_to_ttir.py       主程序
├── make_sample_model.py  生成示例模型
├── README.md             使用说明
├── REPORT.md             本报告
└── out/                  运行后生成的文件
```

主程序里的关键函数：

| 函数 | 作用 |
|---|---|
| `load_and_validate_model` | 读取 ONNX，检查是不是单个二维 MatMul |
| `build_kernel_plan` | 计算分块、网格、参数类型、编译期常量 |
| `render_kernel_source` | 根据执行计划生成 Triton Python 源码 |
| `write_kernel_source` | 检查语法并写文件 |
| `load_generated_kernel` | 从生成的 .py 文件里加载 kernel 函数 |
| `extract_ttir` | 调用 Triton 前端生成 TTIR，并验证 |
| `main` | 命令行入口，按顺序执行五步 |

---

## 8. 这次实际遇到的问题

### 8.1 Triton 的类型字符串是 `*fp32`，不是 `*f32`

第一次运行时，程序报错：

```text
KeyError: 'f32'
```

原因是 Triton 在解析参数类型时，指针类型必须写成 `*fp32`。`f32` 是 ONNX 侧使用的短名字，两者不是同一套写法。

修正后：

```python
TRITON_POINTER_TYPE = "*fp32"
```

这个坑说明：**ONNX 的类型名和 Triton 的类型名之间需要一个转换层。**

### 8.2 `@triton.jit` 必须写在真实文件里

Triton 需要读取函数的源代码行。如果函数是动态 `exec` 出来的，会找不到源码。所以程序必须：

1. 先把源码写到真实的 `.py` 文件；
2. 再用 `importlib` 加载；
3. 最后交给 Triton。

### 8.3 形状不是 2 的幂，必须写 mask

130、70、50 都不是分块的整数倍：

- A 的行方向最后一块只有 2 行有效
- C 的列方向最后一块只有 18 列有效
- K 方向最后一轮只有 6 个元素有效

如果不写 mask，就会读到或写到矩阵范围外。

### 8.4 分块大小有硬约束

- 每一维必须是 2 的幂
- `tl.dot` 的每一维至少为 16
- 单个分块最多 1,048,576 个元素
- `num_warps` 必须是 2 的幂

程序在 `build_kernel_plan` 里会对这些条件做检查。

---

## 9. 这个 demo 没有做什么

1. **没有启动 GPU**：只生成 TTIR，不执行 kernel。
2. **没有数值对比**：没有和 onnxruntime 比较计算结果。
3. **没有算子融合**：只支持单个 MatMul。
4. **没有动态形状**：所有维度必须是固定数字。
5. **没有导出权重**：如果模型里有 initializer，只会记录它的形状，不会把权重内容写到文件里。
6. **没有多 kernel 编排**：只有一个 kernel。
7. **没有性能调优**：分块大小需要手动指定，没有自动选择。

---

## 10. 下一步可以做什么

如果要把这个 demo 继续扩展，可以按这个顺序：

1. **加数值对比**
   - 用 cuda-python 分配显存、拷贝 A/B、启动 kernel
   - 和 onnxruntime 的结果比较

2. **加算子融合**
   - 把 `MatMul + Add + Relu` 融成一个 kernel

3. **加权重处理**
   - 把 initializer 导出成独立的二进制文件
   - 在运行时上传到显存

4. **加多 kernel 编排**
   - 按顺序启动多个 kernel
   - 管理中间结果的 buffer

5. **加更多算子**
   - Softmax / LayerNorm
   - Conv / Pool
   - Reshape / Transpose

---

## 11. 术语表

| 词 | 大白话解释 |
|---|---|
| ONNX | 一种通用的模型文件格式，可以把训练好的模型交给不同的推理引擎 |
| 算子（operator） | 模型里的一次基本计算，例如矩阵乘、加法、激活函数 |
| MatMul | 矩阵乘法，`C = A @ B` |
| IR | Intermediate Representation，中间表示；编译器内部用来描述程序的数据结构 |
| TTIR | Triton 的第一层中间表示；描述每个块做什么，但还没绑定具体 GPU 细节 |
| tile（分块） | 把大矩阵切成小块，每次只处理一块 |
| program | Triton 的一次程序实例；每个 program 负责输出矩阵的一小块 |
| grid | 一共要启动多少个 program |
| mask（遮罩） | 用布尔条件挡住越界的位置，避免读到或写到矩阵范围外 |
| constexpr | 编译期常量；编译时就知道数值，会被直接写进生成的代码 |
| signature（签名） | kernel 每个参数的类型清单，告诉 Triton 该怎么处理这些参数 |
| stride（步长） | 在矩阵里移动一行或一列时，地址要加多少 |
| verify | 检查 IR 是否合法；不合法会直接报错 |
| 数值对比 | 同一组输入分别交给两个实现计算，再比较输出是否一致 |

---

## 12. 最后总结

这个 demo 验证了：

1. **外部 `.onnx` 文件可以被真正读取和检查**，不是脚本内部假造一个流程。
2. **单算子模型可以完整走通 ONNX → TTIR**：读取、检查、计划、生成源码、调用 Triton 前端、写出 TTIR。
3. **输出是合法的 TTIR**：写出的文本能被重新解析，并通过 `verify()`。
4. **分块大小可调**：修改命令行参数，TTIR 会随之变化。

它还不能证明“算出来的结果是对的”，因为还没有启动 GPU 和做数值对比；但它证明了转换流程本身是通的。
