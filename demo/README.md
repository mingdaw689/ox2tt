# ONNX → TTIR 转换 demo

这是一个能在当前环境直接运行的转换器 demo：

```text
读取一个 .onnx 文件
  → 检查模型、取出矩阵形状
  → 计算分块和启动网格
  → 生成 Triton Python 源码
  → 调用 Triton 前端
  → 写出 .ttir 文件
```

## 支持范围

- 一个模型里只有一个算子：`MatMul`
- 二维矩阵、静态形状
- 数据类型：f32
- 只输出 TTIR；不启动 GPU，不生成 cubin
- 如果矩阵 B 是模型里的常量（initializer），可以转换，但只会记录它的形状，不会导出权重内容

其他情况会给出明确的错误提示。

## 环境

本 demo 在下面这个环境中验证通过：

| 项目 | 版本 / 说明 |
|---|---|
| Python | 3.10.21 |
| Triton | 3.8.0 |
| onnx | 1.21.0 |
| GPU | Tesla V100S，计算能力 7.0（sm70） |

不需要 torch。

## 快速开始

```bash
cd /root/ox2tt

# 1. 生成一个只包含 MatMul 的示例模型
python demo/make_sample_model.py

# 2. 转换成 TTIR
python demo/onnx_to_ttir.py demo/out/sample_matmul.onnx -o demo/out/sample_matmul.ttir --verbose
```

运行后，`demo/out/sample_matmul.ttir` 就是转换结果。

## 常用命令

```bash
# 最简单的用法
python demo/onnx_to_ttir.py 输入.onnx

# 指定输出文件
python demo/onnx_to_ttir.py 输入.onnx -o 输出.ttir

# 查看每个阶段在做什么
python demo/onnx_to_ttir.py 输入.onnx -o 输出.ttir --verbose

# 同时保存中间结果（图结构、执行计划、生成的 Triton 源码）
python demo/onnx_to_ttir.py 输入.onnx -o 输出.ttir --emit-intermediates

# 指定分块大小
python demo/onnx_to_ttir.py 输入.onnx -o 输出.ttir \
  --block-m 32 --block-n 32 --block-k 16

# 指定目标 GPU（默认 70，也就是 V100）
python demo/onnx_to_ttir.py 输入.onnx -o 输出.ttir --arch 70
```

## 输出文件

默认只写一个文件：

```text
输出.ttir
```

加上 `--emit-intermediates` 后，还会在输出文件旁边生成：

```text
输出.graph.json    从 ONNX 读出来的图信息
输出.plan.json     分块、网格、参数类型等执行计划
输出.kernel.py     自动生成的 Triton Python 源码
```

## 示例模型

`make_sample_model.py` 默认生成：

```text
A: f32 [130, 70]
B: f32 [70, 50]
C: f32 [130, 50]
```

也可以自定义形状：

```bash
python demo/make_sample_model.py -o /tmp/model.onnx --m 64 --k 32 --n 64
```

## 目录结构

```text
demo/
├── onnx_to_ttir.py       主程序：ONNX → TTIR
├── make_sample_model.py  生成单算子 MatMul 的示例模型
├── README.md             本文件
├── REPORT.md             详细技术报告
└── out/                  运行后生成的模型、TTIR 和中间文件
```

## 说明

这是一次性 demo，不是正式实现。它只用来验证“单算子 ONNX → TTIR”这条流程能跑通，不支持完整的 ONNX 算子集、动态形状、算子融合、权重处理或多 kernel 编排。
