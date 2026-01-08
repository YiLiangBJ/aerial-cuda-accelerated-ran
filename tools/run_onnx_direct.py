#!/usr/bin/env python3
"""Run an ONNX model directly (no TensorRT) with synthetic inputs.

/opt/nvidia/ + 冒烟测试(Shape Smoke Test)”，帮助你理解： 
- ONNX 模型有哪些输入/输出张量
- 每个张量的 dtype / shape 是什么
- onnxruntime 是如何执行一次推理的
source /opt/venv/bin/main() 流程）：
1) 选择 onnxruntime provider（CPU / CUDA）
2) 加载 ONNX：ort.InferenceSession(model_path, providers=...)
3) 遍历 session.get_inputs()，打印每个输入张量的 name/type/shape
4) 根据输入签名生成“'numpy 随机数)，保证 dtype/shape 匹配PY'
5) 调用 sess.run(output_names, feeds) 执行一次推理
6) 打印输出张量的 dtype/shape

source /opt/venv/bin/activate
- 本脚本生成的是随机数据，不是有物理意义的无线信号。
- 目的仅是验证“ + 输入维度是否被接受”。

 pyaerial/models/neural_rx.onnx：
- rx_slot_real / rx_slot_imag: 形状类似 [B, 3276, 12, 4]
  - B: batch（默认 1）
  - 3276: 某种“flattened RE”长度（来自 notebook 的配置/推理接口约定）
  - 12: OFDM symbols 数（该 ONNX 里通常是静态写死 12）
  - 4: 特征/天线等维度（具体语义依模型定义）
- h_hat_real / h_hat_imag: 形状类似 [B, 4914, 1, 4]

 --symbols 改成 14：
- 若 ONNX 的 symbols 维是静态 12，onnxruntime 会直接报：Got 14 Expected 12。
- 这意味着“就这个 ONNX 文件本身”不支持 14 symbols；要支持必须重新导出/训练成动态维。

source /opt/venv/bin/activate
  直接运行（使用默认模型与默认 symbols=12）：
    /opt/venv/bin/python -u tools/run_onnx_direct.py

  指定 symbols：
    /opt/venv/bin/python -u tools/run_onnx_direct.py --symbols 12
    /opt/venv/bin/python -u tools/run_onnx_direct.py --symbols 14

  指定模型：
    /opt/venv/bin/python -u tools/run_onnx_direct.py --model pyaerial/models/neural_rx.onnx
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    import onnxruntime as ort
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "onnxruntime is required (this repo usually has onnxruntime-gpu).\n"
        f"Import error: {e}"
    )


def _pick_providers(prefer_cuda: bool) -> List[str]:
    """选择 onnxruntime 执行后端(provider)。

    - CPUExecutionProvider：一定可用；适合做 shape/正确性检查。
    - CUDAExecutionProvider：如果装了 且环境可用，就能 onnxruntime-gpu GPU 跑。

    返回值是一个列表，onnxruntime 会按顺序尝试。
    """

    available = ort.get_available_providers()
    if prefer_cuda and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _ort_type_to_numpy_dtype(ort_type: str) -> np.dtype:
    """把 onnxruntime 的  numpy dtype。

    onnxruntime 的输入/输出类型通常长这样：
      - tensor(float)
      - tensor(int32)
      - tensor(float16)

    我们需要它来创建 dtype 匹配的 numpy 输入。
    """

    m = re.match(r"^tensor\((.+)\)$", ort_type)
    if not m:
        raise ValueError(f"Unsupported ORT type: {ort_type}")

    elem = m.group(1)
    mapping = {
        "float": np.float32,
        "float16": np.float16,
        "double": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
    }

    if elem not in mapping:
        raise ValueError(f"Unsupported element type: {elem} (from {ort_type})")

    return np.dtype(mapping[elem])


def _resolve_shape(
    input_name: str,
    ort_shape: Sequence[Any],
    symbols: int,
    batch: int,
    rx_slot_len: int,
    h_hat_len: int,
) -> Tuple[int, ...]:
    """把 ORT 返回的 shape（可能含动态维）解析成具体整数 shape。

source /opt/venv/bin/activate
    - int：静态维度
    - str：动态维度名称（例如 'unk__644'）
    - None：动态维度

    为了分配 numpy 数组，我们必须把它变成全是 int 的 tuple。

    本函数的策略（偏向 neural_rx.onnx）：
#    - 保留

    - 如果输入名含 rx_slot 且某个维度是静态 12，则认为它是 symbols 维，替换为传入的 symbols
    - 其余动态维度：
      * dim0 -> batch
      * rx_slot_* dim1 -> rx_slot_len（默认 3276）
      * h_hat_* dim1   -> h_hat_len（默认 4914）
      * 其他动态维 -> 1

    注意：就算这里把 symbols 14，也不代表 替换成
    如果 ONNX 把 symbols 写死为 12，onnxruntime 仍会报 Got 14 Expected 12。
    """

    # 先把静态维度保留，动态维度 # None 标记。
    dims: List[int | None] = []
    for d in ort_shape:
        if isinstance(d, int) and d > 0:
            dims.append(d)
        else:
            dims.append(None)

    # 特判：如果某些维度在 ONNX 里写死 12，把它看成 symbols 轴。
    if "rx_slot" in input_name:
        dims = [symbols if d == 12 else d for d in dims]

    # 把剩余的动态维度填成具体数。
    for i, d in enumerate(dims):
        if d is not None:
            continue

        if i == 0:
            #  0 维是 batch。
            dims[i] = batch
            continue

        if "rx_slot" in input_name and i == 1:
            # neural_rx.onnx：rx_slot_* 形状预期是 [B, 3276, num_symbols, 4]
            dims[i] = rx_slot_len
            continue

        if "h_hat" in input_name and i == 1:
            # neural_rx.onnx：h_hat_* 形状预期是 [B, 4914, 1, 4]
            dims[i] = h_hat_len
            continue

        # 兜底策略：其他动态维度填 1。
        dims[i] = 1

    return tuple(int(x) for x in dims)  # type: ignore[arg-type]


def _make_input_array(name: str, shape: Tuple[int, ...], ort_type: str, seed: int) -> np.ndarray:
    """按 dtype/shape 生成一个假的输入张量。

#source /opt/venv/bin/activate

    - 整数/布尔：小范围随机值

    这些值本身不重要，只要 dtype/shape 能通过 ORT 校验并让图跑起来。
    """

    dtype = _ort_type_to_numpy_dtype(ort_type)
    rng = np.random.default_rng(seed)

    if dtype in (np.float16, np.float32, np.float64):
        return rng.standard_normal(size=shape).astype(dtype)

    if dtype in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ):
        return rng.integers(low=0, high=4, size=shape, dtype=dtype)

    if dtype == np.bool_:
        return rng.integers(low=0, high=2, size=shape).astype(np.bool_)

    raise ValueError(f"Unhandled dtype: {dtype}")


def main() -> int:
    # argparse 负责从命令行解析参数。
    ap = argparse.ArgumentParser(description="Run ONNX directly with synthetic inputs")

    # 让 --model 不是必填，这样你在 VS Code 里直接 Run Python File 也能跑。
    ap.add_argument(
        "--model",
        default="./pyaerial/models/neural_rx.onnx",
        help="Path to .onnx (default: ./pyaerial/models/neural_rx.onnx)",
    )

    # 你想试的“符号数”维度。
    ap.add_argument("--symbols", type=int, default=12, help="Number of symbols to test")

    # batch 默认 1：只跑一条样本。
    ap.add_argument("--batch", type=int, default=1)

    # 下面两个是 neural_rx.onnx 的约定长度（来自仓库 notebook/推理接口），换别的模型可能要改。
    ap.add_argument(
        "--rx-slot-len",
        type=int,
        default=3276,
        help="Length of rx_slot_* axis (neural_rx default: 3276)",
    )
    ap.add_argument(
        "--h-hat-len",
        type=int,
        default=4914,
        help="Length of h_hat_* axis (neural_rx default: 4914)",
    )

    # 如果可用，优先用 CUDA provider；否则自动回退 CPU
    ap.add_argument("--cuda", action="store_true", help="Prefer CUDAExecutionProvider")

    # 固定随
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        raise SystemExit(f"Model not found: {model_path}")

    providers = _pick_providers(args.cuda)

    print(f"Model: {model_path}")
    print(f"Providers: {providers}")
    print(f"Test symbols: {args.symbols}")

    # 创 session：此时 onnxruntime 会加载模型、做图优化、准备执行。
    sess = ort.InferenceSession(model_path, providers=providers)

    # 准备输入 feeds：dict[input_name] = numpy_array
    print("\nInputs:")
    feeds: Dict[str, np.ndarray] = {}
    for idx, inp in enumerate(sess.get_inputs()):
        # inp.name: 输入张量名（必须和 feeds 的 key 一致）
        # inp.type: 输入 dtype（如 tensor(float)）
        # inp.shape: 输入 shape（可能含动态维）
        resolved = _resolve_shape(
            inp.name,
            inp.shape,
            symbols=args.symbols,
            batch=args.batch,
            rx_slot_len=args.rx_slot_len,
            h_hat_len=args.h_hat_len,
        )

        print(f"- {inp.name}: type={inp.type} shape={inp.shape} -> {resolved}")
        feeds[inp.name] = _make_input_array(inp.name, resolved, inp.type, seed=args.seed + idx)

    # 输出名字列表：sess.run 需要你指定取哪些输出。
    print("\nOutputs:")
    output_names = [o.name for o in sess.get_outputs()]
    for out in sess.get_outputs():
        print(f"- {out.name}: type={out.type} shape={out.shape}")

    print("\nRunning...")
    try:
        # 这行就是“真正开始推理”的地方：
        # - output_names: 你希望拿到的输出张量
        # - feeds: 输入张量 dict
        outputs = sess.run(output_names, feeds)
    except Exception as e:
        print("FAILED")
        print(str(e))
        return 2

    print("OK")
    for name, arr in zip(output_names, outputs):
        if isinstance(arr, np.ndarray):
            print(f"- {name}: {arr.dtype} {tuple(arr.shape)}")
        else:
            print(f"- {name}: {type(arr)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
