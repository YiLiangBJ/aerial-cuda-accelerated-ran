# SRS 详细算法说明与代码映射

日期：2025-12-19

目的：把 `5GModel`（MATLAB）中 SRS 的接收/估计算法步骤抽象出来，映射到 `pyaerial`（Python）与 `cuPHY`（C++/CUDA）实现的关键文件，并给出验证与对比建议。

---

## 1. 算法高层步骤（来自 `5GModel/nr_matlab/srs/detSrs.m`）

1. 接收的频域数据：输入为 `X_tf`（频域资源格，维度：subcarrier x symbol x antenna）。
2. 根据 SRS 配置（PDU）计算参数：
   - SRS 端口数 `N_ap_SRS`、符号数 `N_symb_SRS`、重复次数 `R`、comb size `K_TC`、cyclic shifts、comb offset 等。
   - 由表（`srs_BW_table`）计算每 comb 带宽 `M_sc_b_SRS`。
3. 生成 reference sequence：
   - 用 Gold 序列（`build_Gold_sequence`）和 Low-PAPR 生成器（`LowPaprSeqGen`）得到 `r_bar`（参考 ZC 序列或低峰值序列）。
4. 计算映射起始位置 `k0`（频域位置），按 port 与 symbol 计算频域索引 `freq_idx` 与符号索引 `sym_idx`。
5. 相关/匹配滤波：
   - 对每个 UE/port，在对应的 RE 上做乘积/相关：`xcor = conj(r) .* Xtf(freq_idx, sym_idx, antenna)`。
6. 天线功率 / RSSI 估计：
   - 计算每个接收天线的接收功率 `ant_rssi_lin` / `ant_rssi_dB` 以及总体 `rssi_dB`。
7. 时偏估计：
   - 通过对相邻 RE 的相位差求和（`xcor_sum`），计算整体相位旋转 `phaRot` 并转换为时间偏差 `to_est_ms`。
8. 信道估计（per-RB / per-estimate）：
   - 聚合 RE（分块 RB）并平均得 `Hest`，同时估计噪声功率 `Pn` 与信号功率 `Ps`，计算 SNR（wide 和 per-RB）。
9. 输出结构：
   - `SrsOutputList` 包含 `to_est_ms`, `Hest`, `nRbHest`, `wideSnr`, `rbSnr`, `rssi`, `ant_rssi` 等字段。

---

## 2. MATLAB -> pyaerial -> cuPHY 的代码映射

- MATLAB（参考实现与 TV）
  - 生成 SRS：`5GModel/nr_matlab/srs/genSrs.m`（生成 `X_tf`、写 HDF5 TV）。
  - 检测/估计：`5GModel/nr_matlab/srs/detSrs.m`（xcor、phaRot、Hest、SNR 计算、保存 TV）。
  - 过滤/投影/去偏工具：`build_SRS_filter.m`, `update_projCoeffs.m` 等。

- Python / pyaerial（API 封装与流程）
  - API 定义：`pyaerial/src/aerial/phy5g/srs/srs_api.py`（`SrsRxPipeline`, `SrsRxConfig`, `SrsReport`）。
  - Pipeline 实现：`pyaerial/src/aerial/phy5g/srs/srs_rx.py`（把输入转换为 CuPy，调用 `pycuphy.SrsRx.run()`，然后组装 `SrsReport`）。
  - pybind 绑定：`pyaerial/pybind11/pycuphy_srs_rx.cpp`, `pycuphy_srs_rx.hpp`, `pycuphy_srs_util.*`（把 Python 参数转换并调用到 cuPHY）。

- cuPHY（高性能实现）
  - 核心类（声明）：`cuPHY/src/cuphy_channels/srs_rx.hpp`（`SrsRx` 类，缓冲区、launch cfg、chEst 句柄、kernel nodes 等）。
  - 实现（可执行和库）：`cuPHY/src/cuphy_channels/srs_rx.cpp`（实现 `cuphyCreateSrsRx`, `cuphySetupSrsRx`, `cuphyRunSrsRx`, `cuphyDestroySrsRx` 等）；`srs_tx.cpp` 同理用于发射。 
  - 构建/测试：`cuPHY/src/cuphy_channels/CMakeLists.txt`, `cuPHY/test/cuphy_unit_test.sh`（包含 `runSRS`，使用 TV 文件做回归测试）。

---

## 3. 相互调用细节（更细粒度）

1. Python 层 `SrsRx.__call__` 做的数据准备：
   - 将每个 `rx_data` 转为 `cupy` 数组并包装为 `pycuphy.CudaArrayComplexFloat`（保证连续性与 Fortran-order）；
   - 通过 `pycuphy.SrsRx(num_cells, num_rx_ant, chest_algo_idx, enable_delay_offset_correction, chest_params, num_max_srs_ues, cuda_stream)` 创建底层对象；
   - 调用 `.run(rx_data, srs_ue_configs, srs_cell_configs)`，该方法在 pybind 层将参数打包为 C 结构并调用 `cuPHY` 的 C 接口。

2. pybind 层职责：
   - 类型转换（NumPy/CuPy -> 内部 C buffer）；
   - 管理 CUDA stream 与异步拷贝；
   - 将 `cuPHY` 返回的结构（如 `cuphySrsReport_t`）转换为可在 Python 侧访问的 NumPy 数组或原生字段。

3. cuPHY 侧流水线（核心运算）
   - kernel launch 配置由 `m_srsChEstLaunchCfg` 等结构定义；
   - 执行内核：频域索引加载、与 `r_bar` 的相关/乘积、沿 antenna/port 求平均、相位旋转汇总、SNR 估计；
   - 可能使用 graph（CUDA graph）或 stream 来串联 copy/kernel/normalize 等步骤以减少同步开销；
   - 结果写入 host/GPU buffer，供 `get_*` 接口读取。

---

## 4. 验证策略（如何确保 cuPHY 与 5GModel 算法一致）

1. 使用 `5GModel` 生成 TV（HDF5）。确保 TV 包含以下字段：SrsParams、X_tf（或压缩后的样本）、以及参考的 `Hest`, `to_est_ms`, `rbSnr` 等。
2. 在 Python 环境中用 `pyaerial` 读取 TV 并用 `SrsRx` pipeline 运行：

```bash
# 伪命令，确保已安装 pyaerial（可用 -e 模式）并在合适的 PYTHONPATH
python -c "from aerial.phy5g.srs.srs_rx import SrsRx; ... # load h5 and call SrsRx"
```

3. 对比字段：`to_est_ms`, `Hest`（元素或平均），`rbSnr`, `wideSnr`, `rssi`。考虑数值公差（浮点精度、fp16/32 压缩、算法近似）。
4. 若差异存在：
   - 检查数据布局（Fortran vs C order）；
   - 检查是否启用了任何 fp16/approx 或量化（`SimCtrl.fp16AlgoSel` 等）；
   - 查看 `pyaerial` 的 pybind 层是否执行了额外的预处理或 scaling；
   - 在 cuPHY 侧开启更详细的日志（NVLOG）或在单元测试中增加对中间结果（xcor、phaRot）的小规模 dump。

---

## 5. 建议的下一步工作（实操）

- 如果你想要我继续，我可以：
  1. 把上述验证流程写成一个可运行的 Python 脚本 `tools/verify_srs_with_tv.py`，自动读取 TV 并运行 `pyaerial` 的 `SrsRx`，然后生成差异报告（CSV/JSON）。
  2. 或者直接在本地（当前仓库）演示一次：读取一个存在的 SRS TV（我会搜索并选一个），运行 `pyaerial` 的 `SrsRx`，并把比对结果追加到 `docs/srs_detailed.md`。

请告诉我你想让我执行哪一步（1 或 2，或其它）。
