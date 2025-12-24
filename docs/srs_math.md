# SRS 数学化解读与 MATLAB 调试流程

日期：2025-12-19

目的：把 `5GModel/nr_matlab/srs/detSrs.m` 的实现按数学公式逐步解释，并给出一个可运行的 MATLAB 调试流程（可使用 `genSrs.m` 生成 TV 或直接使用已有 TV）。

---

## 一、数学化解读（与代码行关联）

为了便于对照，下面用数学符号描述 `detSrs.m` 的主要计算步骤，并在每一步对照实现位置（函数/段落）。记号说明：

- 令 X_tf(f, l, a) 为频域资源格：频率索引 f，符号索引 l，接收天线 a。（代码中的 `Xtf(freq_idx+1, sym_idx+1, idxAnt)`）
- 参考序列 r_{l',p}[n]：第 l' 个 SRS 符号、第 p 个端口上的参考子载波序列，长度为 M_sc_b_SRS（由 `LowPaprSeqGen` 产生）。（代码中 `r_bar`）
- 频域起始位置 k0(l', p)：第 l'、p 对应的首个子载波索引。（代码中 `k0` 计算）
- 交叉相关（per-RE）: 对每个接收天线 a，构造 xcor[n, l', a, p] = conj(r_{l',p}[n]) * X_tf[k0 + n, l0 + l', a]

数学步骤：

1) 生成参考序列（ZC / Low-PAPR）：
   - Gold 序列 c(m) 用于 hopping/index 推导。
   - 基序列 r_bar(l') = LowPaprSeqGen(M_sc_b_SRS, u(l'), v(l'))。
   - 代码：`r_bar(l_prime+1,:) = LowPaprSeqGen(...)`。

2) 频域映射（构造频域索引）：
   - k0(l', p) = n_shift * N_sc_RB + k_TC + sum_b K_TC * M_sc_b_SRS * nb(b)
   - freq_idx = k0(l',p) + [0:K_TC:(M_sc_b_SRS-1)*K_TC]
   - 代码：`k0` 的计算循环和 `freq_idx = k0(...) + [0:K_TC:...]`。

3) 相关/匹配滤波（核心）：
   - 对每个接收天线 a，计算
     Xcor(n, l', a, p) = conj(r_{l',p}[n]) * X_tf(k0 + n, l0 + l', a)
   - 等价于对接收频域符号与参考序列做 element-wise 乘积（matched-filter）。
   - 代码：`xcor(:, l_prime+1, idxAnt, p+1) = conj(r(:)) .* Xtf(freq_idx+1, sym_idx+1, idxAnt);`

4) 天线 RSSI（接收功率）估计：
   - ant_rssi_lin[a] = mean_n,l' | Xcor(n, l', a, :) |^2  （平均所有 RE 和端口）
   - rssi_lin = mean_a ant_rssi_lin[a]
   - 换算 dB：rssi_dB = 10*log10(rssi_lin)
   - 代码：计算 `ant_rssi_lin` 与 `rssi_dB` 区段。

5) 时偏（Timing Offset）估计：
   - 对每个符号 idxSym、每个天线 idxAnt、每个端口 idxPort，先平均得到 Hest_port（向量），然后计算邻项乘积和：
     xcor_sum(idxSym, idxAnt, idxPort) = sum_{n=2..N} Hest_port[n] * conj(Hest_port[n-1])
   - 全局相位旋转 phaRot = angle(sum_{sym,ant,port} xcor_sum(sym,ant,port))
   - 时间偏 to_est = -phaRot / (2π)，再换算成秒：to_est_sec = to_est / (Δf * N_ap_SRS * K_TC)
   - 代码：`xcor_sum` 计算、`phaRot = angle(sum(sum(sum(xcor_sum))))`、`to_est_sec` 计算。

6) 信道估计与 SNR 估计（Hest, Ps, Pn）：
   - 将 Hest_port 的 RE 分为若干估计块（每块覆盖 nRePerEst RE），对每块取平均得到 Hest(idxEst)。
   - 估计信号功率 Ps 为平均 |Havg|^2；噪声功率 Pn 为平均 |Hdiff|^2（取决于算法选择，代码中 `algSel==1` 使用三点线性拟合并用 estFactor1 校正）。
   - wideband SNR = 10*log10(mean(Ps)/mean(Pn)) - 10*log10(N_ap_SRS)
   - per-RB SNR（rbSnr）通过聚合并 reshape 得到。
   - 代码：循环中的 `Hest5`, `Havg`, `Hdiff`, `Ps`, `Pn` 计算与最后的 `SNR_wide`, `SNR_rb`。

7) 输出打包：
   - 将 `to_est_ms`, `Hest`, `nRbHest`, `wideSnr`, `rbSnr`, `rssi`, `ant_rssi` 放入 `SrsOutputList`。
   - 代码：最后部分构建 `SrsOutputList` 条目并返回。

---

## 二、如何一步步在 MATLAB 里 debug SRS 接收流程（可运行脚本）

下面给出两种可行路径：
- A（推荐）: 在 MATLAB 环境中直接运行 `detSrs.m` 对一个由 `genSrs.m` 生成的测试向量进行端到端检测。优点：完全在 MATLAB 中，可用断点/变量查看；缺点：需要 MATLAB。 
- B：使用已有的 TV（HDF5），用 MATLAB 的 `detSrs` 读取 TV 并运行检测（适合你已有 TV）。

下面给出完整的可运行步骤（假设你在 MATLAB 工作目录为 `5GModel/nr_matlab`）：

步骤 0：准备
- 打开 MATLAB，确保路径包含 `5GModel/nr_matlab` 及子目录（`addpath(genpath('.../5GModel/nr_matlab'))`）。
- 若使用已有 TV，记下 TV 文件路径（例如 `tvDir/TVname_SRS_UE0_CUPHY_s0p0.h5`）。

步骤 A（从头生成并检测）
1. 准备 carrier 与 pdu（示例）：

```matlab
% 伪代码示例，具体字段按照项目的 PDU 结构填写
carrier.N_slot_frame_mu = 10; % 举例
carrier.N_symb_slot = 14;
carrier.idxSlotInFrame = 0;
carrier.idxFrame = 0;
carrier.delta_f = 30e3; % 子载波间隔

pdu.numAntPorts = 0; % 0->1 port, 1->2 ports, 2->4 ports (见 genSrs mappings)
pdu.numSymbols = 0; % 0->1 symbol
pdu.numRepetitions = 0; % 0->1
pdu.combSize = 0; % 0->comb size 2
pdu.timeStartPosition = 0;
pdu.sequenceId = 10;
pdu.configIndex = 0;
pdu.bandwidthIndex = 0;
pdu.combOffset = 0;
pdu.cyclicShift = 0;
pdu.frequencyPosition = 0;
pdu.frequencyShift = 0;
pdu.frequencyHopping = 0;
pdu.resourceType = 0;
pdu.Tsrs = 1;
pdu.Toffset = 0;
pdu.groupOrSequenceHopping = 0;

% 构造空频域资源格 Xtf（适当尺寸的复数矩阵）
Nf = 4096; % 根据仿真带宽选择
Nsym = carrier.N_symb_slot;
Nant = 4; % 接收天线数（例）
Xtf = zeros(Nf, Nsym, Nant);

% 生成 SRS 到 Xtf
Xtf = genSrs(pdu, srsTable, carrier, Xtf, 1);
```

2. 运行检测并查看中间变量

```matlab
% 单个PDU的列表结构
pduList = {pdu};
% detSrs 会调用 detSrs_cuphy，确保函数在路径中
SrsOutputList = detSrs(pduList, srsTable, carrier, Xtf);

% 查看输出
disp(SrsOutputList{1}.to_est_ms);
disp(size(SrsOutputList{1}.Hest));
```

调试建议：在 `detSrs.m` 中设置断点，按顺序观察 `r_bar`, `k0`, `xcor`, `xcor_sum`, `phaRot`, `Hest`, `Ps`, `Pn` 等变量；用 `plot(abs(xcor(:,1,1,1)))` 可视化相关强度，`angle(...)` 查看相位行为。

步骤 B（使用已有 TV 文件）
1. 读取 TV：

```matlab
h5file = 'path/to/TVname.h5';
SrsParams = hdf5read(h5file, 'SrsParams_0');
Xtf = hdf5read(h5file, 'X_tf');
```

2. 构造 `pduList`/`carrier`（从 `SrsParams` 填充）并调用 `detSrs`：

```matlab
pduList = ... % 从 SrsParams 填
carrier = ...
SrsOutputList = detSrs(pduList, srsTable, carrier, Xtf);
```

---

## 三、调试技巧与常见问题

- 若 `Xtf` 中信号看不到（低 RSSI），确认频率映射 `k0` 是否与你的 `Xtf` 布局匹配（采样率 / FFT size / RB mapping）。
- 检查 `K_TC`、`M_sc_b_SRS` 与 `combSize` 是否一致。频域索引偏差会导致相关几乎为零。
- 如果 `phaRot` 看起来随机且 `to_est_ms` 不可信，画出 `angle(sum(xcor_sum,dim))` 的实部/虚部变化趋势来检查相位稳定性。
- 若对比 `cuPHY` 输出存在偏差，注意 `cuPHY` 可能使用 fp16 或不同的缩放/量化策略；在 MATLAB 侧把 `X_tf` 转为 fp16 模拟也有助于排查精度差异。

---

## 四、如果你需要我帮忙做的事情

- 我可以生成一个 MATLAB 脚本 `tools/run_debug_detSrs.m`（放在仓库）来自动化上述 A 或 B 流程，并在关键步骤保存中间变量到 `.mat` 文件供你加载调试。
- 我也可以帮你挑一个现成的 TV（如果仓库里有）并在 Python 层运行 `pyaerial.SrsRx` 做一次端到端对比并把结果写入 docs。

请选择下一步（例如让我要先创建 `tools/run_debug_detSrs.m` 自动脚本并运行一次示例）。