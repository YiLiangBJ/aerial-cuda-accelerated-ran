
from collections import defaultdict
import re
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Silence TensorFlow.
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import cupy as cp
import numpy as np
import sionna
import tensorflow as tf

from aerial.phy5g.pdsch import PdschTx
from aerial.phy5g.pusch import PuschRx
from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.algorithms import TrtEngine
from aerial.phy5g.algorithms import TrtTensorPrms
from aerial.phy5g.ldpc import get_mcs
from aerial.phy5g.ldpc import random_tb
from aerial.phy5g.ldpc import get_tb_size
from aerial.phy5g.ldpc import LdpcDeRateMatch
from aerial.phy5g.ldpc import LdpcDecoder
from aerial.phy5g.ldpc import CrcChecker
from aerial.pycuphy.types import PuschLdpcKernelLaunch
from aerial.phy5g.config import PuschConfig
from aerial.phy5g.config import PuschUeConfig
from aerial.util.cuda import get_cuda_stream
from simulation_monitor import SimulationMonitor

# Configure the notebook to use only a single GPU and allocate only as much memory as needed.
# For more details, see https://www.tensorflow.org/guide/gpu.
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Simulation parameters.
random_seed = 42
np.random.seed(random_seed)
try:
    import cupy as _cp
    _cp.random.seed(random_seed)
except Exception:
    pass
try:
    tf.random.set_seed(random_seed)
except Exception:
    pass


num_slots = 10000
min_num_tb_errors = 250

# Run multiple scenarios back-to-back.
# Each scenario has its own (mcs_index, channel_model, esno_db_range) and will be saved separately.
SCENARIOS = [
    {
        "name": "mcs2_TDLB100-400",
        "mcs_index": 2,
        "channel_model": "TDLB100-400",
        # "channel_model": "Rayleigh",
        "esno_db_range": np.arange(-5.0, 25.0, 3),
    },
    {
        "name": "mcs16_TDLC300-100",
        "mcs_index": 16,
        "channel_model": "TDLC300-100",
        "esno_db_range": np.arange(5.0, 22.0, 3.0),
    },
    {
        "name": "mcs20_TDLA30-10",
        "mcs_index": 20,
        "channel_model": "TDLA30-10",
        "esno_db_range": np.arange(5.0, 22.0, 3.0),
    },
]

# In terminal/batch runs, showing matplotlib windows can block. Set SHOW_PLOTS=0 to disable.
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "1") != "0"


# channel_model = "TDLA30-10" # Channel model: Suitable values:
#                            # "Rayleigh" - Rayleigh block fading channel model (sionna.channel.RayleighBlockFading)
#                            # "TDLA30-10", "TDLB100-400", "TDLC300-100" - convenience names mapped to Sionna TR 38.901 TDL
#                            #     (internally mapped with Doppler->speed conversion)
#                            # "TDL-A30" or "TDL-A30-10" (also accepts "TDLA30-10" style) - TR 38.901 TDL models
#                            # "CDL-x", where x is one of ["A", "B", "C", "D", "E"] - for 3GPP CDL channel models
#                            #          as per TR 38.901.

num_tx_ant = 1             # UE antennas
num_rx_ant = 4             # gNB antennas
layers = 1                 # Number of layers
mcs_table = 0              # MCS table index
dmrs_ports = 1             # Used DMRS port.
# Numerology and frame structure. See TS 38.211.
num_ofdm_symbols = 14
fft_size = 4096
cyclic_prefix_length = 288
subcarrier_spacing = 30e3
num_guard_subcarriers = (410, 410)
num_slots_per_frame = 20

# System/gNB configuration
# num_tx_ant = 1             # UE antennas
# num_rx_ant = 4             # gNB antennas
cell_id = 41               # Physical cell ID
enable_pusch_tdi = 1       # Enable time interpolation for equalizer coefficients
eq_coeff_algo = 1          # Equalizer algorithm

# PUSCH parameters
rnti = 1234                # UE RNTI
scid = 0                   # DMRS scrambling ID
data_scid = 0              # Data scrambling ID
# layers = 2                 # Number of layers
# mcs_index = 2              # MCS index as per TS 38.214 table.
# mcs_table = 0              # MCS table index
# dmrs_ports = [0, 1]             # Used DMRS port.
start_prb = 0              # Start PRB index.
num_prbs = 273             # Number of allocated PRBs.
start_sym = 0              # Start symbol index.
num_symbols = 12           # Number of symbols.
dmrs_scrm_id = 41          # DMRS scrambling ID
dmrs_syms = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]  # Indicates which symbols are used for DMRS.
# dmrs_syms = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  # Indicates which symbols are used for DMRS.
dmrs_max_len = 1
dmrs_add_ln_pos = 2
num_dmrs_cdm_grps_no_data = 2
def _slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s


def _build_pusch_configs(mcs_index: int):
    """Build per-scenario PUSCH configs (depends on MCS/tb_size)."""
    mod_order, code_rate = get_mcs(mcs_index, mcs_table + 1)  # Different indexing for MCS table.
    tb_size = get_tb_size(  # TB size in bits
        mod_order=mod_order,
        code_rate=code_rate,
        dmrs_syms=dmrs_syms,
        num_prbs=num_prbs,
        start_sym=start_sym,
        num_symbols=num_symbols,
        num_layers=layers)

    pusch_ue_config = PuschUeConfig(
        scid=scid,
        layers=layers,
        dmrs_ports=dmrs_ports,
        rnti=rnti,
        data_scid=data_scid,
        mcs_table=mcs_table,
        mcs_index=mcs_index,
        code_rate=int(code_rate * 10),
        mod_order=mod_order,
        tb_size=tb_size // 8
    )

    pusch_configs = [PuschConfig(
        ue_configs=[pusch_ue_config],
        num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
        dmrs_scrm_id=dmrs_scrm_id,
        start_prb=start_prb,
        num_prbs=num_prbs,
        dmrs_syms=dmrs_syms,
        dmrs_max_len=dmrs_max_len,
        dmrs_add_ln_pos=dmrs_add_ln_pos,
        start_sym=start_sym,
        num_symbols=num_symbols
    )]

    return mod_order, code_rate, tb_size, pusch_configs

# Channel parameters
carrier_frequency = 3.5e9  # Carrier frequency in Hz.                    
delay_spread = 100e-9      # Nominal delay spread in [s]. Please see the CDL documentation
                           # about how to choose this value.
link_direction = "uplink"
# channel_model = "TDLA30-10" # Channel model: Suitable values:
#                            # "Rayleigh" - Rayleigh block fading channel model (sionna.channel.RayleighBlockFading)
#                            # "TDLA30-10", "TDLB100-400", "TDLC300-100" - convenience names mapped to Sionna TR 38.901 TDL
#                            #     (internally mapped with Doppler->speed conversion)
#                            # "TDL-A30" or "TDL-A30-10" (also accepts "TDLA30-10" style) - TR 38.901 TDL models
#                            # "CDL-x", where x is one of ["A", "B", "C", "D", "E"] - for 3GPP CDL channel models
#                            #          as per TR 38.901.
speed = 0.8333             # UE speed [m/s]. The direction of travel will chosen randomly within the x-y plane.


def _doppler_hz_to_speed_mps(doppler_hz: float, carrier_frequency_hz: float) -> float:
    """Convert maximum Doppler frequency [Hz] to speed [m/s].

    Uses f_d = (v/c) * f_c => v = f_d * c / f_c.
    """
    c_mps = 299_792_458.0
    if carrier_frequency_hz <= 0:
        raise ValueError(f"carrier_frequency must be > 0, got {carrier_frequency_hz}")
    return float(doppler_hz) * c_mps / float(carrier_frequency_hz)


def _parse_tdl_channel_model(name: str):
    """Parse a TDL channel model name.

    Supports:
        - Names like: "TDLA30-10", "TDLB100-400", "TDLC300-100"
    - Generic names: "TDL-A30", "TDL-A30-10", "TDLB100", etc.

        Notes:
        - For backwards compatibility, a trailing correlation label token (e.g., "Low")
            is tolerated and ignored.

    Returns: (tdl_model_str, delay_spread_seconds, doppler_hz_or_None)
    """
    s = name.strip()

    # Allow a few common spellings.
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", "-")

    # Tolerate (but ignore) a trailing correlation label token.
    s = re.sub(r"\s*(Low|Med|Medium|High)\s*$", "", s, flags=re.IGNORECASE)

    # Normalize: allow optional dash after TDL.
    m = re.match(r"^TDL-?([A-Ea-e])(\d+)(?:-(\d+))?$", s)
    if not m:
        return None

    model_letter = m.group(1).upper()
    delay_ns = int(m.group(2))
    doppler_hz = int(m.group(3)) if m.group(3) is not None else None

    # Sionna expects model strings like "A30", "B100", "C300".
    tdl_model = f"{model_letter}{delay_ns}"
    delay_spread_s = delay_ns * 1e-9
    return tdl_model, delay_spread_s, doppler_hz

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = (SCRIPT_DIR.parent / "models").resolve()

def _ensure_trt_engine(model_dir: Path) -> Path:
    """Create the TensorRT engine file once (if missing)."""
    nrx_onnx_file = model_dir / "neural_rx.onnx"
    nrx_trt_file = model_dir / "neural_rx.trt"

    if nrx_trt_file.exists() and os.environ.get("FORCE_REBUILD_TRT", "0") != "1":
        return nrx_trt_file

    if not nrx_onnx_file.exists():
        raise SystemExit(f"Missing ONNX model: {nrx_onnx_file}")

    command = (
        "trtexec "
        f"--onnx={nrx_onnx_file} "
        f"--saveEngine={nrx_trt_file} "
        "--skipInference "
        "--inputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,int32:chw,int32:chw "
        "--outputIOFormats=fp32:chw,fp32:chw "
        "--shapes=rx_slot_real:1x3276x12x4,rx_slot_imag:1x3276x12x4,h_hat_real:1x4914x1x4,h_hat_imag:1x4914x1x4 "
        "> /dev/null"
    )
    return_val = os.system(command)
    if return_val != 0 or not nrx_trt_file.exists():
        raise SystemExit("Failed to create the TRT engine file! (check trtexec and model paths)")
    print("TRT engine model created.")
    return nrx_trt_file

NRX_TRT_FILE = _ensure_trt_engine(MODEL_DIR)

pusch_tx = PdschTx(
    cell_id=cell_id,
    num_rx_ant=num_tx_ant,
    num_tx_ant=num_tx_ant,
)

# This is the fully fused PUSCH receiver chain.
pusch_rx = PuschRx(
    cell_id=cell_id,
    num_rx_ant=num_rx_ant, 
    num_tx_ant=num_rx_ant,
    enable_pusch_tdi=enable_pusch_tdi,
    eq_coeff_algo=eq_coeff_algo,
    # To make this equal separate PUSCH Rx components configuration:
    ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL
)



class NeuralRx:
    """PUSCH neural receiver class.
    
    This class encapsulates the PUSCH neural receiver chain built using
    pyAerial components.
    """

    def __init__(self,
                 num_rx_ant,
                 enable_pusch_tdi,
                 eq_coeff_algo):
        """Initialize the neural receiver."""
        self.cuda_stream = get_cuda_stream()

        # Build the components of the receiver. The channel estimator outputs just the LS
        # channel estimates.
        self.channel_estimator = ChannelEstimator(
            num_rx_ant=num_rx_ant,
            ch_est_algo=3,  # This is LS channel estimation.
            cuda_stream=self.cuda_stream
        )

        # Create the pyAerial TRT engine object. This wraps TensorRT and links it together
        # with the rest of cuPHY. Here pyAerial's Python bindings to the engine are used
        # to run inference with the neural receiver model.
        # The inputs of the neural receiver are:
        # - LS channel estimates
        # - The Rx slot
        # - Active DMRS ports (active layers out of the layers that the neural receiver supports)
        # - DMRS OFDM symbol locations (indices)
        # - DMRS subcarrier positions within a PRB (indices)        
        # Note that the shapes are given without batch size.
        nrx_trt_file = str(NRX_TRT_FILE)
        self.trt_engine = TrtEngine(
            # trt_model_file="../models/neural_rx.trt",
            trt_model_file=nrx_trt_file,
            max_batch_size=1,
            input_tensors=[TrtTensorPrms('rx_slot_real', (3276, 12, 4), np.float32),
                           TrtTensorPrms('rx_slot_imag', (3276, 12, 4), np.float32),
                           TrtTensorPrms('h_hat_real', (4914, 1, 4), np.float32),
                           TrtTensorPrms('h_hat_imag', (4914, 1, 4), np.float32),
                           TrtTensorPrms('active_dmrs_ports', (1,), np.float32),
                           TrtTensorPrms('dmrs_ofdm_pos', (3,), np.int32),
                           TrtTensorPrms('dmrs_subcarrier_pos', (6,), np.int32)],
            output_tensors=[TrtTensorPrms('output_1', (8, 1, 3276, 12), np.float32),
                            TrtTensorPrms('output_2', (1, 3276, 12, 8), np.float32)]
        )

        # LDPC (de)rate matching and decoding implemented using pyAerial.
        self.derate_match = LdpcDeRateMatch(
            enable_scrambling=True,
            cuda_stream=self.cuda_stream
        )
        self.decoder = LdpcDecoder(cuda_stream=self.cuda_stream)
        self.crc_checker = CrcChecker(cuda_stream=self.cuda_stream)
    
    def run(
        self,
        rx_slot,
        slot,
        pusch_configs
    ):
        """Run the receiver."""
        # Channel estimation.
        ch_est = self.channel_estimator.estimate(
            rx_slot=rx_slot,
            slot=slot,
            pusch_configs=pusch_configs
        )

        # This is the neural receiver part. 
        # It outputs the LLRs for all symbols.
        dmrs_ofdm_pos = np.where(np.array(pusch_configs[0].dmrs_syms))[0].astype(np.int32)
        dmrs_ofdm_pos = dmrs_ofdm_pos[None, ...]
        dmrs_subcarrier_pos = np.array([[0, 2, 4, 6, 8, 10]], dtype=np.int32)
        active_dmrs_ports = np.ones((1, 1), dtype=np.float32)
        rx_slot_in = rx_slot[None, :, pusch_configs[0].start_sym:pusch_configs[0].start_sym+pusch_configs[0].num_symbols, :]
        ch_est_in = np.transpose(ch_est[0], (0, 3, 1, 2)).reshape(ch_est[0].shape[0] * ch_est[0].shape[3], ch_est[0].shape[1], ch_est[0].shape[2])
        ch_est_in = ch_est_in[None, ...]
        input_tensors = {
            "rx_slot_real": np.real(rx_slot_in).astype(np.float32),
            "rx_slot_imag": np.imag(rx_slot_in).astype(np.float32),
            "h_hat_real": np.real(ch_est_in).astype(np.float32),
            "h_hat_imag": np.imag(ch_est_in).astype(np.float32),
            "active_dmrs_ports": active_dmrs_ports.astype(np.float32),
            "dmrs_ofdm_pos": dmrs_ofdm_pos.astype(np.int32),
            "dmrs_subcarrier_pos": dmrs_subcarrier_pos.astype(np.int32)
        }
        outputs = self.trt_engine.run(input_tensors)
        
        # The neural receiver outputs some values also for DMRS symbols, remove those
        # from the output.
        data_syms = np.array(pusch_configs[0].dmrs_syms[pusch_configs[0].start_sym:pusch_configs[0].start_sym + pusch_configs[0].num_symbols]) == 0
        llrs = np.take(outputs["output_1"][0, ...], np.where(data_syms)[0], axis=3)
        
        coded_blocks = self.derate_match.derate_match(
            input_llrs=[llrs],
            pusch_configs=pusch_configs
        )
    
        code_blocks = self.decoder.decode(
            input_llrs=coded_blocks,
            pusch_configs=pusch_configs
        )

        decoded_tbs, _ = self.crc_checker.check_crc(
            input_bits=code_blocks,
            pusch_configs=pusch_configs
        )

        return decoded_tbs

neural_rx = NeuralRx(
    num_rx_ant=num_rx_ant, 
    enable_pusch_tdi=enable_pusch_tdi,
    eq_coeff_algo=eq_coeff_algo
)

# Define the resource grid.
resource_grid = sionna.phy.ofdm.ResourceGrid(
    num_ofdm_symbols=num_ofdm_symbols,
    fft_size=fft_size,
    subcarrier_spacing=subcarrier_spacing,
    num_tx=1,
    num_streams_per_tx=1,
    cyclic_prefix_length=cyclic_prefix_length,
    num_guard_carriers=num_guard_subcarriers,
    dc_null=False,
    pilot_pattern=None,
    pilot_ofdm_symbol_indices=None
)
resource_grid_mapper = sionna.phy.ofdm.ResourceGridMapper(resource_grid)
remove_guard_subcarriers = sionna.phy.ofdm.RemoveNulledSubcarriers(resource_grid)

# Define the antenna arrays.
ue_array = sionna.phy.channel.tr38901.Antenna(
    polarization="single",
    polarization_type="V",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency
)
gnb_array = sionna.phy.channel.tr38901.AntennaArray(
    num_rows=1,
    num_cols=int(num_rx_ant/2),
    polarization="dual",
    polarization_type="cross",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency
)


def _build_sionna_ofdm_channel(channel_model: str):
    channel_model = re.sub(r"\s*(Low|Med|Medium|High)\s*$", "", channel_model, flags=re.IGNORECASE)

    if channel_model == "Rayleigh":
        ch_model = sionna.phy.channel.RayleighBlockFading(
            num_rx=1,
            num_rx_ant=num_rx_ant,
            num_tx=1,
            num_tx_ant=num_tx_ant
        )

    elif channel_model.startswith("TDL"):
        parsed = _parse_tdl_channel_model(channel_model)
        if parsed is None:
            raise ValueError(
                f"Invalid TDL channel model '{channel_model}'. "
                "Examples: 'TDLA30-10', 'TDLB100-400', 'TDLC300-100', 'TDL-A30', 'TDL-A30-10'."
            )

        tdl_model, delay_spread_s, doppler_hz = parsed
        tdl_speed = _doppler_hz_to_speed_mps(doppler_hz, carrier_frequency) if doppler_hz is not None else speed

        ch_model = sionna.phy.channel.tr38901.TDL(
            model=tdl_model,
            delay_spread=delay_spread_s,
            carrier_frequency=carrier_frequency,
            min_speed=tdl_speed,
            max_speed=tdl_speed,
            num_tx_ant=num_tx_ant,
            num_rx_ant=num_rx_ant,
        )

    elif "CDL" in channel_model:
        cdl_model = channel_model[-1]

        ch_model = sionna.phy.channel.tr38901.CDL(
            cdl_model,
            delay_spread,
            carrier_frequency,
            ue_array,
            gnb_array,
            link_direction,
            min_speed=speed
        )
    else:
        raise ValueError(f"Invalid channel model {channel_model}!")

    return sionna.phy.channel.OFDMChannel(
        ch_model,
        resource_grid,
        add_awgn=True,
        normalize_channel=True,
        return_channel=False
    )


def apply_channel(channel, tx_tensor, No):
    """Transmit the Tx tensor through the radio channel."""
    tx_tensor = tf.transpose(tx_tensor, (2, 1, 0))
    tx_tensor = tf.reshape(tx_tensor, (1, -1))[None, None]
    tx_tensor = resource_grid_mapper(tx_tensor)
    rx_tensor = channel(tx_tensor, No)
    rx_tensor = remove_guard_subcarriers(rx_tensor)
    rx_tensor = rx_tensor[0, 0]
    rx_tensor = tf.transpose(rx_tensor, (2, 1, 0))
    return rx_tensor

cases = ["PUSCH Rx", "Neural Rx"]


def run_scenario(scn: dict) -> str:
    scn_name = scn["name"]
    mcs_index = int(scn["mcs_index"])
    channel_model = str(scn["channel_model"])
    esno_db_range = np.array(scn["esno_db_range"], dtype=float)

    mod_order, code_rate, tb_size, pusch_configs = _build_pusch_configs(mcs_index)
    channel = _build_sionna_ofdm_channel(channel_model)

    cfg = {
        "Scenario": {"name": scn_name},
        "Sim": {
            "use_cupy": True,
            "esno_db_range": np.array(esno_db_range),
            "num_slots": num_slots,
            "min_num_tb_errors": min_num_tb_errors,
            "random_seed": int(random_seed)
        },
        "Frame": {
            "num_ofdm_symbols": num_ofdm_symbols,
            "fft_size": fft_size,
            "cyclic_prefix_length": cyclic_prefix_length,
            "subcarrier_spacing": subcarrier_spacing,
            "num_guard_subcarriers": list(num_guard_subcarriers),
            "num_slots_per_frame": num_slots_per_frame
        },
        "System": {
            "num_tx_ant": num_tx_ant,
            "num_rx_ant": num_rx_ant,
            "cell_id": cell_id,
            "enable_pusch_tdi": enable_pusch_tdi,
            "eq_coeff_algo": eq_coeff_algo
        },
        "PUSCH": {
            "rnti": rnti,
            "scid": scid,
            "data_scid": data_scid,
            "layers": layers,
            "mcs_index": mcs_index,
            "mcs_table": mcs_table,
            "mod_order": int(mod_order),
            "code_rate": float(code_rate),
            "dmrs_ports": dmrs_ports,
            "start_prb": start_prb,
            "num_prbs": num_prbs,
            "start_sym": start_sym,
            "num_symbols": num_symbols,
            "dmrs_scrm_id": dmrs_scrm_id,
            "dmrs_syms": list(dmrs_syms),
            "dmrs_max_len": dmrs_max_len,
            "dmrs_add_ln_pos": dmrs_add_ln_pos,
            "num_dmrs_cdm_grps_no_data": num_dmrs_cdm_grps_no_data,
            "precoding_matrix": None,
            "tb_size_bits": int(tb_size)
        },
        "Channel": {
            "carrier_frequency": carrier_frequency,
            "delay_spread": delay_spread,
            "link_direction": link_direction,
            "channel_model": re.sub(r"\s*(Low|Med|Medium|High)\s*$", "", channel_model, flags=re.IGNORECASE),
            "speed": speed
        },
        "Algo": {
            "use_trt": True,
            "trt_engine": str(NRX_TRT_FILE)
        }
    }

    monitor = SimulationMonitor(cases, esno_db_range, config=cfg)

    for esno_db in esno_db_range:
        monitor.step(float(esno_db))
        num_tb_errors = defaultdict(int)

        for slot_idx in range(num_slots):
            slot_number = slot_idx % num_slots_per_frame

            tb_input_np = random_tb(
                mod_order=mod_order,
                code_rate=code_rate,
                dmrs_syms=dmrs_syms,
                num_prbs=num_prbs,
                start_sym=start_sym,
                num_symbols=num_symbols,
                num_layers=layers)
            tb_input = cp.array(tb_input_np, dtype=cp.uint8, order='F')

            tx_tensor = pusch_tx.run(
                tb_inputs=[tb_input],
                num_ues=1,
                slot=slot_number,
                num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
                dmrs_scrm_ids=[dmrs_scrm_id],
                start_prb=start_prb,
                num_prbs=num_prbs,
                dmrs_syms=dmrs_syms,
                start_sym=start_sym,
                num_symbols=num_symbols,
                scids=[scid],
                layers=[layers],
                dmrs_ports=[dmrs_ports],
                rntis=[rnti],
                data_scids=[data_scid],
                code_rates=[code_rate * 10],
                mod_orders=[mod_order]
            )

            tx_tensor_tf = tf.experimental.dlpack.from_dlpack(tx_tensor.toDlpack())
            No = pow(10., -float(esno_db) / 10.)
            rx_tensor_tf = apply_channel(channel, tx_tensor_tf, No)
            rx_tensor = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(rx_tensor_tf))

            tb_crcs, tbs = pusch_rx.run(
                rx_slot=rx_tensor,
                slot=slot_number,
                pusch_configs=pusch_configs
            )

            num_tb_errors["PUSCH Rx"] += int(np.array_equal(tbs[0], tb_input_np) == False)

            tbs = neural_rx.run(
                rx_slot=rx_tensor,
                slot=slot_number,
                pusch_configs=pusch_configs
            )
            num_tb_errors["Neural Rx"] += int(np.array_equal(tbs[0], tb_input_np) == False)

            monitor.update(num_tbs=slot_idx + 1, num_tb_errors=num_tb_errors)
            if (np.array(list(num_tb_errors.values())) >= min_num_tb_errors).all():
                break

        monitor.finish_step(num_tbs=slot_idx + 1, num_tb_errors=num_tb_errors)

    if SHOW_PLOTS:
        monitor.finish()

    out_dir = str((SCRIPT_DIR / 'results' / _slugify(scn_name)).resolve())
    base_name = _slugify(f'neural_receiver_results_{scn_name}')
    out = monitor.save(out_dir, base_name, fmt='mat')
    print('Saved results to', out)
    return out


all_out = []
for scn in SCENARIOS:
    print("\n=== Running scenario:", scn["name"], "===")
    all_out.append(run_scenario(scn))

print("\nAll scenarios finished.")
for p in all_out:
    print(" -", p)
