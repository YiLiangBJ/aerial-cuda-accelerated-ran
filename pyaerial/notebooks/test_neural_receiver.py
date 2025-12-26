
from collections import defaultdict
import os
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

esno_db_range = np.arange(-4.0, -2.8, 0.2)
num_slots = 10000
min_num_tb_errors = 250

# Numerology and frame structure. See TS 38.211.
num_ofdm_symbols = 14
fft_size = 4096
cyclic_prefix_length = 288
subcarrier_spacing = 30e3
num_guard_subcarriers = (410, 410)
num_slots_per_frame = 20

# System/gNB configuration
num_tx_ant = 1             # UE antennas
num_rx_ant = 4             # gNB antennas
cell_id = 41               # Physical cell ID
enable_pusch_tdi = 1       # Enable time interpolation for equalizer coefficients
eq_coeff_algo = 1          # Equalizer algorithm

# PUSCH parameters
rnti = 1234                # UE RNTI
scid = 0                   # DMRS scrambling ID
data_scid = 0              # Data scrambling ID
layers = 1                 # Number of layers
mcs_index = 7              # MCS index as per TS 38.214 table.
mcs_table = 0              # MCS table index
dmrs_ports = 1             # Used DMRS port.
start_prb = 0              # Start PRB index.
num_prbs = 273             # Number of allocated PRBs.
start_sym = 0              # Start symbol index.
num_symbols = 12           # Number of symbols.
dmrs_scrm_id = 41          # DMRS scrambling ID
dmrs_syms = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]  # Indicates which symbols are used for DMRS.
dmrs_max_len = 1
dmrs_add_ln_pos = 2
num_dmrs_cdm_grps_no_data = 2
mod_order, code_rate = get_mcs(mcs_index, mcs_table+1)  # Different indexing for MCS table.
tb_size = get_tb_size(  # TB size in bits
    mod_order=mod_order,
    code_rate=code_rate,
    dmrs_syms=dmrs_syms,
    num_prbs=num_prbs,
    start_sym=start_sym,
    num_symbols=num_symbols,
    num_layers=layers)

# Channel parameters
carrier_frequency = 3.5e9  # Carrier frequency in Hz.                    
delay_spread = 100e-9      # Nominal delay spread in [s]. Please see the CDL documentation
                           # about how to choose this value.
link_direction = "uplink"
channel_model = "Rayleigh" # Channel model: Suitable values:
                           # "Rayleigh" - Rayleigh block fading channel model (sionna.channel.RayleighBlockFading)
                           # "CDL-x", where x is one of ["A", "B", "C", "D", "E"] - for 3GPP CDL channel models
                           #          as per TR 38.901.
speed = 0.8333             # UE speed [m/s]. The direction of travel will chosen randomly within the x-y plane.

MODEL_DIR = "../models"
nrx_onnx_file = f"{MODEL_DIR}/neural_rx.onnx"
nrx_trt_file = f"{MODEL_DIR}/neural_rx.trt"
command = f"trtexec " + \
    f"--onnx={nrx_onnx_file} " + \
    f"--saveEngine={nrx_trt_file} " + \
    f"--skipInference " + \
    f"--inputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,int32:chw,int32:chw " + \
    f"--outputIOFormats=fp32:chw,fp32:chw " + \
    f"--shapes=rx_slot_real:1x3276x12x4,rx_slot_imag:1x3276x12x4,h_hat_real:1x4914x1x4,h_hat_imag:1x4914x1x4 " + \
    f"> /dev/null"
return_val = os.system(command)
if return_val == 0:
    print("TRT engine model created.")
else:
    raise SystemExit("Failed to create the TRT engine file!")

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

# PUSCH configuration object. Note that default values are used for some parameters
# not given here.
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
# Note that this is a list. One UE group only in this case.
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
        self.trt_engine = TrtEngine(
            trt_model_file="../models/neural_rx.trt",
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
        pusch_configs=pusch_configs
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

if channel_model == "Rayleigh":
    ch_model = sionna.phy.channel.RayleighBlockFading(
        num_rx=1,
        num_rx_ant=num_rx_ant,
        num_tx=1,
        num_tx_ant=num_tx_ant
    )
    
elif "CDL" in channel_model:
    cdl_model = channel_model[-1]
    
    # Configure a channel impulse reponse (CIR) generator for the CDL model.
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

channel = sionna.phy.channel.OFDMChannel(
    ch_model,
    resource_grid,
    add_awgn=True,
    normalize_channel=True,
    return_channel=False
)

def apply_channel(tx_tensor, No):
    """Transmit the Tx tensor through the radio channel."""
    # Add batch and num_tx dimensions that Sionna expects and reshape.
    tx_tensor = tf.transpose(tx_tensor, (2, 1, 0))
    tx_tensor = tf.reshape(tx_tensor, (1, -1))[None, None]        
    tx_tensor = resource_grid_mapper(tx_tensor)
    rx_tensor = channel(tx_tensor, No)
    rx_tensor = remove_guard_subcarriers(rx_tensor)
    rx_tensor = rx_tensor[0, 0]
    rx_tensor = tf.transpose(rx_tensor, (2, 1, 0))
    return rx_tensor

cases = ["PUSCH Rx", "Neural Rx"]
# Build grouped configuration dict to record simulation and algorithm settings
cfg = {
    'Sim': {
        'use_cupy': True,
        'esno_db_range': np.array(esno_db_range),
        'num_slots': num_slots,
        'min_num_tb_errors': min_num_tb_errors,
        'random_seed': int(random_seed)
    },
    'Frame': {
        'num_ofdm_symbols': num_ofdm_symbols,
        'fft_size': fft_size,
        'cyclic_prefix_length': cyclic_prefix_length,
        'subcarrier_spacing': subcarrier_spacing,
        'num_guard_subcarriers': list(num_guard_subcarriers),
        'num_slots_per_frame': num_slots_per_frame
    },
    'System': {
        'num_tx_ant': num_tx_ant,
        'num_rx_ant': num_rx_ant,
        'cell_id': cell_id,
        'enable_pusch_tdi': enable_pusch_tdi,
        'eq_coeff_algo': eq_coeff_algo
    },
    'PUSCH': {
        'rnti': rnti,
        'scid': scid,
        'data_scid': data_scid,
        'layers': layers,
        'mcs_index': mcs_index,
        'mcs_table': mcs_table,
        'dmrs_ports': dmrs_ports,
        'start_prb': start_prb,
        'num_prbs': num_prbs,
        'start_sym': start_sym,
        'num_symbols': num_symbols,
        'dmrs_scrm_id': dmrs_scrm_id,
        'dmrs_syms': list(dmrs_syms),
        'dmrs_max_len': dmrs_max_len,
        'dmrs_add_ln_pos': dmrs_add_ln_pos,
        'num_dmrs_cdm_grps_no_data': num_dmrs_cdm_grps_no_data,
        'precoding_matrix': None,
        'tb_size_bits': int(tb_size)
    },
    'Channel': {
        'carrier_frequency': carrier_frequency,
        'delay_spread': delay_spread,
        'link_direction': link_direction,
        'channel_model': channel_model,
        'speed': speed
    },
    'Algo': {
        'use_trt': True
    }
}

monitor = SimulationMonitor(cases, esno_db_range, config=cfg)

# Loop the Es/No range.
bler = []
for esno_db in esno_db_range:
    monitor.step(esno_db)
    num_tb_errors = defaultdict(int)
    
    # Run multiple slots and compute BLER.
    for slot_idx in range(num_slots):
        slot_number = slot_idx % num_slots_per_frame                
        
        # Get modulation order and coderate.
        tb_input_np = random_tb(
            mod_order=mod_order,
            code_rate=code_rate,
            dmrs_syms=dmrs_syms,
            num_prbs=num_prbs,
            start_sym=start_sym,
            num_symbols=num_symbols,
            num_layers=layers)
        tb_input = cp.array(tb_input_np, dtype=cp.uint8, order='F')
        
        # Transmit PUSCH. This is where we set the dynamically changing parameters.
        # Input parameters are given as lists as the interface supports multiple UEs.
        tx_tensor = pusch_tx.run(
            tb_inputs=[tb_input],          # Input transport block in bytes.           
            num_ues=1,                     # We simulate only one UE here.
            slot=slot_number,              # Slot number.
            num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
            dmrs_scrm_ids=[dmrs_scrm_id],  # DMRS scrambling ID.
            start_prb=start_prb,           # Start PRB index.
            num_prbs=num_prbs,             # Number of allocated PRBs.
            dmrs_syms=dmrs_syms,           # List of binary numbers indicating which symbols are DMRS.
            start_sym=start_sym,           # Start symbol index.
            num_symbols=num_symbols,       # Number of symbols.
            scids=[scid],                  # DMRS scrambling ID.
            layers=[layers],               # Number of layers (transmission rank).
            dmrs_ports=[dmrs_ports],       # DMRS port(s) to be used.
            rntis=[rnti],                  # UE RNTI.
            data_scids=[data_scid],        # Data scrambling ID.
            code_rates=[code_rate * 10],   # Code rate x 1024 x 10.
            mod_orders=[mod_order]         # Modulation order.            
        )
                
        # Channel transmission using TF and Sionna.
        tx_tensor = tf.experimental.dlpack.from_dlpack(tx_tensor.toDlpack())
        No = pow(10., -esno_db / 10.)
        rx_tensor = apply_channel(tx_tensor, No)
        rx_tensor = tf.experimental.dlpack.to_dlpack(rx_tensor)
        rx_tensor = cp.from_dlpack(rx_tensor)        
        
        # Run the fused PUSCH receiver.
        # Note that this is where we set the dynamically changing parameters.
        tb_crcs, tbs = pusch_rx.run(
            rx_slot=rx_tensor,           
            slot=slot_number,
            pusch_configs=pusch_configs
        )
        num_tb_errors["PUSCH Rx"] += int(np.array_equal(tbs[0], tb_input_np) == False)
        
        # Run the neural receiver.
        tbs = neural_rx.run(
            rx_slot=rx_tensor,
            slot=slot_number,
            pusch_configs=pusch_configs
        )
        num_tb_errors["Neural Rx"] += int(np.array_equal(tbs[0], tb_input_np) == False)

        monitor.update(num_tbs=slot_idx + 1, num_tb_errors=num_tb_errors)
        if (np.array(list(num_tb_errors.values())) >= min_num_tb_errors).all():
            break  # Next Es/No value.
    
    monitor.finish_step(num_tbs=slot_idx + 1, num_tb_errors=num_tb_errors)  
monitor.finish()

# Save results using monitor.save (raises on fatal errors)
try:
    out = monitor.save('results', 'neural_receiver_results', fmt='mat')
    print('Saved results to', out)
except Exception as e:
    try:
        out = monitor.save('results', 'neural_receiver_results', fmt='npz')
        print('Saved results to', out)
    except Exception as e2:
        print('Could not save results:', e2)