import os
import socket
import json
import time
import getpass
import numpy as np

def _collect_env_metadata():
    meta = {}
    meta['host'] = socket.gethostname()
    meta['user'] = getpass.getuser()
    meta['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    try:
        import subprocess
        git_rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        meta['git_commit'] = git_rev
    except Exception:
        meta['git_commit'] = None
    return meta


def save_results_mat(out_dir, base_name, monitor, config=None, fmt='mat'):
    """Save simulation results and metadata to disk.

    Args:
        out_dir: directory to save into
        base_name: base filename without extension
        monitor: SimulationMonitor instance (must expose current_esno_db_range and bler)
        config: dict of configuration parameters (optional)
        fmt: 'mat' or 'npz'
    Returns: path to saved file
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = _collect_env_metadata()
    # Prepare results
    esno = np.array(monitor.current_esno_db_range)
    bler = {case.replace(' ', '_'): np.array(monitor.bler[case]) for case in monitor.cases}

    # Merge config into metadata
    if config is None:
        config = {}
    meta['config'] = config

    ts = time.strftime('%Y%m%dT%H%M%S')
    if fmt == 'mat':
        try:
            from scipy.io import savemat
        except Exception as e:
            raise RuntimeError('scipy required to save mat files') from e
        out_path = os.path.join(out_dir, f"{base_name}_{ts}.mat")
        mat_dict = {
            'esno_db_range': esno,
            'metadata': meta
        }
        # add bler arrays as separate fields
        for k, v in bler.items():
            mat_dict['bler_' + k] = v
        savemat(out_path, mat_dict)
        return out_path
    else:
        out_path = os.path.join(out_dir, f"{base_name}_{ts}.npz")
        np.savez(out_path, esno_db_range=esno, **{('bler_' + k): v for k, v in bler.items()})
        # save metadata as json sidecar
        with open(out_path + '.meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        return out_path
