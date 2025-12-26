import os
import socket
import json
import time
import getpass
import numpy as np

# Optional cupy support: treat cupy arrays like numpy arrays when sanitizing
try:
    import cupy as _cp
except Exception:
    _cp = None

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
    # Helper to sanitize config/metadata for MATLAB/JSON saving
    def _sanitize(obj):
        # numpy arrays -> lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # numpy scalars
        if isinstance(obj, (np.generic,)):
            try:
                return obj.item()
            except Exception:
                return float(obj)

        # cupy arrays -> convert to numpy then list
        if _cp is not None:
            try:
                if isinstance(obj, _cp.ndarray):
                    return _cp.asnumpy(obj).tolist()
            except Exception:
                pass

        # containers
        if isinstance(obj, (list, tuple)):
            return [_sanitize(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _sanitize(v) for k, v in obj.items()}

        # bytes -> str
        try:
            if isinstance(obj, bytes):
                return obj.decode('utf-8', errors='replace')
        except Exception:
            pass

        # Basic JSON types pass-through
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        # Fallback: stringify unknown objects
        try:
            return str(obj)
        except Exception:
            return None

    try:
        meta['config'] = _sanitize(config)
    except Exception:
        meta['config'] = str(config)

    ts = time.strftime('%Y%m%dT%H%M%S')
    if fmt == 'mat':
        try:
            from scipy.io import savemat
        except Exception as e:
            raise RuntimeError('scipy required to save mat files') from e
        out_path = os.path.join(out_dir, f"{base_name}_{ts}.mat")
        # Ensure bler arrays are plain numpy arrays
        mat_dict = {
            'esno_db_range': np.asarray(esno),
            'metadata': meta
        }
        # add bler arrays as separate fields (force numpy arrays)
        for k, v in bler.items():
            try:
                mat_dict['bler_' + k] = np.asarray(v)
            except Exception:
                mat_dict['bler_' + k] = np.asarray(list(v))

        # Debug: print what will be saved (keys and shapes)
        try:
            keys_info = {k: (None if not hasattr(v, 'shape') else tuple(v.shape)) for k, v in mat_dict.items()}
        except Exception:
            keys_info = {k: type(v).__name__ for k, v in mat_dict.items()}
        print(f"Saving MAT file with keys: {list(mat_dict.keys())}")
        print(f"Keys info: {keys_info}")

        savemat(out_path, mat_dict)
        return out_path
    else:
        out_path = os.path.join(out_dir, f"{base_name}_{ts}.npz")
        # Ensure arrays are numpy arrays
        np.savez(out_path, esno_db_range=np.asarray(esno), **{('bler_' + k): np.asarray(v) for k, v in bler.items()})
        # save metadata as json sidecar using safe fallback for unknown types
        with open(out_path + '.meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        return out_path
