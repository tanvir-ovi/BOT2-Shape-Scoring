"""Result folders, logging, file helpers and control of the Colab runtime."""
import os
import sys
import json
import time
import uuid
import hashlib
import platform
import importlib
from types import SimpleNamespace

RESULT_FOLDERS = ('data', 'runs', 'predictions', 'metrics', 'statistics', 'embeddings', 'figures', 'logs')


def make_paths(results_dir):
    P = SimpleNamespace(root=results_dir)
    for name in RESULT_FOLDERS:
        setattr(P, name, os.path.join(results_dir, name))
        os.makedirs(getattr(P, name), exist_ok=True)
    return P


def atomic_save(path, writer):
    """Writes through a temporary file in the same folder, so an interrupted write never leaves a partial file."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    base, ext = os.path.splitext(path)
    tmp = f'{base}.tmp{uuid.uuid4().hex[:8]}{ext}'
    writer(tmp)
    os.replace(tmp, path)


def _plain(o):
    if hasattr(o, 'item'):
        return o.item()
    if hasattr(o, 'tolist'):
        return o.tolist()
    return str(o)


def save_json(path, obj):
    def write(p):
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=1, default=_plain)
    atomic_save(path, write)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_csv(df, path, **kw):
    atomic_save(path, lambda p: df.to_csv(p, **kw))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def parallel_map(fn, items, workers=None):
    from multiprocessing import Pool
    with Pool(workers or os.cpu_count()) as pool:
        return pool.map(fn, items, chunksize=16)


class Log:
    """Prints a message and appends it with a time stamp to a log file on Drive."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, *parts, quiet=False):
        msg = ' '.join(str(p) for p in parts)
        if not quiet:
            print(msg, flush=True)
        try:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(time.strftime('%Y-%m-%d %H:%M:%S  ') + msg + '\n')
        except OSError:
            pass


def environment():
    info = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ('numpy', 'pandas', 'sklearn', 'scipy', 'skimage', 'cv2', 'matplotlib', 'torch', 'timm', 'transformers',
                 'safetensors'):
        try:
            info[name] = importlib.import_module(name).__version__
        except Exception:
            info[name] = None
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
            info['cuda'] = torch.version.cuda
    except Exception:
        pass
    return info


def code_hashes(package_dir):
    return {f: sha256(os.path.join(package_dir, f)) for f in sorted(os.listdir(package_dir)) if f.endswith('.py')}


def quiet_libraries():
    """Hides download progress bars and library warnings, so that the notebook output is the run log and the results.
    Call before torch, timm or transformers are imported."""
    import warnings
    for var in ('HF_HUB_DISABLE_PROGRESS_BARS', 'HF_HUB_DISABLE_IMPLICIT_TOKEN'):
        os.environ.setdefault(var, '1')
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
    os.environ.setdefault('HF_HUB_VERBOSITY', 'error')
    for category in (UserWarning, FutureWarning, DeprecationWarning):
        warnings.filterwarnings('ignore', category=category)


def in_colab():
    return 'google.colab' in sys.modules


def release_runtime(log=None):
    """Writes pending Drive files, then deletes the Colab runtime so that no compute units are used after the run."""
    if log is not None:
        log('Releasing the runtime.')
    if not in_colab():
        return
    from google.colab import drive, runtime
    try:
        drive.flush_and_unmount()
    except Exception as e:
        print('Drive flush failed:', e, flush=True)
    runtime.unassign()


def release_runtime_on_error(log=None):
    """From now on, an exception in any cell shows its traceback and then releases the runtime."""
    if not in_colab():
        return
    from IPython import get_ipython
    shell = get_ipython()
    if shell is None:
        return

    def handler(shell, etype, value, tb, tb_offset=None):
        shell.showtraceback((etype, value, tb), tb_offset=tb_offset)
        if log is not None:
            log(f'Stopped by {etype.__name__}: {value}')
        release_runtime(log)

    shell.set_custom_exc((Exception,), handler)
