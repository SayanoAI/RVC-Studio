from datetime import datetime
import gc
import glob
# from importlib.util import LazyLoader, find_spec, module_from_spec
import multiprocessing
import os
import sys
from types import SimpleNamespace
import numpy as np
import psutil
import torch

from webui import get_cwd

torch.manual_seed(1337)
CWD = get_cwd()

def get_subprocesses(pid = os.getpid()):
    # Get a list of all subprocesses started by the current process
    subprocesses = psutil.Process(pid).children(recursive=True)
    python_processes = [p for p in subprocesses if p.status()=="running"]
    for p in python_processes:
        cpu_percent = p.cpu_percent()
        memory_percent = p.memory_percent()
        process = SimpleNamespace(**{
            'pid': p.pid,
            "name": p.name(),
            'cpu_percent': f"{cpu_percent:.2f}%",
            'memory_percent': f"{memory_percent:.2f}%",
            'status': p.status(),
            'time_started': datetime.fromtimestamp(p.create_time()).isoformat(),
            'kill': p.kill
            })
        yield process

def get_filenames(root=CWD,folder="**",exts=["*"],name_filters=[""]):
    fnames = []
    for ext in exts:
        fnames.extend(glob.glob(f"{root}/{folder}/*.{ext}",recursive=True))
    return sorted([ele for ele in fnames if any([nf.lower() in ele.lower() for nf in name_filters])])

def get_rvc_models():
    fnames = get_filenames(root=os.path.join(CWD,"models"),folder="RVC",exts=["pth","pt"])
    return fnames

def get_index(arr,value):
    if arr is not None:
        if value in arr: return arr.index(value)
        elif value is not None:
            for i,item in enumerate(arr):
                k1, k2 = str(item), str(value)
                if (k1 in k2) or (k2 in k1): return i
    return 0

def gc_collect():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # import streamlit as st
    # st.cache_resource.clear()
    # st.cache_data.clear()
    gc.collect()

# def lazyload(name):
#     if name in sys.modules:
#         return modules[name]
#     else:
#         spec = find_spec(name)
#         loader = LazyLoader(spec.loader)
#         module = module_from_spec(spec)
#         modules[name] = module
#         loader.exec_module(module)
#         return module
    
def get_optimal_torch_device(index = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            f"cuda:{index % torch.cuda.device_count()}"
        )  # Very fast
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_optimal_threads(offset=0):
    cores = multiprocessing.cpu_count() - offset
    return max(np.floor(cores * (1-psutil.cpu_percent())),1)