from datetime import datetime
import gc
import glob
# from importlib.util import LazyLoader, find_spec, module_from_spec
import multiprocessing
import os
import platform
from time import sleep
import numpy as np
import psutil
import requests
import torch
from lib import ObjectNamespace, get_cwd

torch.manual_seed(1337)
CWD = get_cwd()

def get_subprocesses(pid = os.getpid()):
    # Get a list of all subprocesses started by the current process
    subprocesses = psutil.Process(pid).children(recursive=True)
    python_processes = [p for p in subprocesses if p.status()=="running"]
    for p in python_processes:
        cpu_percent = p.cpu_percent()
        memory_percent = p.memory_percent()
        process = ObjectNamespace(**{
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
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.set_threshold(100,10,1)
    gc.collect()
    
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

def pid_is_active(pid: int):        
    """ Check For the existence of a unix pid. https://stackoverflow.com/a/568285"""
    try:
        if platform.system() == "Windows":
            return psutil.pid_exists(pid)
        elif platform.system() == "Linux":
            os.kill(pid, 0)
    except Exception as e:
        print(e)
        return False
    else:
        return True
    
def poll_url(url,timeout=10):
    for i in range(timeout): # wait for server to start up
        try:
            with requests.get(url) as req:
                if req.status_code==200: return True
        except Exception:
            sleep(1.)
            print(f"waited {i+1} seconds...")
    return False