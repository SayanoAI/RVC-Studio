import gc
from io import BytesIO
import base64

import numpy as np
import torch

def to_response(item: dict, filter: str=""):
    response = {}
    for (k,v) in item.items():
        if filter not in k: continue
        if hasattr(v,"keys"): response[k]=v.keys()
        if hasattr(v,"__dict__"): response[k]=v.__dict__.keys()
        else:
            sv = str(v)
            if len(sv)<32: response[k]=v
            else: response[k]=sv[:32]
    return response

def bytes2audio(data: str):
    try:
        iofile = BytesIO(base64.b64decode(data))
        decoded = np.load(iofile)
        return decoded["audio"], decoded["sr"]+0
    except Exception as e:
        print(e)
    return None

def audio2bytes(audio: np.array, sr: int):
    try:
        iofile = BytesIO()
        np.savez_compressed(iofile,audio=audio,sr=sr)
        return base64.b64encode(iofile.getvalue()).decode("utf-8")
    except Exception as e:
        print(e)
    return ""

def gc_collect():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.set_threshold(100,10,1)
    gc.collect()