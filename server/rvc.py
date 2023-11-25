from functools import lru_cache
import hashlib
import json
import os
from server.utils import bytes2audio
from vc_infer_pipeline import get_vc, vc_single
from webui.audio import load_input_audio, save_input_audio
from webui.downloader import BASE_CACHE_DIR, BASE_MODELS_DIR
from webui import config
from server import STATUS
from webui.utils import get_filenames

CACHE_DIR = os.path.join(BASE_CACHE_DIR,"temp","rvc")
os.makedirs(CACHE_DIR,exist_ok=True)

@lru_cache
def load_model(name: str):
    try:
        path = os.path.join(BASE_MODELS_DIR,"RVC",f"{name}.pth")
        model = get_vc(path,config)
        STATUS.rvc[name] = model
        return model
    except Exception as e:
        print(e)
    return False

def list_rvc_models():
    fnames = get_filenames(root=BASE_MODELS_DIR,folder="RVC",exts=["pth","pt"])
    return [os.path.basename(os.path.splitext(name)[0]) for name in fnames]

def convert_vocals(name: str, audio_data: str, **kwargs):
    try:
        if model:=load_model(name):
            tempfile = os.path.join(CACHE_DIR,f"{hashlib.md5(json.dumps(dict(name=name,audio_data=audio_data,**kwargs)).encode('utf-8')).hexdigest()}.wav")
            if os.path.isfile(tempfile): return load_input_audio(tempfile)

            input_audio = bytes2audio(audio_data)
            output_audio = vc_single(input_audio=input_audio,**model,**kwargs)
            save_input_audio(tempfile,output_audio,to_int16=True)
            return output_audio

    except Exception as e:
        print(e)
    
    return None