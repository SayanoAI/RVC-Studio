from functools import lru_cache
import os
from server.utils import bytes2audio
from vc_infer_pipeline import get_vc, vc_single
from webui.downloader import BASE_MODELS_DIR
from webui import config
from server import STATUS
from webui.utils import get_filenames

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
            input_audio = bytes2audio(audio_data)
            return vc_single(input_audio=input_audio,**model,**kwargs)
    except Exception as e:
        print(e)
    
    return None