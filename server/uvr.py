import hashlib
import os
from typing import List
from server.utils import bytes2audio
from uvr5_cli import split_audio
from webui.audio import save_input_audio
from webui.downloader import BASE_CACHE_DIR, BASE_MODELS_DIR
from webui import config
from webui.utils import get_filenames

CACHE_DIR = os.path.join(BASE_CACHE_DIR,"temp","uvr")
os.makedirs(CACHE_DIR,exist_ok=True)

def add_basepath(paths: List[str]):
    return [os.path.join(BASE_MODELS_DIR, path) for path in paths]

def list_uvr_models():
    fnames = get_filenames(root=BASE_MODELS_DIR,name_filters=["voc","inst"])
    return [os.path.relpath(name,BASE_MODELS_DIR) for name in fnames]

def list_uvr_denoise_models():
    fnames = get_filenames(root=BASE_MODELS_DIR,name_filters=["echo","reverb","noise","kara"])
    return [os.path.relpath(name,BASE_MODELS_DIR) for name in fnames]

def split_vocals(uvr_models: List[str], preprocess_models: List[str], postprocess_models: List[str], audio_data: str, **kwargs):
    try:
        input_audio = bytes2audio(audio_data)
        if input_audio:
            tempfile = os.path.join(CACHE_DIR,f"{hashlib.md5(audio_data.encode('utf-8')).hexdigest()}.wav")
            save_input_audio(tempfile,input_audio,to_int16=True)
            vocals,instrumental,_ = split_audio(
                model_paths=add_basepath(uvr_models),
                preprocess_models=add_basepath(preprocess_models),
                postprocess_models=add_basepath(postprocess_models),
                audio_path=tempfile,device=config.device,**kwargs)
            return vocals, instrumental
    except Exception as e:
        print(e)
    return None