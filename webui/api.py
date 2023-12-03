import requests

from lib.audio import audio2bytes, bytes2audio, load_input_audio
from webui import RVC_INFERENCE_URL, UVR_INFERENCE_URL
from lib.utils import gc_collect

def get_rvc_models():
    fnames = []
    try:
        with requests.get(RVC_INFERENCE_URL) as req:
            if req.status_code==200:
                fnames = req.json()
    except Exception as e:
        print(e)
    return fnames

def get_uvr_models():
    fnames = []
    try:
        with requests.get(UVR_INFERENCE_URL) as req:
            if req.status_code==200:
                fnames = req.json()
    except Exception as e:
        print(e)
    return fnames

def get_uvr_preprocess_models():
    fnames = []
    try:
        with requests.get(f"{UVR_INFERENCE_URL}/preprocess") as req:
            if req.status_code==200:
                fnames = req.json()
    except Exception as e:
        print(e)
    return fnames

def get_uvr_postprocess_models():
    fnames = []
    try:
        with requests.get(f"{UVR_INFERENCE_URL}/postprocess") as req:
            if req.status_code==200:
                fnames = req.json()
    except Exception as e:
        print(e)
    return fnames

def split_vocals(audio_path,**args):
    input_audio = load_input_audio(audio_path)
    audio_data = audio2bytes(*input_audio)
    body = dict(audio_data=audio_data,**args)
    with requests.post(UVR_INFERENCE_URL,json=body) as req:
        if req.status_code==200:
            data = req.json()
            vocals = data.get("vocals")
            if vocals: vocals = bytes2audio(vocals)
            instrumentals = data.get("instrumentals")
            if instrumentals: instrumentals = bytes2audio(instrumentals)
            return vocals, instrumentals, input_audio

    return None, None, None

def convert_vocals(model_name,input_audio,**kwargs):
    gc_collect()
    audio_data = audio2bytes(*input_audio)
    body = dict(name=model_name,audio_data=audio_data,**kwargs)
    
    with requests.post(RVC_INFERENCE_URL,json=body) as req:
        if req.status_code==200:
            response = req.json()
            audio = bytes2audio(response["data"])
            return audio