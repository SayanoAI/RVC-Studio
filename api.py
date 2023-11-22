from fastapi import FastAPI
from server.rvc import STATUS, convert_vocals, list_rvc_models
from server.types import RVCInferenceParams, UVRInferenceParams
from server.utils import audio2bytes, to_response
from server.uvr import list_uvr_denoise_models, list_uvr_models, split_vocals

server = FastAPI()

@server.get("/")
async def get_status():
    return to_response(STATUS)

@server.get("/rvc")
async def get_rvc():
    return list_rvc_models()

@server.post("/rvc")
async def rvc_infer(body: RVCInferenceParams):
    output_audio = convert_vocals(**body.__dict__)
    if output_audio: return audio2bytes(*output_audio)
    return ""

@server.get("/uvr")
async def get_uvr():
    return list_uvr_models()

@server.get("/uvr/preprocess")
async def get_uvr_preprocess():
    return list_uvr_denoise_models()

@server.get("/uvr/postprocess")
async def get_uvr_postprocess():
    return list_uvr_denoise_models()

@server.post("/uvr")
async def uvr_infer(body: UVRInferenceParams):
    response = {}
    result = split_vocals(**body.__dict__)
    if result:
        vocals, instrumentals = result
        response["vocals"] = audio2bytes(*vocals)
        response["instrumentals"] = audio2bytes(*instrumentals)
    return response