import argparse
import subprocess
from fastapi import FastAPI
from server.rvc import STATUS, convert_vocals, list_rvc_models
from server.types import RVCInferenceParams, UVRInferenceParams
from server.utils import audio2bytes, gc_collect, to_response
from server.uvr import list_uvr_denoise_models, list_uvr_models, split_vocals
from webui.utils import get_optimal_threads

server = FastAPI()

@server.get("/")
async def get_status():
    return to_response(STATUS)

@server.get("/rvc")
async def get_rvc():
    return list_rvc_models()

@server.post("/rvc")
async def rvc_infer(body: RVCInferenceParams):
    gc_collect()
    output_audio = convert_vocals(**vars(body))
    if output_audio: return audio2bytes(*output_audio)
    return ""

# @server.post("/uvr/rvc")
# async def uvr_rvc_infer(body: UVRRVCInferenceParams):
#     gc_collect()
#     response = {}
#     result = split_vocals(audio_data=body.audio_data,**body.uvr_params.__dict__)
#     if result:
#         vocals, instrumentals = result
#         output_audio = convert_vocals(audio_data=audio2bytes(*vocals),**body.rvc_params.__dict__)
#         if output_audio:
#             response["vocals"] = audio2bytes(*output_audio)
#             response["instrumentals"] = audio2bytes(*instrumentals)
#     return response

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
    gc_collect()
    response = {}
    result = split_vocals(**vars(body))
    if result:
        vocals, instrumentals = result
        response["vocals"] = audio2bytes(*vocals)
        response["instrumentals"] = audio2bytes(*instrumentals)
    return response

def main():
    parser = argparse.ArgumentParser(description="Start API server to run RVC and UVR")
    parser.add_argument( "-w", "--workers", type=int, default=get_optimal_threads(1), help="Number of workers to use", required=False)
    parser.add_argument( "-r", "--reload", action="store_true", default=False, help="Reloads on change", required=False)
    parser.add_argument( "-p", "--port", type=int, default=5555, help="Port of server", required=False)
    parser.add_argument( "-d", "--host", type=str, default="localhost", help="Domain of server", required=False)
    args = parser.parse_args()

    cmd=f"python -m uvicorn api:server {'--reload' if args.reload else ''} --workers={args.workers} --port={args.port} --host={args.host}"
    subprocess.call(cmd)

if __name__ == "__main__":
    main()