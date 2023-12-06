import argparse
import subprocess
from fastapi import FastAPI
from lib.audio import audio2bytes
from server import STATUS
from server.rvc import convert_vocals, list_rvc_models
from server.types import RVCInferenceParams, UVRInferenceParams
from server.uvr import list_uvr_denoise_models, list_uvr_models, split_vocals
from lib.utils import get_optimal_threads, gc_collect
from lib import config

server = FastAPI()

@server.get("/")
async def get_status():
    STATUS.rvc=list_rvc_models()
    STATUS.uvr=list_uvr_models()
    STATUS.denoise=list_uvr_denoise_models()
    gc_collect()
    return STATUS

@server.get("/rvc")
async def get_rvc():
    gc_collect()
    return list_rvc_models()

@server.post("/rvc")
async def rvc_infer(body: RVCInferenceParams):
    response = {}
    gc_collect()
    output_audio = convert_vocals(**vars(body))
    if output_audio: response["data"] = audio2bytes(*output_audio)
    gc_collect()
    return response

@server.get("/uvr")
async def get_uvr():
    gc_collect()
    return list_uvr_models()

@server.get("/uvr/preprocess")
async def get_uvr_preprocess():
    gc_collect()
    return list_uvr_denoise_models()

@server.get("/uvr/postprocess")
async def get_uvr_postprocess():
    gc_collect()
    return list_uvr_denoise_models()

@server.post("/uvr")
async def uvr_infer(body: UVRInferenceParams):
    response = {}
    gc_collect()
    result = split_vocals(**vars(body))
    if result:
        vocals, instrumentals = result
        response["vocals"] = audio2bytes(*vocals)
        response["instrumentals"] = audio2bytes(*instrumentals)
    gc_collect()
    return response

def main():
    gc_collect()
    parser = argparse.ArgumentParser(description="Start API server to run RVC and UVR")
    parser.add_argument( "-w", "--workers", type=int, default=get_optimal_threads(1), help="Number of workers to use", required=False)
    parser.add_argument( "-r", "--reload", action="store_true", default=False, help="Reloads on change", required=False)
    parser.add_argument( "-p", "--port", type=int, default=5555, help="Port of server", required=False)
    parser.add_argument( "-d", "--host", type=str, default="localhost", help="Domain of server", required=False)
    args = parser.parse_args()

    cmd=f"{config.python_cmd} -m uvicorn api:server {'--reload' if args.reload else ''} --workers={args.workers} --port={args.port} --host={args.host}"
    subprocess.call(cmd)

if __name__ == "__main__":
    main()