from typing import List, Tuple
import requests
import os

RVC_DOWNLOAD_LINK = 'https://huggingface.co/datasets/SayanoAI/RVC-Studio/resolve/main/'

BASE_DIR = os.getcwd()
BASE_MODELS_DIR = os.path.join(BASE_DIR,"models")

MDX_MODELS = ["MDXNET/Kim_Vocal_2.onnx","MDXNET/UVR-MDX-NET-vocal_FT.onnx"]
VR_MODELS = ["UVR/UVR-DeEcho-DeReverb.pth","UVR/HP5-vocals+instrumentals.pth","UVR/6_HP-Karaoke-UVR.pth"]
RVC_MODELS = [
    "RVC/Sayano.pth","RVC/.index/added_IVF1063_Flat_nprobe_1_Sayano_v2.index",
    # "RVC/Mae.pth","RVC/.index/added_IVF1063_Flat_nprobe_1_Mae_v2.index",
    # "RVC/Fuji.pth","RVC/.index/added_IVF1063_Flat_nprobe_1_Fuji_v2.index",
    "RVC/Yuuko.pth","RVC/.index/added_IVF1063_Flat_nprobe_1_Yuuko_v2.index"]
BASE_MODELS = ["hubert_base.pt", "rmvpe.pt"]
VITS_MODELS = ["VITS/pretrained_ljs.pth"]
PRETRAINED_MODELS = [
    "pretrained_v2/D48k.pth",
    "pretrained_v2/G48k.pth",
    "pretrained_v2/D40k.pth",
    "pretrained_v2/G40k.pth",
    "pretrained_v2/f0D48k.pth",
    "pretrained_v2/f0G48k.pth",
    "pretrained_v2/f0D40k.pth",
    "pretrained_v2/f0G40k.pth"]

def download_file(params: Tuple[str, str]):
    model_path, download_link = params
    if os.path.isfile(model_path): raise FileExistsError(f"{model_path} already exists!")
    
    with requests.get(download_link,stream=True) as r:
        r.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def download_link_generator(download_link: str,model_list: List[str]):
    for model in model_list:
        model_path = os.path.join(BASE_MODELS_DIR,model)
        yield (model_path, f"{download_link}{model}")