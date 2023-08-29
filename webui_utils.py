import gc
import io
import os
import torch
import librosa
import numpy as np
# import logging
from fairseq import checkpoint_utils
from vc_infer_pipeline import VC
from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
import glob
import streamlit as st
import soundfile as sf
import locale
import json
from scipy.io import wavfile
import io

# logging.getLogger("numba").setLevel(logging.WARNING)
# logging.getLogger("markdown_it").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)
# logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.manual_seed(114514)

MAX_INT16 = 32768

class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[
                0
            ]  # getlocale can't identify the system's language ((None, None))
        if not os.path.exists(f"./lib/i18n/{language}.json"):
            language = "en_US"
        self.language = language
        # print("Use Language:", language)
        self.language_map = self.load_language_list(language)
        self.print()

    def __call__(self, key):
        return self.language_map.get(key, key)

    def print(self):
        print("Use Language:", self.language)

    @staticmethod
    def load_language_list(language):
        with open(f"./i18n/{language}.json", "r", encoding="utf-8") as f:
            language_list = json.load(f)
        return language_list

@st.cache_data
def load_config():
    return Config(), I18nAuto()

config, i18n = load_config()

@st.cache_data
def get_index(arr,value): return arr.index(value) if value in arr else 0

def gc_collect():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    st.cache_resource.clear()
    st.cache_data.clear()

def get_vc(model_path,device="cpu"):
    print("loading %s" % model_path)
    cpt = torch.load(model_path, map_location=device)
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    hubert_model = load_hubert()
    return {"vc": vc, "cpt": cpt, "net_g": net_g, "hubert_model": hubert_model}

def load_hubert(device=config.device):
    try:
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            ["./models/hubert_base.pt"],
            suffix="",
        )
        hubert_model = models[0]
        hubert_model = hubert_model.to(device)
        if config.is_half:
            hubert_model = hubert_model.half()
        else:
            hubert_model = hubert_model.float()
        hubert_model.eval()
        return hubert_model
    except Exception as e:
        print(e)
        return None

def vc_single(
        cpt=None,
        net_g=None,
        vc=None,
        hubert_model=None,
    sid=0,
    input_audio=None,
    input_audio_path=None,
    f0_up_key=0,
    f0_file=None,
    f0_method="crepe",
    file_index="",  # .index file
    index_rate=.75,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=.25,
    protect=0.33,
):
    if hubert_model == None:
        hubert_model = load_hubert()

    if not (cpt and net_g and vc and hubert_model):
        return None

    tgt_sr = cpt["config"][-1]
    
    version = cpt.get("version", "v1")

    if input_audio is None and input_audio_path is None:
        return None
    f0_up_key = int(f0_up_key)
    try:
        audio = input_audio[0] if input_audio is not None else load_input_audio(input_audio_path, 16000)
        
        audio,_ = remix_audio((audio,input_audio[1] if input_audio is not None else 16000), target_sr=16000, norm=True,  to_mono=True)

        times = [0, 0, 0]
        if_f0 = cpt.get("f0", 1)
        
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            f0_file=f0_file,
        )
        
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        print(index_info)
        
        return (audio_opt, resample_sr if resample_sr >= 16000 and tgt_sr != resample_sr else tgt_sr)
    except Exception as info:
        print(info)
        return None

@st.cache_data(show_spinner=False)
def load_input_audio(fname,sr=None,**kwargs):
    sound = librosa.load(fname,sr=sr,**kwargs)
    print(f"loading sound {fname} {sound[0].shape} {sound[1]}")
    return sound

@st.cache_data
def audio_to_bytes(audio,sr):
    byte_io = io.BytesIO(bytes())
    wavfile.write(byte_io, sr, audio)
    return byte_io.read()
    
def save_input_audio(fname,input_audio,sr=None,to_int16=False):
    print(f"saving sound to {fname}")
    audio=np.array(input_audio[0],dtype="float32")
    if to_int16:
        max_a = np.abs(audio).max() * .99
        if max_a<1:
            audio=(audio*max_a*MAX_INT16)
        audio=audio.astype("int16")
    try:        
        sf.write(fname, audio, sr if sr else input_audio[1])
        return True
    except:
        return False

def get_filenames(root=".",folder="**",exts=["*"],name_filters=[""]):
    fnames = []
    for ext in exts:
        fnames.extend(glob.glob(f"{root}/{folder}/*.{ext}",recursive=True))
    return sorted([ele for ele in fnames if any([nf.lower() in ele.lower() for nf in name_filters])])

@st.cache_data(show_spinner=False)
def remix_audio(input_audio,target_sr=None,norm=False,to_int16=False,resample=False,to_mono=False,axis=0,**kwargs):
    audio = np.array(input_audio[0],dtype="float32")
    if target_sr is None: target_sr=input_audio[1]

    print(f"before remix: shape={audio.shape}, max={audio.max()}, min={audio.min()}, mean={audio.mean()} sr={input_audio[1]}")
    if resample or input_audio[1]!=target_sr:
        audio = librosa.core.resample(np.array(input_audio[0],dtype="float32"),orig_sr=input_audio[1],target_sr=target_sr,**kwargs)
    
    if to_mono and audio.ndim>1: audio=audio.mean(axis)

    if norm: audio = librosa.util.normalize(audio)

    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1: audio /= audio_max
    
    if to_int16: audio = np.clip(audio * MAX_INT16, a_min=-MAX_INT16+1, a_max=MAX_INT16-1).astype("int16")
    print(f"after remix: shape={audio.shape}, max={audio.max()}, min={audio.min()}, mean={audio.mean()}, sr={target_sr}")

    return audio, target_sr

@st.cache_data
def merge_audio(audio1,audio2,sr=40000):
    print(f"merging audio audio1={audio1[0].shape,audio1[1]} audio2={audio2[0].shape,audio2[1]} sr={sr}")
    m1,_=remix_audio(audio1,target_sr=sr)
    m2,_=remix_audio(audio2,target_sr=sr)
    
    maxlen = max(len(m1),len(m2))
    m1=librosa.util.pad_center(m1,maxlen)
    m2=librosa.util.pad_center(m2,maxlen)

    mixed = librosa.util.stack([m1,m2],0)

    return remix_audio((mixed,sr),to_int16=True,norm=True,to_mono=True,axis=0)