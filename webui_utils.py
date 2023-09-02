import gc
import os
import torch
import librosa
# import logging
from config import Config
import glob
import streamlit as st

import locale
import json

from web_utils.audio import remix_audio

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

def get_filenames(root=".",folder="**",exts=["*"],name_filters=[""]):
    fnames = []
    for ext in exts:
        fnames.extend(glob.glob(f"{root}/{folder}/*.{ext}",recursive=True))
    return sorted([ele for ele in fnames if any([nf.lower() in ele.lower() for nf in name_filters])])

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