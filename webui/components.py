import json
import os
from types import SimpleNamespace
from typing import Tuple
import streamlit as st

from webui import PITCH_EXTRACTION_OPTIONS, i18n
from webui.contexts import ProgressBarContext
from webui.downloader import save_file, save_file_generator
from webui.utils import gc_collect, get_filenames, get_index, get_subprocesses

def __default_mapper(x: Tuple[str,any]):
     return x

def file_uploader_form(save_dir, title="Upload your files", types=None, params_mapper=__default_mapper, **kwargs):
    with st.container():
        with st.form(save_dir+title, clear_on_submit=True):
            files = st.file_uploader(title,type=types,**kwargs)
            if st.form_submit_button("Save") and files is not None:
                file_list = [params_mapper(params)
                                for params in save_file_generator(save_dir,files if isinstance(files, list) else [files])]
                with ProgressBarContext(file_list,save_file,"saving files...") as pb:
                    pb.run()
                    del file_list
                    st.toast("Successfully saved files!")

def active_subprocess_list():
    with st.expander(i18n("process.pids")):
        for p in get_subprocesses():
            col1,col2,col3,col4=st.columns(4)
            try:
                col1.write(p.pid)
                col2.write(p.name)
                col3.write(p.time_started)
                if col4.button(i18n("process.kill_one_pid"),key=f"process.kill_one_pid.{p.pid}"):
                    for c in get_subprocesses(p.pid):
                        c.kill()
                    p.kill()
                    gc_collect()
                    st.experimental_rerun()
            except Exception as e:
                print(e)

def initial_vocal_separation_params(folder=None):
    if folder:
        config_file = os.path.join(os.getcwd(),"configs",folder,"vocal_separation_params.json")
        os.makedirs(os.path.dirname(config_file),exist_ok=True)
        if os.path.isfile(config_file):
            with open(config_file,"r") as f:
                data = json.load(f)
                return SimpleNamespace(**data)

    return SimpleNamespace(
        preprocess_models=[],
        postprocess_models=[],
        agg=10,
        merge_type="median",
        model_paths=[],
        use_cache=True,
    )
def save_vocal_separation_params(folder,data):
    config_file = os.path.join(os.getcwd(),"configs",folder,"vocal_separation_params.json")
    os.makedirs(os.path.dirname(config_file),exist_ok=True)
    with open(config_file,"w") as f:
        return f.write(json.dumps(data,indent=2))
        
def vocal_separation_form(state):
    uvr5_models=get_filenames(root="./models",name_filters=["vocal","instrument"])
    uvr5_denoise_models=get_filenames(root="./models",name_filters=["echo","reverb","noise"])
    
    state.preprocess_models = st.multiselect(
            i18n("inference.preprocess_model"),
            options=uvr5_denoise_models,
            default=[name for name in state.preprocess_models if name in uvr5_denoise_models])
    state.model_paths = st.multiselect(
        i18n("inference.model_paths"),
        options=uvr5_models,
        format_func=lambda item: os.path.basename(item),
        default=[name for name in state.model_paths if name in uvr5_models])
    state.postprocess_models = st.multiselect(
            i18n("inference.postprocess_model"),
            options=uvr5_denoise_models,
            default=[name for name in state.postprocess_models if name in uvr5_denoise_models])
    col1, col2, col3 = st.columns(3)
    
    state.merge_type = col2.radio(
        i18n("inference.merge_type"),
        options=["median","mean"],horizontal=True,
        index=get_index(["median","mean"],state.merge_type))
    state.agg = col1.slider(i18n("inference.agg"),min_value=0,max_value=20,step=1,value=state.agg)
    state.use_cache = col3.checkbox(i18n("inference.use_cache"),value=state.use_cache)
    return state

def initial_voice_conversion_params(folder=None):
    if folder:
        config_file = os.path.join(os.getcwd(),"configs",folder,"voice_conversion_params.json")
        os.makedirs(os.path.dirname(config_file),exist_ok=True)
        if os.path.isfile(config_file):
            with open(config_file,"r") as f:
                data = json.load(f)
                return SimpleNamespace(**data)
            
    return SimpleNamespace(
        f0_up_key=0,
        f0_method=["rmvpe"],
        f0_autotune=False,
        merge_type="median",
        index_rate=.75,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=.2,
        protect=0.2,
        )
def save_voice_conversion_params(folder,data):
    config_file = os.path.join(os.getcwd(),"configs",folder,"voice_conversion_params.json")
    os.makedirs(os.path.dirname(config_file),exist_ok=True)
    with open(config_file,"w") as f:
        return f.write(json.dumps(data,indent=2))
def voice_conversion_form(state):
    state.f0_up_key = st.slider(i18n("inference.f0_up_key"),min_value=-12,max_value=12,value=state.f0_up_key,step=1)
    state.f0_method = st.multiselect(i18n("inference.f0_method"),
                                        options=PITCH_EXTRACTION_OPTIONS,
                                        default=state.f0_method)
    col1, col2 = st.columns(2)
    state.merge_type = col1.radio(
        i18n("inference.merge_type"),
        options=["median","mean"],horizontal=True,
        index=get_index(["median","mean"],state.merge_type))
    state.f0_autotune = col2.checkbox(i18n("inference.f0_autotune"),value=state.f0_autotune)
    state.resample_sr = st.select_slider(i18n("inference.resample_sr"),
                                        options=[0,16000,24000,22050,40000,44100,48000],
                                        value=state.resample_sr)
    state.index_rate=st.slider(i18n("inference.index_rate"),min_value=0.,max_value=1.,step=.05,value=state.index_rate)
    state.filter_radius=st.slider(i18n("inference.filter_radius"),min_value=0,max_value=7,step=1,value=state.filter_radius)
    state.rms_mix_rate=st.slider(i18n("inference.rms_mix_rate"),min_value=0.,max_value=1.,step=.05,value=state.rms_mix_rate)
    state.protect=st.slider(i18n("inference.protect"),min_value=0.,max_value=.5,step=.01,value=state.protect)
    return state

