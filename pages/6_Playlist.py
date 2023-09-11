import os
import sys
import pandas as pd
import streamlit as st

from webui import DEVICE_OPTIONS, MENU_ITEMS, PITCH_EXTRACTION_OPTIONS, i18n, config
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from webui.components import active_subprocess_list
from webui.utils import gc_collect, get_filenames, get_index


from webui.player import PlaylistPlayer
from types import SimpleNamespace
from webui.contexts import SessionStateContext
from webui.audio import SUPPORTED_AUDIO

CWD = os.getcwd()
if CWD not in sys.path:
    sys.path.append(CWD)

def get_models(folder="."):
    fnames = get_filenames(root="./models",folder=folder,exts=["pth","pt"])
    return fnames

def init_inference_state():
    state = SimpleNamespace(
        player=None,
        playlist = get_filenames(exts=SUPPORTED_AUDIO,name_filters=[""],folder="songs"),
        models=get_models(folder="RVC"),
        model_name=None,
        uvr5_models=get_filenames(root="./models",name_filters=["vocal","instrument"]),
        preprocess_models=[""]+get_filenames(root="./models",name_filters=["echo","reverb","noise","karaoke"]),
        
        split_vocal_config=SimpleNamespace(
            agg=10,
            device="cuda" if config.has_gpu else "cpu",
            preprocess_model="",
            uvr5_name=[],
            merge_type="median",
            use_cache=True,
        ),
        vocal_change_config=SimpleNamespace(
            f0_up_key=0,
            f0_method=["rmvpe"],
            index_rate=.75,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=.2,
            protect=0.2,
        ),
        shuffle=False,
        loop=False,
        volume=1.0
    )
    return vars(state)

def refresh_data(state):
    state.uvr5_models = get_filenames(root="./models",name_filters=["vocal","instrument"])
    state.preprocess_models = [""]+get_filenames(root="./models",name_filters=["echo","reverb","noise","karaoke"])
    state.models = get_models(folder="RVC")
    state.playlist = get_filenames(exts=SUPPORTED_AUDIO,name_filters=[""],folder="songs")
    gc_collect()
    return state
    
def render_vocal_separation_form(state):
    with st.form("inference.split_vocals.expander"):
        preprocess_model = st.selectbox(
            i18n("inference.preprocess_model"),
            options=state.preprocess_models,
            index=get_index(state.preprocess_models,state.split_vocal_config.preprocess_model))
        uvr5_name = st.multiselect(
            i18n("inference.uvr5_name"),
            options=state.uvr5_models,
            format_func=lambda item: os.path.basename(item),
            default=[name for name in state.split_vocal_config.uvr5_name if name in state.uvr5_models])
        
        col1, col2 = st.columns(2)
        device = col1.radio(
            i18n("inference.device"),
            disabled=not config.has_gpu,
            options=DEVICE_OPTIONS,horizontal=True,
            index=get_index(DEVICE_OPTIONS,state.split_vocal_config.device))
        merge_type = col1.radio(
            i18n("inference.merge_type"),
            options=["median","mean"],horizontal=True,
            index=get_index(["median","mean"],state.split_vocal_config.merge_type))
        
        agg = col2.slider(i18n("inference.agg"),min_value=0,max_value=20,step=1,value=state.split_vocal_config.agg)
        # use_cache=col2.checkbox(i18n("inference.use_cache"),value=state.use_cache)
        
        if col1.form_submit_button(i18n("inference.save.button"),type="primary"):
            state.split_vocal_config = SimpleNamespace(
                agg=agg,
                device=device,
                preprocess_model=preprocess_model,
                uvr5_name=uvr5_name,
                merge_type=merge_type
            )
        elif len(uvr5_name)<1: st.write(i18n("inference.uvr5_name"))
    return state

def render_vocal_conversion_form(state):
    with st.form("inference.convert_vocals.expander"):
        f0_up_key = st.select_slider(i18n("inference.f0_up_key"),options=[-12,-5,0,7,12],value=state.vocal_change_config.f0_up_key)
        f0_method = st.multiselect(i18n("inference.f0_method"),
                                            options=PITCH_EXTRACTION_OPTIONS,
                                            default=state.vocal_change_config.f0_method)
        resample_sr = st.select_slider(i18n("inference.resample_sr"),
                                            options=[0,16000,24000,22050,40000,44100,48000],
                                            value=state.vocal_change_config.resample_sr)
        index_rate=st.slider(i18n("inference.index_rate"),min_value=0.,max_value=1.,step=.05,value=state.vocal_change_config.index_rate)
        filter_radius=st.slider(i18n("inference.filter_radius"),min_value=0,max_value=7,step=1,value=state.vocal_change_config.filter_radius)
        rms_mix_rate=st.slider(i18n("inference.rms_mix_rate"),min_value=0.,max_value=1.,step=.05,value=state.vocal_change_config.rms_mix_rate)
        protect=st.slider(i18n("inference.protect"),min_value=0.,max_value=.5,step=.01,value=state.vocal_change_config.protect)
        
        if st.form_submit_button(i18n("inference.save.button"),type="primary"):
            state.vocal_change_config = SimpleNamespace(
                f0_up_key=f0_up_key,
                f0_method=f0_method,
                resample_sr=resample_sr,
                index_rate=index_rate,
                filter_radius=filter_radius,
                rms_mix_rate=rms_mix_rate,
                protect=protect
            )
    return state

def update_volume(state):
    def __set_volume():
        if state.player: state.player.set_volume(state.volume)
    return __set_volume

if __name__=="__main__":
    with SessionStateContext("playlist",initial_state=init_inference_state()) as state:

        col1, col2, col3 = st.columns(3)
        state.model_name = col1.selectbox(
            i18n("inference.voice.selectbox"),
            options=state.models,
            index=get_index(state.models,state.model_name),
            format_func=lambda option: os.path.basename(option).split(".")[0]
            )
        state.volume = col2.slider("Volume",min_value=0.0, max_value=1.0,step=0.1, value=state.volume, on_change=update_volume(state))
        state.loop = col3.checkbox("Loop",value=state.loop)
        state.shuffle = col3.checkbox("Shuffle",value=state.shuffle)

        col1, col2, col3, col4 = st.columns(4)

        if col1.button(i18n("inference.refresh_data.button"),use_container_width=True):
            state = refresh_data(state)
            st.experimental_rerun()

        if col2.button("Pause" if state.player else "Play", type="primary",use_container_width=True,
                    disabled=not (state.split_vocal_config.uvr5_name and state.model_name)):
            if state.player is None:
                state.player = PlaylistPlayer(state.playlist,
                                              shuffle=state.shuffle,
                                              loop=state.loop,
                                              volume=state.volume,
                                                model_name=state.model_name,
                                                config=config,
                                                **vars(state.split_vocal_config),
                                                **vars(state.vocal_change_config))
            else:
                if state.player.paused:
                    state.player.resume()
                else:
                    state.player.pause()
            st.experimental_rerun()

        if col3.button("Skip", use_container_width=True,disabled = state.player is None):
            state.player.skip()
            st.experimental_rerun()
        if col4.button("Stop",type="primary",use_container_width=True,disabled = state.player is None):
            if state.player is not None: state.player.stop()
            del state.player
            state.player = None
            st.experimental_rerun()
            
        with st.expander("Settings", expanded=not (state.player and len(state.split_vocal_config.uvr5_name))):
            vs_tab, vc_tab = st.tabs(["Split Vocal", "Vocal Change"])

            with vs_tab:
                state = render_vocal_separation_form(state)

            with vc_tab:
                state = render_vocal_conversion_form(state)

        active_subprocess_list()

        if state.player:
            st.write(state.player)
            df = pd.DataFrame(state.player.playlist,columns=["Songs"])
            st.table(df)