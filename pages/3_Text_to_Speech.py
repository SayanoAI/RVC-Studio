import hashlib
import os
import sys
import streamlit as st

from webui import DEVICE_OPTIONS, MENU_ITEMS, TTS_MODELS, config, get_cwd, i18n
from webui.downloader import OUTPUT_DIR
st.set_page_config(layout="centered",menu_items=MENU_ITEMS)

from webui.components import initial_voice_conversion_params, voice_conversion_form



from webui.utils import ObjectNamespace
from tts_cli import generate_speech
from vc_infer_pipeline import get_vc, vc_single
from webui.contexts import SessionStateContext
from webui.audio import save_input_audio

from webui.utils import gc_collect, get_index, get_optimal_torch_device, get_rvc_models

CWD = get_cwd()

def load_model(_state):
    if _state.rvc_models is None: _state.rvc_models = get_vc(_state.model_name,config=config,device=_state.device)        
    return _state.rvc_models

def convert_vocals(_state,input_audio,**kwargs):
    print(f"converting vocals... {_state.model_name} - {kwargs}")
    models=load_model(_state)
    _state.tts_options = ObjectNamespace(**kwargs)
    return vc_single(input_audio=input_audio,**models,**kwargs)

def init_inference_state():
    return ObjectNamespace(
        models=get_rvc_models(),
        device=get_optimal_torch_device(),
        tts_options=initial_voice_conversion_params(),
    )

def refresh_data(state):
    state.models = get_rvc_models()
    gc_collect()
    return state
    
def clear_data(state):
    del state.vc, state.cpt, state.net_g, state.hubert_model
    gc_collect()
    return state

def get_filename(audio_name,model_name):
    song = os.path.basename(audio_name).split(".")[0]
    singer = os.path.basename(model_name).split(".")[0]
    return f"{singer}.{song}"
    
def one_click_speech(state):
    state.tts_audio = generate_speech(state.tts_text,speaker=os.path.basename(state.model_name).split(".")[0],method=state.tts_method, device=state.device)
    state.converted_voice = convert_vocals(state,state.tts_audio,**(state.tts_options))

if __name__=="__main__":
    with SessionStateContext("tts",initial_state=init_inference_state()) as state:
        with st.container():
            left, right = st.columns(2)
            state.tts_method = left.selectbox(
                i18n("tts.model.selectbox"),
                options=TTS_MODELS,
                index=get_index(TTS_MODELS,state.tts_method),
                format_func=lambda option: option.upper()
                )
            col1, col2 = left.columns(2)
            if col1.button(i18n("inference.refresh_data.button"),use_container_width=True):
                state = refresh_data(state)
                st.experimental_rerun()
            
            state.model_name = right.selectbox(
                i18n("inference.voice.selectbox"),
                options=state.models,
                index=get_index(state.models,state.model_name),
                format_func=lambda option: os.path.basename(option).split(".")[0]
                )
            if right.button(i18n("inference.clear_data.button")):
                state = clear_data(state)
                st.experimental_rerun()
            

        st.subheader(i18n("tts.inference"))
        with st.expander(i18n("tts.options")):
            with st.form("tts.options.form"):
                col1, col2 = st.columns(2)
                device = col1.radio(
                    i18n("inference.device"),
                    disabled=not config.has_gpu,
                    options=DEVICE_OPTIONS,horizontal=True,
                    index=get_index(DEVICE_OPTIONS,state.device))
                
                tts_options = voice_conversion_form(state.tts_options)

                if st.form_submit_button(i18n("inference.save.button")):
                    state.tts_options = tts_options
                    state.device=device
                    st.experimental_rerun()

        with st.container():
            state.tts_text = st.text_area("Speech",value=state.tts_text if state.tts_text else "",max_chars=600)

            if st.button("One Click Convert", disabled=not state.tts_text):
                one_click_speech(state)

            col1, col2 = st.columns(2)
            
            if col1.button("Generate Speech", disabled=not state.tts_text):
                state.tts_audio = generate_speech(state.tts_text,speaker=os.path.basename(state.model_name).split(".")[0],method=state.tts_method, device=state.device)
            if state.tts_audio:
                col1.audio(state.tts_audio[0],sample_rate=state.tts_audio[1])

                if col2.button("Convert Speech"):
                    state.converted_voice = convert_vocals(state,state.tts_audio,**(state.tts_options))

                if state.converted_voice:
                    col2.audio(state.converted_voice[0],sample_rate=state.converted_voice[1])
                    if col2.button("Save Converted Speech"):
                        name = os.path.basename(state.model_name).split(".")[0]
                        save_input_audio(
                            os.path.join(OUTPUT_DIR,"tts",name,
                                         f"{hashlib.md5(state.tts_text.encode('utf-8')).hexdigest()}.wav"),
                                         state.converted_voice)