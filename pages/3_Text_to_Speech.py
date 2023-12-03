import hashlib
import os
import streamlit as st

from webui import DEVICE_OPTIONS, MENU_ITEMS, TTS_MODELS
from lib import config, ObjectNamespace, i18n, OUTPUT_DIR
from webui.api import convert_vocals, get_rvc_models
st.set_page_config(layout="centered",menu_items=MENU_ITEMS)

from webui.components import initial_voice_conversion_params, save_voice_conversion_params, voice_conversion_form
from tts_cli import generate_speech
from webui.contexts import SessionStateContext
from lib.audio import save_input_audio

from lib.utils import gc_collect, get_index, get_optimal_torch_device

def init_inference_state():
    return ObjectNamespace(
        models=get_rvc_models(),
        device=get_optimal_torch_device(),
        tts_options=initial_voice_conversion_params("tts"),
    )

def call_rvc(model_name,input_audio,**kwargs):
    with st.status(f"converting vocals... {model_name} - {kwargs}") as status:
        try:
            return convert_vocals(model_name,input_audio,**kwargs)
        except Exception as e:
            print(e)
            status.error(e)
            status.update(state="error")
            return None

def refresh_data(state):
    state.models = get_rvc_models()
    gc_collect()
    return state
    
def clear_data(state):
    gc_collect()
    return state

def get_filename(audio_name,model_name):
    song = os.path.basename(audio_name).split(".")[0]
    singer = os.path.basename(model_name).split(".")[0]
    return f"{singer}.{song}"
    
def one_click_speech(state):
    state.tts_audio = generate_speech(state.tts_text,speaker=os.path.basename(state.model_name).split(".")[0],method=state.tts_method, device=state.device)
    state.converted_voice = call_rvc(state.model_name,state.tts_audio,**(state.tts_options))

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
            

        st.subheader(i18n("tts.inference"))
        with st.expander(i18n("tts.options")):
            with st.form("tts.options.form"):
                col1, col2 = st.columns(2)
                device = col1.radio(
                    i18n("inference.device"),
                    disabled=not config.has_gpu,
                    options=DEVICE_OPTIONS,horizontal=True,
                    index=get_index(DEVICE_OPTIONS,state.device))
                
                state.tts_options = voice_conversion_form(state.tts_options)

                if st.form_submit_button(i18n("inference.save.button")):
                    
                    state.device = device
                    save_voice_conversion_params("tts",state.tts_options)

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
                    state.converted_voice = convert_vocals(state.model_name,state.tts_audio,**(state.tts_options))

                if state.converted_voice:
                    col2.audio(state.converted_voice[0],sample_rate=state.converted_voice[1])
                    if col2.button("Save Converted Speech"):
                        name = os.path.basename(state.model_name).split(".")[0]
                        save_input_audio(
                            os.path.join(OUTPUT_DIR,"tts",name,
                                         f"{hashlib.md5(state.tts_text.encode('utf-8')).hexdigest()}.wav"),
                                         state.converted_voice)