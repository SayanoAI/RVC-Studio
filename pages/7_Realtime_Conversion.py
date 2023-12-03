import os

# import av
import streamlit as st
# from twilio.rest import Client

from vc_infer_pipeline import get_vc

from webui import DEVICE_OPTIONS
from lib import i18n, config
from webui.api import get_rvc_models
from webui.components import initial_voice_conversion_params, save_voice_conversion_params, voice_conversion_form
from webui.contexts import SessionStateContext
from webui.recorder import RecorderPlayback
from lib.utils import ObjectNamespace, gc_collect, get_index, get_optimal_torch_device
from pyaudio import PyAudio

def render_rvc_options_form(state):
    state.voice_model = st.selectbox(
            i18n("inference.voice.selectbox"),
            options=state.voice_model_list,
            index=get_index(state.voice_model_list,state.voice_model),
            format_func=lambda option: os.path.basename(option).split(".")[0]
            )
    state.rvc_options = voice_conversion_form(state.rvc_options,use_hybrid=False)
    return state

def load_model(_state):
    if _state.rvc_models is None or _state.rvc_models["model_name"]!=_state.voice_model:
        del _state.rvc_models
        _state.rvc_models = get_vc(_state.voice_model,config=config,device=_state.device)
        gc_collect()
    return _state.rvc_models

def render_recorder_settings(state):
    if state.recorder is not None and state.recorder.recording:
        if st.button("Stop",type="primary"):
            state.recorder.stop()
            del state.recorder
            state.recorder = None
            gc_collect()
            st.experimental_rerun()
        state.recorder.silence_threshold = st.slider(
            "Silence Threshold",
            min_value=0.,max_value=0.5, step=0.01,
            value=state.recorder.silence_threshold
        )

@st.cache_data
def get_sound_devices(device_type: str):
    p = PyAudio()
    devices = [
        i for i in range(p.get_device_count())
        if p.get_device_info_by_index(i)[device_type]>0
    ]
    p.terminate()
    return devices

def init_state():
    return ObjectNamespace(
        rvc_options = initial_voice_conversion_params("realtime-rvc"),
        voice_model = "",
        voice_model_list = get_rvc_models(),
        device=get_optimal_torch_device(),
        sample_rate=16000,
        input_device_index=None,
        output_device_index=None,
        recorder = None,
        p = PyAudio()
    )
if __name__ == "__main__":
    with SessionStateContext("realtime-rvc",initial_state=init_state()) as state:
        st.header("Real Time RVC")
        col1, col2, col3 = st.columns([1,2,2])
        state.device = col1.radio(
            i18n("inference.device"),
            disabled=not config.has_gpu,
            options=DEVICE_OPTIONS,horizontal=True,
            index=get_index(DEVICE_OPTIONS,state.device))
        INPUT_DEVICES = get_sound_devices("maxInputChannels")
        state.input_device_index = col2.selectbox("Input Device",
            options=INPUT_DEVICES,
            format_func=lambda i: f"{i}. "+state.p.get_device_info_by_index(i)["name"],
            index=get_index(INPUT_DEVICES,state.input_device_index)
        )
        OUTPUT_DEVICES = get_sound_devices("maxOutputChannels")
        state.output_device_index = col3.selectbox("Output Device",
            options=OUTPUT_DEVICES,
            format_func=lambda i: f"{i}. "+state.p.get_device_info_by_index(i)["name"],
            index=get_index(OUTPUT_DEVICES,state.output_device_index)
        )
        with st.expander(f"Voice Model: {state.recorder}", expanded=state.recorder is None):
            with st.form("realtime-voice"):
                state = render_rvc_options_form(state)
                if st.form_submit_button("Start",use_container_width=True,type="primary"):
                    if state.recorder is None:
                        state.recorder = RecorderPlayback()
                        state.recorder.start(state.voice_model,config=config,device=state.device,
                                            input_device_index=state.input_device_index,
                                            output_device_index=state.output_device_index,
                                            **state.rvc_options)
                    elif state.voice_model==state.recorder.voice_model:
                        state.recorder.update_options(state.rvc_options)
                    else: state.recorder.load_rvc_model(state.voice_model, config, state.device)
                    save_voice_conversion_params("realtime-rvc",state.rvc_options)

        
        render_recorder_settings(state)
