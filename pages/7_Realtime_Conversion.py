from collections import deque
import logging
import logging.handlers
import threading
import time
import os
from typing import List

# import av
import numpy as np
import pydub
import streamlit as st
# from twilio.rest import Client

from streamlit_webrtc import WebRtcMode, webrtc_streamer
from vc_infer_pipeline import get_vc, vc_single

from webui import DEVICE_OPTIONS, get_cwd, i18n, config
from webui.components import initial_voice_conversion_params, save_voice_conversion_params, voice_conversion_form
from webui.contexts import SessionStateContext
from webui.utils import ObjectNamespace, gc_collect, get_filenames, get_index, get_optimal_torch_device
import pyaudio
import webrtcvad
import noisereduce
from av.audio import AudioFrame

CWD = get_cwd()

logger = logging.getLogger(__name__)

def render_rvc_options_form(state):
    state.voice_model = st.selectbox(
            i18n("inference.voice.selectbox"),
            options=state.voice_model_list,
            index=get_index(state.voice_model_list,state.voice_model),
            format_func=lambda option: os.path.basename(option).split(".")[0]
            )
    state.rvc_options = voice_conversion_form(state.rvc_options)
    return state

def load_model(_state):
    if _state.rvc_models is None or _state.rvc_models["model_name"]!=_state.voice_model:
        del _state.rvc_models
        _state.rvc_models = get_vc(_state.voice_model,config=config,device=_state.device)
        gc_collect()
    return _state.rvc_models

def render_recorder_app(state):
    
    frames_deque_lock = threading.Lock()
    frames_deque = deque([])

    def is_speech(frame):
        try:
            audio = frame.to_ndarray() #noisereduce.reduce_noise(y=frame.to_ndarray(),sr=frame.sample_rate)
            length = int(0.03 * frame.sample_rate) # 30ms length
            # state.vad.set_mode(3)
            if state.vad.is_speech(audio.tobytes(), frame.sample_rate, length=length):
                return frame
        except Exception as e:
            print(e)
            return False

    def process_rvc_frames(audio_frames):
        sound_chunk = pydub.AudioSegment.empty()

        for audio_frame in audio_frames:
            if is_speech(audio_frame):
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

        if len(sound_chunk) > 0:
            sound_chunk = sound_chunk.set_channels(1).set_frame_rate(16000)
            sound_chunk = sound_chunk.get_array_of_samples()
            
            audio = noisereduce.reduce_noise(y=sound_chunk, sr=16000)
            
            changed_voice = vc_single(
                input_audio=(audio,16000),
                **state.rvc_options,
                **state.rvc_models
            )
            return changed_voice

    async def queued_rvc_callback(frames: List[AudioFrame]):
        
        new_frames = []

        with frames_deque_lock:
            if len(frames) == 0:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
            else:
                rvc_frame = process_rvc_frames(frames)
                if rvc_frame:
                    frames_deque.append(rvc_frame)
            

        # Return empty frames to be silent.
            
            for frame in frames:
                input_array = frame.to_ndarray()
                new_frame = AudioFrame.from_ndarray(
                    np.zeros(input_array.shape, dtype=input_array.dtype),
                    layout=frame.layout.name,
                )
                new_frame.sample_rate = frame.sample_rate
                new_frames.append(new_frame)

        return new_frames
    
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDRECV,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
        # rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        # audio_frame_callback=rvc_callback,
        async_processing=True,
        queued_audio_frames_callback=queued_rvc_callback
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    stream = None
    while True:
        if stream is None:
            stream = state.pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=state.sample_rate,
                output=True,
                output_device_index=state.output_device_index
            )
            stream.start_stream()
            status_indicator.write("Stream started")

        if webrtc_ctx.state.playing:
            status_indicator.write("Running. Say something!")

            with frames_deque_lock:
                while len(frames_deque) > 0:
                    changed_voice = frames_deque.popleft()
                    audio, sr = changed_voice
                    CHUNKSIZE = 1024
                    dtype = "int16" if np.abs(changed_voice[0]).max()>1 else "float32" 
                    
                    for i in range(0,len(audio),CHUNKSIZE):
                        if webrtc_ctx.state.playing:
                            data = audio[i:i+CHUNKSIZE].astype(dtype)
                            stream.write(data.tostring())
        else:
            status_indicator.write("Stopped.")
            stream.stop_stream()
            stream.close()
            print("stream closed")
            break
    
        

def get_voice_list():
    models_list = [os.path.relpath(path,CWD) for path in get_filenames(root=os.path.join(CWD,"models"),folder="RVC",exts=["pth"])]
    return models_list

@st.cache_data
def get_output_sound_devices(_state):
    devices = [
        i for i in range(_state.pyaudio.get_device_count())
        if _state.pyaudio.get_device_info_by_index(i)["maxOutputChannels"]>0
        and _state.pyaudio.get_device_info_by_index(i)["maxInputChannels"]==0
    ]
    return devices

def init_state():
    return ObjectNamespace(
        rvc_options = initial_voice_conversion_params("realtime-rvc"),
        voice_model = "",
        voice_model_list = get_voice_list(),
        device=get_optimal_torch_device(),
        rvc_models=None,
        sample_rate=16000,
        input_device=None,
        output_device_index=None,
        pyaudio = pyaudio.PyAudio(),
        vad = webrtcvad.Vad(2)
    )
if __name__ == "__main__":
    with SessionStateContext("realtime-rvc",initial_state=init_state()) as state:
        st.header("Real Time RVC")
        col1, col2 = st.columns(2)
        state.device = col1.radio(
            i18n("inference.device"),
            disabled=not config.has_gpu,
            options=DEVICE_OPTIONS,horizontal=True,
            index=get_index(DEVICE_OPTIONS,state.device))
        OUTPUT_DEVICES = get_output_sound_devices(state)
        state.output_device_index = col2.selectbox("Output Device",
            options=OUTPUT_DEVICES,
            format_func=lambda i: f"{i}. "+state.pyaudio.get_device_info_by_index(i)["name"],
            index=get_index(OUTPUT_DEVICES,state.pyaudio.get_default_output_device_info()["index"])
        )
        with st.expander(f"Voice Model: {state.rvc_models['model_name'] if state.rvc_models else None}", expanded=state.rvc_models is None):
            with st.form("realtime-voice"):
                state = render_rvc_options_form(state)
                if st.form_submit_button("Save",use_container_width=True,type="primary"):
                    del state.rvc_models
                    gc_collect()
                    state.rvc_models = load_model(state)
                    state.sample_rate = state.rvc_models["cpt"]["config"][-1]
                    save_voice_conversion_params("realtime-rvc",state.rvc_options)

        
        if state.rvc_models: render_recorder_app(state)
