import os
import streamlit as st

st.set_page_config(layout="centered")

from types import SimpleNamespace
from tts_cli import generate_speech, train_speaker_embedding
from vc_infer_pipeline import get_vc, vc_single
from web_utils.contexts import SessionStateContext
from web_utils.audio import save_input_audio

from webui_utils import gc_collect, get_filenames, get_index, config, i18n

@st.cache_resource(show_spinner=False)
def load_model(_state,model_name):

    index_file = get_filenames(root="./models/RVC",folder=".index",exts=["index"],name_filters=[os.path.basename(model_name).split(".")[0]])
    file_index = index_file[0] if len(index_file) else ""
    if _state.file_index==file_index and _state.vc and _state.cpt and _state.net_g and _state.hubert_model:
        return {
            "vc": _state.vc,
            "cpt": _state.cpt,
            "net_g": _state.net_g,
            "file_index": index_file[0] if len(index_file) else "",
            "hubert_model": _state.hubert_model
        }
    else:
        _state = clear_data(_state)
        data = get_vc(model_name,config=config,device=_state.device)
        _state.vc = data["vc"]
        _state.cpt = data["cpt"]
        _state.net_g = data["net_g"]
        _state.hubert_model = data["hubert_model"]
        data["file_index"] = index_file[0] if len(index_file) else ""
        return data

def convert_vocals(_state,input_audio,**kwargs):
    print(f"converting vocals... {_state.model_name} - {kwargs}")
    models=load_model(_state,_state.model_name)
    _state.tts_options = SimpleNamespace(**kwargs)
    return vc_single(input_audio=input_audio,**models,**kwargs)

def get_models(folder="."):
    fnames = get_filenames(root="./models",folder=folder,exts=["pth","pt"])
    return fnames

@st.cache_data
def init_inference_state():
    state = SimpleNamespace(
        models=get_models(folder="RVC"),
        device="cuda" if config.has_gpu else "cpu",
        tts_options=SimpleNamespace(
            f0_up_key=6,
            f0_method="rmvpe",
            index_rate=.8,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=.25,
            protect=0.25
        )
    )
    return vars(state)

def refresh_data(state):
    state.models = get_models(folder="RVC")
    gc_collect()
    return state
    
def clear_data(state):
    del state.vc, state.cpt, state.net_g, state.hubert_model
    gc_collect()
    return state

DEVICE_OPTIONS = ["cpu","cuda"]
PITCH_EXTRACTION_OPTIONS = ["crepe","rmvpe"]

@st.cache_data
def get_filename(audio_name,model_name):
    song = os.path.basename(audio_name).split(".")[0]
    singer = os.path.basename(model_name).split(".")[0]
    return f"{singer}.{song}"

def download_song(output_audio,output_audio_name,ext="mp3"):
    output_dir = os.sep.join([os.getcwd(),"output"])
    os.makedirs(output_dir,exist_ok=True)
    output_file = os.sep.join([output_dir,f"{output_audio_name}.{ext}"])
    return f"saved to {output_file}.{ext}: {save_input_audio(output_file,output_audio,to_int16=True)}"
    
def one_click_speech(state):
    speaker = train_speaker_embedding(os.path.basename(state.model_name).split(".")[0])
    state.tts_audio = generate_speech(state.tts_text,speaker=speaker,method=state.tts_method, device=state.device)
    state.converted_voice = convert_vocals(state,state.tts_audio,**vars(state.tts_options))

TTS_MODELS = ["speecht5","bark","tacotron2","edge","vits"]

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
                
                f0_up_key = st.slider(i18n("inference.f0_up_key"),min_value=-12,max_value=12,step=1,value=state.tts_options.f0_up_key)
                f0_method = st.selectbox(i18n("inference.f0_method"),
                                                    options=PITCH_EXTRACTION_OPTIONS,
                                                    index=get_index(PITCH_EXTRACTION_OPTIONS,state.tts_options.f0_method))
                resample_sr = st.select_slider(i18n("inference.resample_sr"),
                                                    options=[0,16000,24000,22050,40000,44100,48000],
                                                    value=state.tts_options.resample_sr)
                index_rate=st.slider(i18n("inference.index_rate"),min_value=0.,max_value=1.,step=.05,value=state.tts_options.index_rate)
                filter_radius=st.slider(i18n("inference.filter_radius"),min_value=0,max_value=7,step=1,value=state.tts_options.filter_radius)
                rms_mix_rate=st.slider(i18n("inference.rms_mix_rate"),min_value=0.,max_value=1.,step=.05,value=state.tts_options.rms_mix_rate)
                protect=st.slider(i18n("inference.protect"),min_value=0.,max_value=.5,step=.01,value=state.tts_options.protect)
                
                if st.form_submit_button(i18n("inference.save.button")):
                    state.tts_options = SimpleNamespace(
                        f0_up_key=f0_up_key,
                        f0_method=f0_method,
                        resample_sr=resample_sr,
                        index_rate=index_rate,
                        filter_radius=filter_radius,
                        rms_mix_rate=rms_mix_rate,
                        protect=protect
                    )
                    state.device=device

        with st.container():
            state.tts_text = st.text_area("Speech",state.tts_text,max_chars=600)

            if st.button("One Click Convert"):
                one_click_speech(state)

            col1, col2 = st.columns(2)
            
            if col1.button("Generate Speech"):
                with st.spinner("performing TTS speaker embedding..."):
                    speaker = train_speaker_embedding(os.path.basename(state.model_name).split(".")[0])
                with st.spinner("performing TTS speaker inference..."):
                    state.tts_audio = generate_speech(state.tts_text,speaker=speaker,method=state.tts_method, device=state.device)
            if state.tts_audio:
                col1.audio(state.tts_audio[0],sample_rate=state.tts_audio[1])

                if col2.button("Convert Speech"):
                    state.converted_voice = convert_vocals(state,state.tts_audio,**vars(state.tts_options))

                if state.converted_voice:
                    col2.audio(state.converted_voice[0],sample_rate=state.converted_voice[1])
                    if col2.button("Save Converted Speech"):
                        download_song(state.converted_voice,get_filename(state.model_name,str(hash(state.tts_text))[-8:]),ext="wav")