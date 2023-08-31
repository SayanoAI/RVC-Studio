import streamlit as st
from types import SimpleNamespace

from webui_utils import get_filenames, get_index, load_config, load_input_audio

_, i18n = load_config()

@st.cache_data
def init_processing_state():
    state = SimpleNamespace(**{
        "f1": None,
        "f2": None,
        "s1": None,
        "s2": None,
        "s3": None,
        "f3": None,
        "a1": None,
        "a2": None,
        "a3": None
    })
    return state

DEVICE_OPTIONS = ["cpu","cuda"]
PITCH_EXTRACTION_OPTIONS = ["harvest","crepe","rmvpe"]
INSTRUMENTAL_FILE_OPTIONS = get_filenames(exts=["wav","flac","ogg","mp3"])
VOCAL_FILE_OPTIONS = get_filenames(exts=["wav","flac","ogg","mp3"])

def render(state):
    col1,col2=st.columns(2)
    
    f1=col1.selectbox("Instrumental",options=INSTRUMENTAL_FILE_OPTIONS,
                      index=get_index(INSTRUMENTAL_FILE_OPTIONS,st.session_state.processing.f1))
    st.session_state.processing.f1=f1
    f2=col2.selectbox("Vocal",options=VOCAL_FILE_OPTIONS,
                      index=get_index(VOCAL_FILE_OPTIONS,st.session_state.processing.f2))
    st.session_state.processing.f2=f2
    
    if st.button("Merge",disabled=not (st.session_state.processing.f1 and st.session_state.processing.f2)):
        s1 = load_input_audio(f1)
        st.session_state.processing.s1=s1
        st.session_state.processing.a1=s1.export().read()

        s2 = load_input_audio(f2)
        st.session_state.processing.s2=s2
        st.session_state.processing.a2=s2.export().read()

        s3=s1.overlay(s2)
        st.session_state.processing.s3=s3
        st.session_state.processing.a3=s3.export().read()

    if st.session_state.processing.a1: col1.audio(st.session_state.processing.a1)
    if st.session_state.processing.a2: col2.audio(st.session_state.processing.a2)
    if st.session_state.processing.a3: st.audio(st.session_state.processing.a3)

    

    return state

def init_state():
    st.session_state["processing"] = st.session_state.get("processing",init_processing_state())

init_state()

if __name__=="__main__": st.session_state.processing=render(st.session_state.processing)