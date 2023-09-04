# First
from io import BytesIO
import os
from pathlib import Path
import platform
import sys
from pytube import YouTube
import streamlit as st
st.set_page_config("RVC Studio",layout="centered")

from lib.downloader import BASE_MODELS, MDX_MODELS, PRETRAINED_MODELS, RVC_DOWNLOAD_LINK, RVC_MODELS, VITS_MODELS, VR_MODELS, download_link_generator, download_file

CWD = os.getcwd()
if CWD not in sys.path:
    sys.path.append(CWD)

from web_utils.contexts import ProgressBarContext, SessionStateContext

@st.cache_data(show_spinner=False)
def download_audio_to_buffer(url):
    buffer = BytesIO()
    youtube_video = YouTube(url)
    audio = youtube_video.streams.get_audio_only()
    default_filename = audio.default_filename
    audio.stream_to_buffer(buffer)
    return default_filename, buffer

def render_download_ffmpeg(lib_name="ffmpeg.exe"):
    col1, col2 = st.columns(2)
    is_downloaded = os.path.exists(lib_name)
    col1.checkbox(os.path.basename(lib_name),value=is_downloaded,disabled=True)
    if col2.button("Download",disabled=is_downloaded,key=lib_name):
        link = f"{RVC_DOWNLOAD_LINK}ffmpeg.exe"
        with st.spinner(f"Downloading from {link} to {lib_name}"):
            download_file((lib_name,link))
            st.experimental_rerun()

def render_model_checkboxes(generator):
    not_downloaded = []
    for model_path,link in generator:
        col1, col2 = st.columns(2)
        is_downloaded = os.path.exists(model_path)
        col1.checkbox(os.path.basename(model_path),value=is_downloaded,disabled=True)
        if not is_downloaded: not_downloaded.append((model_path,link))
        if col2.button("Download",disabled=is_downloaded,key=model_path):
            with st.spinner(f"Downloading from {link} to {model_path}"):
                download_file((model_path,link))
                st.experimental_rerun()
    return not_downloaded

if __name__=="__main__":

    model_tab, audio_tab = st.tabs(["Model Download","Audio Download"])
    with model_tab:
        st.title("Download required models")

        with st.expander("Base Models"):
            generator = download_link_generator(RVC_DOWNLOAD_LINK, BASE_MODELS)
            to_download = render_model_checkboxes(generator)
            if st.button("Download All",key="download-all-base-models",disabled=len(to_download)==0):
                with ProgressBarContext(to_download,download_file,"Downloading models") as pb:
                    pb.run()

        st.subheader("Required Models for training")
        with st.expander("Pretrained Models"):
            generator = download_link_generator(RVC_DOWNLOAD_LINK, PRETRAINED_MODELS)
            to_download = render_model_checkboxes(generator)
            if st.button("Download All",key="download-all-pretrained-models",disabled=len(to_download)==0):
                with ProgressBarContext(to_download,download_file,"Downloading models") as pb:
                    pb.run()
        with st.container():
            if platform.system() == "Windows":
                render_download_ffmpeg()
            elif platform.system() == "Linux":
                st.markdown("run `apt update && apt install -y -qq ffmpeg espeak` in your terminal")

        st.subheader("Required Models for inference")
        with st.expander("RVC Models"):
            generator = download_link_generator(RVC_DOWNLOAD_LINK, RVC_MODELS)
            to_download = render_model_checkboxes(generator)
            if st.button("Download All",key="download-all-rvc-models",disabled=len(to_download)==0):
                with ProgressBarContext(to_download,download_file,"Downloading models") as pb:
                    pb.run()
        with st.expander("Vocal Separation Models"):
            generator = download_link_generator(RVC_DOWNLOAD_LINK, VR_MODELS+MDX_MODELS)
            to_download = render_model_checkboxes(generator)
            if st.button("Download All",key="download-all-vr-models",disabled=len(to_download)==0):
                with ProgressBarContext(to_download,download_file,"Downloading models") as pb:
                    pb.run()
        with st.expander("VITS Models"):
            generator = download_link_generator(RVC_DOWNLOAD_LINK, VITS_MODELS)
            to_download = render_model_checkboxes(generator)
            with ProgressBarContext(to_download,download_file,"Downloading models") as pb:
                st.button("Download All",key="download-all-vits-models",disabled=len(to_download)==0,on_click=pb.run)

    with audio_tab, SessionStateContext("youtube_downloader") as state:
        st.title("Download Audio from Youtube")
        state.url = st.text_input("Insert Youtube URL:",value=state.url if state.url else "")
        if st.button("Fetch",disabled=not state.url):
            with st.spinner("Downloading Audio Stream from Youtube..."):
                state.downloaded_audio = download_audio_to_buffer(state.url)
            st.subheader("Title")
            st.write(state.downloaded_audio[0])
            title_vid = Path(state.downloaded_audio[0]).with_suffix(".mp3").name
            st.subheader("Listen to Audio")
            st.audio(state.downloaded_audio[1], format='audio/mpeg')
            st.subheader("Download Audio File")
            st.download_button(
                label="Download mp3",
                data=state.downloaded_audio[1],
                file_name=title_vid,
                mime="audio/mpeg")