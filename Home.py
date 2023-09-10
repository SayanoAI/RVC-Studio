# First
from io import BytesIO
import os
from pathlib import Path
import platform
import sys
from pytube import YouTube
import streamlit as st
from web_utils import MENU_ITEMS
st.set_page_config("RVC Studio",layout="centered",menu_items=MENU_ITEMS)

from tts_cli import TTS_MODELS_DIR, stt_checkpoint, load_stt_models

from web_utils.audio import SUPPORTED_AUDIO
from web_utils.components import file_uploader_form


from web_utils.downloader import BASE_MODELS, BASE_MODELS_DIR, MDX_MODELS, PRETRAINED_MODELS, RVC_DOWNLOAD_LINK, RVC_MODELS, SONG_DIR, VITS_MODELS, VR_MODELS, download_link_generator, download_file, save_file, save_file_generator

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

def rvc_index_path_mapper(params):
    (data_path, data) = params
    if "index" not in data_path.split(".")[-1]:
        return params
    else: return (os.path.join(BASE_MODELS_DIR,"RVC",".index",os.path.basename(data_path)), data) # index file

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
            file_uploader_form(
                os.path.join(BASE_MODELS_DIR,"RVC"),"Upload your RVC model",
                types=["pth","index","zip"],
                accept_multiple_files=True,
                params_mapper=rvc_index_path_mapper)
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

        with st.expander("STT Models"):
            col1, col2 = st.columns(2)
            stt_path = os.path.join(TTS_MODELS_DIR,stt_checkpoint)
            is_downloaded = os.path.exists(stt_path)
            col1.checkbox(os.path.basename(stt_path),value=is_downloaded,disabled=True)
            if col2.button("Download",disabled=is_downloaded,key=stt_path):
                with st.spinner(f"Downloading {stt_checkpoint} to {stt_path}"):
                    models = load_stt_models() #hacks the from_pretrained downloader
                    del models
                    st.experimental_rerun()
            pass

        with st.expander("LLM Models"):
            pass

    with audio_tab, SessionStateContext("youtube_downloader") as state:
        
        st.title("Download Audio from Youtube")

        state.url = st.text_input("Insert Youtube URL:",value=state.url if state.url else "")
        if st.button("Fetch",disabled=not state.url):
            with st.spinner("Downloading Audio Stream from Youtube..."):
                state.downloaded_audio = download_audio_to_buffer(state.url)

        if state.downloaded_audio:
            title, data = state.downloaded_audio
            st.subheader("Title")
            st.write(title)
            fname = Path(title).with_suffix(".mp3").name
            st.subheader("Listen to Audio")
            st.audio(data, format='audio/mpeg')
            st.subheader("Download Audio File")
            if st.button("Download Song"):
                params = (os.path.join(SONG_DIR,fname),data.read())
                save_file(params)

            # st.download_button(
            #     label="Download mp3",
            #     data=state.downloaded_audio[1],
            #     file_name=title_vid,
            #     mime="audio/mpeg")