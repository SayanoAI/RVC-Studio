# First
from io import BytesIO
from pathlib import Path
from pytube import YouTube
import streamlit as st

st.set_page_config("RVC Studio",menu_items={
    # 'Get Help': '',
    # 'Report a bug': "",
    # 'About': ""
})

from webui_utils import SessionStateContext

@st.cache_data(show_spinner=False)
def download_audio_to_buffer(url):
    buffer = BytesIO()
    youtube_video = YouTube(url)
    audio = youtube_video.streams.get_audio_only()
    default_filename = audio.default_filename
    audio.stream_to_buffer(buffer)
    return default_filename, buffer

if __name__=="__main__":
    with SessionStateContext("youtube_downloader") as state:
    # def render():
        st.title("Download Audio from Youtube")
        state.url = st.text_input("Insert Youtube URL:",value=state.url)
        if st.button("Fetch",disabled=state.url is None):
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

# if __name__=="__main__":
    # render()