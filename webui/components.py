import html
import json
import os
from pathlib import Path
import random

import urllib.request
from webui.api import get_uvr_models, get_uvr_postprocess_models, get_uvr_preprocess_models
from typing import Tuple
import streamlit as st

from webui import PITCH_EXTRACTION_OPTIONS, get_cwd, i18n, ObjectNamespace
from webui.contexts import ProgressBarContext
from webui.downloader import save_file, save_file_generator
from webui.utils import gc_collect, get_index, get_subprocesses

CWD = get_cwd()
    
def __default_mapper(x: Tuple[str,any]):
     return x

def file_uploader_form(save_dir, title="Upload your files", types=None, params_mapper=__default_mapper, **kwargs):
    with st.container():
        with st.form(save_dir+title, clear_on_submit=True):
            files = st.file_uploader(title,type=types,**kwargs)
            if st.form_submit_button("Save") and files is not None:
                file_list = [params_mapper(params)
                                for params in save_file_generator(save_dir,files if isinstance(files, list) else [files])]
                with ProgressBarContext(file_list,save_file,"saving files...") as pb:
                    pb.run()
                    del file_list
                    st.toast("Successfully saved files!")

def active_subprocess_list():
    with st.expander(i18n("process.pids")):
        for p in get_subprocesses():
            col1,col2,col3,col4=st.columns(4)
            try:
                col1.write(p.pid)
                col2.write(p.name)
                col3.write(p.time_started)
                if col4.button(i18n("process.kill_one_pid"),key=f"process.kill_one_pid.{p.pid}"):
                    for c in get_subprocesses(p.pid):
                        c.kill()
                    p.kill()
                    gc_collect()
                    st.experimental_rerun()
            except Exception as e:
                print(e)

def initial_vocal_separation_params(folder=None):
    params = ObjectNamespace(
        preprocess_models=[],
        postprocess_models=[],
        agg=10,
        merge_type="median",
        uvr_models=[],
        use_cache=True,
    )

    if folder:
        try:
            config_file = os.path.join(os.getcwd(),"configs",folder,"vocal_separation_params.json")
            os.makedirs(os.path.dirname(config_file),exist_ok=True)
            if os.path.isfile(config_file):
                with open(config_file,"r") as f:
                    data = json.load(f)
                    params.update(data)
        except Exception as e:
            print(e)
    return params

def save_vocal_separation_params(folder,data):
    config_file = os.path.join(os.getcwd(),"configs",folder,"vocal_separation_params.json")
    os.makedirs(os.path.dirname(config_file),exist_ok=True)
    with open(config_file,"w") as f:
        return f.write(json.dumps(data,indent=2))
        
def vocal_separation_form(state):
    uvr5_models=get_uvr_models()
    uvr5_preprocess_models=get_uvr_preprocess_models()
    uvr5_postprocess_models=get_uvr_postprocess_models()
    
    state.preprocess_models = st.multiselect(
            i18n("inference.preprocess_model"),
            options=uvr5_preprocess_models,
            format_func=lambda item: os.path.basename(item),
            default=[name for name in state.preprocess_models if name in uvr5_preprocess_models])
    state.uvr_models = st.multiselect(
        i18n("inference.model_paths"),
        options=uvr5_models,
        format_func=lambda item: os.path.basename(item),
        default=[name for name in state.uvr_models if name in uvr5_models])
    state.postprocess_models = st.multiselect(
            i18n("inference.postprocess_model"),
            options=uvr5_postprocess_models,
            format_func=lambda item: os.path.basename(item),
            default=[name for name in state.postprocess_models if name in uvr5_postprocess_models])
    col1, col2, col3 = st.columns(3)
    
    state.merge_type = col2.radio(
        i18n("inference.merge_type"),
        options=["median","mean"],horizontal=True,
        index=get_index(["median","mean"],state.merge_type))
    state.agg = col1.slider(i18n("inference.agg"),min_value=0,max_value=20,step=1,value=state.agg)
    state.use_cache = col3.checkbox(i18n("inference.use_cache"),value=state.use_cache)
    return state

def initial_voice_conversion_params(folder=None):
    params = ObjectNamespace(
        f0_up_key=0,
        f0_method=["rmvpe"],
        f0_autotune=False,
        merge_type="median",
        index_rate=.75,
        # filter_radius=3,
        resample_sr=0,
        rms_mix_rate=.2,
        protect=0.2,
        )
    if folder:
        try:
            config_file = os.path.join(os.getcwd(),"configs",folder,"voice_conversion_params.json")
            os.makedirs(os.path.dirname(config_file),exist_ok=True)
            if os.path.isfile(config_file):
                with open(config_file,"r") as f:
                    data = json.load(f)
                    params.update(data)
        except Exception as e:
            print(e)
    return params

def save_voice_conversion_params(folder,data):
    config_file = os.path.join(os.getcwd(),"configs",folder,"voice_conversion_params.json")
    os.makedirs(os.path.dirname(config_file),exist_ok=True)
    with open(config_file,"w") as f:
        return f.write(json.dumps(data,indent=2))
    
def voice_conversion_form(state, use_hybrid=True):
    state.f0_up_key = st.slider(i18n("inference.f0_up_key"),min_value=-12,max_value=12,value=state.f0_up_key,step=1)
    if use_hybrid:
        state.f0_method = st.multiselect(i18n("inference.f0_method"),
                                            options=PITCH_EXTRACTION_OPTIONS,
                                            default=state.f0_method)
    else: state.f0_method = st.selectbox(i18n("inference.f0_method"),
                                        options=PITCH_EXTRACTION_OPTIONS,
                                        index=get_index(PITCH_EXTRACTION_OPTIONS,state.f0_method))
    col1, col2 = st.columns(2)
    state.merge_type = col1.radio(
        i18n("inference.merge_type"),
        options=["median","mean"],horizontal=True,
        index=get_index(["median","mean"],state.merge_type))
    state.f0_autotune = col2.checkbox(i18n("inference.f0_autotune"),value=state.f0_autotune)
    state.resample_sr = st.select_slider(i18n("inference.resample_sr"),
                                        options=[0,16000,24000,22050,40000,44100,48000],
                                        value=state.resample_sr)
    state.index_rate=st.slider(i18n("inference.index_rate"),min_value=0.,max_value=1.,step=.05,value=state.index_rate)
    # state.filter_radius=st.slider(i18n("inference.filter_radius"),min_value=0,max_value=7,step=1,value=state.filter_radius)
    state.rms_mix_rate=st.slider(i18n("inference.rms_mix_rate"),min_value=0.,max_value=1.,step=.05,value=state.rms_mix_rate)
    state.protect=st.slider(i18n("inference.protect"),min_value=0.,max_value=.5,step=.01,value=state.protect)
    return state

# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def file_downloader(params: Tuple[str, str], expected_size=None):
    path, url = params
    download_to = Path(path)
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size >= expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

## Ported from streamlit_tensorboard with modifications
def st_iframe(url: str, width=None, height=None, scrolling=False):
    """Embed iframe within a Streamlit app
    Parameters
    ----------
    url: string
        URL of the server. Defaults to `http://localhost:8188`
    width: int
        The width of the frame in CSS pixels. Defaults to container width.
    height: int
        The height of the frame in CSS pixels. Defaults to container height.
    scrolling: bool
        If True, show a scrollbar when the content is larger than the iframe.
        Otherwise, do not show a scrollbar. Defaults to False.

    Example
    -------
    >>> st_iframe(url="http://localhost:8888", width=1080)
    """

    frame_id = "swagger-frame-{:08x}".format(random.getrandbits(64))
    shell = """
        <iframe id="%HTML_ID%" width="%WIDTH%" height="%HEIGHT%" frameborder="0">
        </iframe>
        <script>
        (function() {
            const frame = document.getElementById(%JSON_ID%);
            frame.src = new URL(%URL%, window.location);
        })();
        </script>
    """

    replacements = [
        ("%HTML_ID%", html.escape(frame_id, quote=True)),
        ("%JSON_ID%", json.dumps(frame_id)),
        ("%HEIGHT%", str(height) if height else "100%"),
        ("%WIDTH%", str(width) if width else "100%"),
        ("%URL%", json.dumps(url)),
    ]

    for (k, v) in replacements:
        shell = shell.replace(k, v)

    return st.components.v1.html(shell, width=width, height=height, scrolling=scrolling)