import os
import subprocess
from time import sleep
import psutil

import requests
from webui import MENU_ITEMS, SERVERS, ObjectNamespace, get_cwd
import streamlit as st
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)
from webui.chat import load_model_data

from webui.utils import get_filenames, pid_is_active, poll_url
from webui.components import active_subprocess_list, st_iframe
from webui.contexts import ProgressBarContext, SessionStateContext

CWD = get_cwd()

st.write(SERVERS["RVC"])
st.write(pid_is_active(SERVERS["RVC"]["pid"]))

def start_server(host,port):
    if pid_is_active(None if SERVERS["RVC"] is None else SERVERS["RVC"].get("pid")):
        pid = SERVERS["RVC"].get("pid")
        process = psutil.Process(pid)
        if process.is_running(): return SERVERS["RVC"]["url"]
    
    base_url = f"http://{host}:{port}"
    cmd = f"python api.py --port={port} --host={host}"
    p = subprocess.Popen(cmd, cwd=CWD, shell=True)

    if poll_url(base_url):
        SERVERS["RVC"] = {
            "url": base_url,
            "pid": p.pid
        }
    
    return base_url

def get_model_list():
    models_list =  [os.path.relpath(path,CWD) for path in get_filenames(root=os.path.join(CWD,"models"),folder="LLM",exts=["bin","gguf"])]
    return models_list

def render_model_params_form(state):
    CONTEXTSIZE_OPTIONS = [256,512,1024,2048,3072,4096,6144,8192,12288,16384,24576,32768,65536]
    state.n_ctx = st.select_slider("Max Context Length", options=CONTEXTSIZE_OPTIONS, value=state.n_ctx)
    state.n_gpu_layers = st.slider("GPU Layers", min_value=0, max_value=64, step=4, value=state.n_gpu_layers)
    
    return state

def initial_state():
    return ObjectNamespace(
        remote_bind=False,
        host="localhost",
        port=5555
    )

@st.cache_data
def get_params(model):
    data = load_model_data(model)
    return data["params"]

if __name__=="__main__":
    with SessionStateContext("rvc-api",initial_state()) as state:
        is_active = pid_is_active(None if SERVERS["RVC"] is None else SERVERS["RVC"].get("pid"))

        with st.form("rvc-api-form"):
            state.remote_bind = st.checkbox("Bind to 0.0.0.0 (Required for docker or remote connections)", value=state.remote_bind)
            state.host = "0.0.0.0" if state.remote_bind else "localhost"
            state.port = st.number_input("Port", value=state.port or 5555)
            state.url = st.text_input("Server URL", value = f"http://{state.host}:{state.port}")

            if st.form_submit_button("Start Server",disabled=is_active):
                with ProgressBarContext([1]*5,sleep,"Waiting for rvc api to load") as pb:
                    start_server(host=state.host,port=state.port)
                    pb.run()
                    st.experimental_rerun()
                
        active_subprocess_list()
        
        if is_active: st_iframe(url=f'{SERVERS["RVC"]["url"]}/docs',height=800)