import os
import subprocess
from time import sleep
import psutil

from webui import MENU_ITEMS, SERVERS
from lib import ObjectNamespace, BASE_DIR, config
import streamlit as st
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from lib.utils import get_subprocesses, pid_is_active, poll_url
from webui.components import active_subprocess_list, st_iframe
from webui.contexts import ProgressBarContext, SessionStateContext

def stop_server(pid):
    if pid_is_active(pid):
        process = psutil.Process(pid)
        if process.is_running():
            for sub in get_subprocesses(pid):
                sub.kill()
            process.kill()

def start_server(host,port):
    pid = SERVERS.INFERENCE_PID
    if pid_is_active(pid):
        process = psutil.Process(pid)
        if process.is_running(): return SERVERS.url
    
    base_url = f"http://{host}:{port}"
    cmd = f"{config.python_cmd} {os.path.join(BASE_DIR,'api.py')} --port={port} --host={host}"
    p = subprocess.Popen(cmd, cwd=BASE_DIR)

    if poll_url(base_url):
        SERVERS.RVC_INFERENCE_URL = f"{base_url}/rvc"
        print(SERVERS)
        SERVERS.UVR_INFERENCE_URL = f"{base_url}/uvr"
        print(SERVERS)
        SERVERS.DOCS_URL = f"{base_url}/docs"
        print(SERVERS)
        SERVERS.INFERENCE_PID = p.pid
        print(SERVERS)
    
    return base_url

def initial_state():
    return ObjectNamespace(
        remote_bind=False,
        host="localhost",
        port=5555
    )

if __name__=="__main__":
    with SessionStateContext("rvc-api",initial_state()) as state:
        pid = SERVERS.INFERENCE_PID
        is_active = pid_is_active(pid)

        with st.form("rvc-api-form"):
            state.remote_bind = st.checkbox("Bind to 0.0.0.0 (Required for docker or remote connections)", value=state.remote_bind)
            state.host = "0.0.0.0" if state.remote_bind else "localhost"
            state.port = st.number_input("Port", value=state.port or 5555)
            state.url = st.text_input("Server URL", value = f"http://{state.host}:{state.port}")

            if st.form_submit_button("Start Server",disabled=is_active):
                with ProgressBarContext([1],sleep,"Waiting for rvc api to load") as pb:
                    start_server(host=state.host,port=state.port)
                    pb.run()
                    st.experimental_rerun()
                
        active_subprocess_list()
        
        if is_active:
            if st.button(f"Stop Server [{pid}]",type="primary"):
                stop_server(pid)
                start_server(host=state.host,port=state.port)
                st.experimental_rerun()

            st_iframe(url=SERVERS.DOCS_URL,height=800)