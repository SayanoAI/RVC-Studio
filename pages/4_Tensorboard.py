import os
import subprocess
import sys
import psutil
from streamlit_tensorboard import st_tensorboard
from web_utils import MENU_ITEMS
import streamlit as st
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from webui_utils import render_subprocess_list
from web_utils.contexts import SessionStateContext


CWD = os.getcwd()
if CWD not in sys.path:
    sys.path.append(CWD)
    
def start_tensorboard(logdir):
    cmd = f"tensorboard --logdir={logdir}"
    p = subprocess.Popen(cmd, shell=True, cwd=CWD,stdout=subprocess.PIPE)
    return p

with SessionStateContext("tensorboard") as state:
    state.logdir=st.text_input("Path to Logs",value=state.logdir if state.logdir else os.path.join(CWD,"logs"))
    if state.logdir:
        st_tensorboard(logdir=state.logdir, port=6006)

        if not state.process:
            if not any(["tensorboard" in p.name() for p in psutil.Process(os.getpid()).children(recursive=True)]):
                if st.button("Start Tensorboard"):
                    state.process = start_tensorboard(state.logdir)

    render_subprocess_list()