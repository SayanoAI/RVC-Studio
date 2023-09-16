import os
import subprocess
import sys
from time import sleep
import psutil
from streamlit_tensorboard import st_tensorboard
from webui import MENU_ITEMS
import streamlit as st
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from webui.components import active_subprocess_list
from webui.contexts import ProgressBarContext, SessionStateContext


CWD = os.getcwd()
if CWD not in sys.path:
    sys.path.append(CWD)
    
def start_tensorboard(logdir):
    cmd = f"tensorboard --logdir={logdir}"
    p = subprocess.Popen(cmd, shell=True, cwd=CWD)
    return p

if __name__=="__main__":
    with SessionStateContext("tensorboard") as state:
        state.logdir=st.text_input("Path to Logs",value=state.logdir if state.logdir else os.path.join(CWD,"logs"))
        if state.logdir:
            st_tensorboard(logdir=state.logdir, port=6006)
            placeholder = st.container()
            tensorboard_is_active = any(["tensorboard" in p.name() for p in psutil.Process(os.getpid()).children(recursive=True)])
            if st.button("Start Tensorboard", disabled=tensorboard_is_active):
                with ProgressBarContext([1]*5,sleep,"Waiting for tensorboard to load") as pb:
                    start_tensorboard(state.logdir)
                    pb.run()
                    st.experimental_rerun()

        active_subprocess_list()