import os
import subprocess
import time
import psutil
import random
import html
import json
from webui import MENU_ITEMS
from lib import BASE_DIR, LOG_DIR
import streamlit as st
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from webui.components import active_subprocess_list
from webui.contexts import ProgressBarContext, SessionStateContext

## Ported from streamlit_tensorboard with modifications
def st_tensorboard(url="http://localhost:6006", width=None, height=800, scrolling=True):
    """Embed Tensorboard within a Streamlit app
    Parameters
    ----------
    url: string
        URL of the Tensorboard server. Defaults to `http://localhost:6060`
    width: int
        The width of the frame in CSS pixels. Defaults to the report’s default element width.
    height: int
        The height of the frame in CSS pixels. Defaults to 800.
    scrolling: bool
        If True, show a scrollbar when the content is larger than the iframe.
        Otherwise, do not show a scrollbar. Defaults to True.

    Example
    -------
    >>> st_tensorboard(url="http://localhost:6006", width=1080)
    """

    frame_id = "tensorboard-frame-{:08x}".format(random.getrandbits(64))
    shell = """
        <iframe id="%HTML_ID%" width="100%" height="%HEIGHT%" frameborder="0">
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
        ("%HEIGHT%", "%d" % height),
        ("%URL%", json.dumps(url)),
    ]

    for (k, v) in replacements:
        shell = shell.replace(k, v)

    return st.components.v1.html(shell, width=width, height=height, scrolling=scrolling)
    
def start_tensorboard(logdir, host="localhost"):
    cmd = f"tensorboard --logdir={logdir} --host={host}"
    p = subprocess.Popen(cmd, cwd=BASE_DIR)
    return p

if __name__=="__main__":
    with SessionStateContext("tensorboard") as state:
        state.url = st.text_input("Tensorboard URL", value = state.url if state.url else "http://localhost:6006")
        if state.url:
            st_tensorboard(url="http://localhost:6006")
        placeholder = st.container()
        tensorboard_is_active = any(["tensorboard" in p.name() for p in psutil.Process(os.getpid()).children(recursive=True)])
        state.logdir=st.text_input("Path to Logs",value=state.logdir if state.logdir else LOG_DIR)
        state.remote_bind = st.checkbox("Bind to 0.0.0.0(Required for docker or remote connections)", value=state.remote_bind if state.remote_bind else False)
        if st.button("Start Tensorboard", disabled=tensorboard_is_active):
            with ProgressBarContext([1]*5,time.sleep,"Waiting for tensorboard to load") as pb:
                start_tensorboard(state.logdir, "localhost" if not state.remote_bind else "0.0.0.0")
                pb.run()
                st.experimental_rerun()

        active_subprocess_list()
