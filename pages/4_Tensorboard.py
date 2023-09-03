import os
import sys
from streamlit_tensorboard import st_tensorboard
import streamlit as st
st.set_page_config(layout="wide")

CWD = os.getcwd()
if CWD not in sys.path:
    sys.path.append(CWD)
    
logdir=st.text_input("Path to Logs",value=os.path.join(CWD,"logs"))
if logdir: st_tensorboard(logdir=logdir, port=6006)