from typing import Tuple
import streamlit as st

from webui import i18n
from webui.contexts import ProgressBarContext
from webui.downloader import save_file, save_file_generator
from webui.utils import gc_collect, get_subprocesses

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
                    st.experimental_rerun()

def active_subprocess_list():
    with st.expander(i18n("process.pids")):
        for p in get_subprocesses():
            col1,col2,col3,col4=st.columns(4)
            col1.write(p.pid)
            col2.write(p.name)
            col3.write(p.time_started)
            if col4.button(i18n("process.kill_one_pid"),key=f"process.kill_one_pid.{p.pid}"):
                for c in get_subprocesses(p.pid):
                    c.kill()
                p.kill()
                gc_collect()
                st.experimental_rerun()