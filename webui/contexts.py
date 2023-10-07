from multiprocessing import Lock, cpu_count
from multiprocessing.pool import ThreadPool
from types import FunctionType
from typing import List
import streamlit as st
from webui.utils import ObjectNamespace, gc_collect

class SessionStateContext:
    def __init__(self, name: str, initial_state={}):
        self.__data__ = ObjectNamespace(**initial_state)
        self.__name__ = name
        self.__initial_state__ = ObjectNamespace() if initial_state is None else initial_state
    
    def __enter__(self):
        if self.__name__ in st.session_state: return st.session_state[self.__name__]
        st.session_state[self.__name__] = self.__data__
        return self.__data__
    
    def __exit__(self, *_):
        if self.__name__ not in st.session_state: st.session_state[self.__name__] = self.__data__
        gc_collect()
    
    def __dir__(self):
        return self.data.__dir__
    def __str__(self):
        return str(self.__data__)
    def __repr__(self):
        return f"SessionStateContext('{self.__name__}',{self.__data__})"

class ProgressBarContext:
    def __init__(self, iter: List, func: FunctionType, text: str="", parallel=False):
        
        self.max_progress = len(iter)
        self.progress = 0
        self.args = iter
        self.func = func
        self.text = text
        self.parallel = parallel
        self.__progressbar__ = st.progress(0, text)
        self.lock = Lock()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        del self.__progressbar__

    def run(self):
        self.__progressbar__.progress(0.0,f"{self.text}: {0}/{self.max_progress}")

        if self.parallel:
            with ThreadPool(min(cpu_count(),self.max_progress)) as pool:
                pool.map(self.__runner__,range(self.max_progress ))
        else:
            for i in range(self.max_progress ):
                self.__runner__(i)
                # self.__progressbar__.progress(float((i+1)/self.max_progress),f"{self.text}: {i+1}/{self.max_progress}")

    def __runner__(self,index):
        self.func(self.args[index])
        self.progress+=1
        self.__progressbar__.progress(float(self.progress/self.max_progress),f"{self.text}: {self.progress}/{self.max_progress}")

# TODO: show terminal logs in streamlit
from contextlib import contextmanager
from io import StringIO
from threading import current_thread
import sys
from streamlit.runtime.scriptrunner.script_run_context import SCRIPT_RUN_CONTEXT_ATTR_NAME

@contextmanager
def st_redirect(src, dst):
    # placeholder = st.empty()
    output_func = dst #getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME, None):
                try:
                    buffer.write(b)
                    output_func(f"{buffer.getvalue()}")
                except:
                    old_write(b)        
            old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stdout(placeholder=st.empty(),output="info"):
    with st_redirect(sys.stdout, getattr(placeholder,output)):
        yield


@contextmanager
def st_stderr(placeholder=st.empty(),output="error"):
    with st_redirect(sys.stderr, getattr(placeholder,output)):
        yield