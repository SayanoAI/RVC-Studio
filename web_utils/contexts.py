from types import FunctionType
from typing import List
import streamlit as st

class SessionStateContext:
    def __init__(self, name: str, initial_state={}):
        self.__data__ = st.session_state.get(name,initial_state)
        self.__name__ = name
        self.__initial_state__ = {} if initial_state is None else initial_state
    
    def __enter__(self):
        # print("Entering the context")
        # print(f"Acquiring {self}")
        return self
    def __exit__(self, *_):
        # print(exc_type, exc_value, traceback)
        # print("Exiting the context")
        # print(f"Releasing {repr(self)}")
        st.session_state[self.__name__] = self.__data__
    
    def __dir__(self):
        return self.data.__dir__
    def __str__(self):
        return str(self.__data__)
    def __repr__(self):
        return f"SessionStateContext('{self.__name__}',{self.__data__})"
    
    def __getitem__(self, name: str):
        if name in self.__data__:
            return self.__data__[name]
        else:
            return self.__getattr__(name)
        
    def __setitem__(self, name: str, value):
        if name in self.__data__:
            self.__data__[name] = value
        else:
            self.__setattr__(name, value)
    def __delitem__(self,name):
        if name in self.__data__:
            del self.__data__[name]
        else:
            self.__delattr__(name)
        
    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            return super().__getattr__(name)
        else:
            return self.__data__.get(name)
    def __setattr__(self, name: str, value):
        if name.startswith("__") and name.endswith("__"):
            super().__setattr__(name, value)
        else:
            self.__data__[name] = value
    def __delattr__(self,name):
        if name.startswith("__") and name.endswith("__"):
            super().__delattr__(name)
        elif name in self.__data__:
            del self.__data__[name]
        else:
            print(f"Failed to delete {name}")

class ProgressBarContext:
    def __init__(self, iter: List, func: FunctionType, text: str=""):
        
        self.max_progress = len(iter)
        self.args = iter
        self.func = func
        self.text = text
        self.__progressbar__ = st.progress(0, text)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        del self
    #     return self.__progressbar__.progress(100, f"{self.text}: finished")

    def run(self):
        for i in range(self.max_progress ):
            self.__progressbar__.progress(float(i/self.max_progress),f"{self.text}: {i}/{self.max_progress}")
            self.func(self.args[i])

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