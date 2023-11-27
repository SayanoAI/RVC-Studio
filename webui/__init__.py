from functools import lru_cache
import os
import shelve
import sys
import weakref
from config import Config
from i18n import I18nAuto

MENU_ITEMS = {
    "Get help": "https://github.com/SayanoAI/RVC-Studio/discussions",
    "Report a Bug": "https://github.com/SayanoAI/RVC-Studio/issues",
    "About": """This project provides a comprehensive platform for training RVC models and generating AI voice covers.
    Check out this github for more info: https://github.com/SayanoAI/RVC-Studio
    """
}

DEVICE_OPTIONS = ["cpu","cuda"]
PITCH_EXTRACTION_OPTIONS = ["crepe","rmvpe","mangio-crepe","rmvpe+"]
TTS_MODELS = ["edge","speecht5"]
N_THREADS_OPTIONS=[1,2,4,8,12,16]
SR_MAP = {"32k": 32000,"40k": 40000, "48k": 48000}

class ObjectNamespace(dict):
    def __init__(self,**kwargs): super().__init__(kwargs)
    def __missing__(self, name: str): return ObjectNamespace()
    def get(self, name: str, default_value=None): return self.__getitem__(name) if name in self.keys() else default_value
    def __getattr__(self, name: str): return self.__getitem__(name) if name in self.keys() else ObjectNamespace()
    def __getitem__(self, name: str):
        value = super().__getitem__(name) # get the value from the parent class
        if isinstance(value, weakref.ref): # check if the value is a weak reference
            value = value() # call the weak reference object to get the referent
        return value # return the referent or the original value
    def __setattr__(self, name: str, value): return self.__setitem__(name, value)
    def __delattr__(self, name: str): return self.__delitem__(name) if name in self.keys() else None
    def __delitem__(self, name: str): return super().__delitem__(name) if name in self.keys() else None
    def __setstate__(self, state):
        for key in state: self[key]=state[key]
    def __getstate__(self): return dict(**self)

class PersistedDict:
    def __init__(self,fname,**kwargs):
        self.__fname__ = fname
        if len(kwargs):
            with shelve.open(fname) as shelf:
                for k in kwargs:
                    shelf[k] = kwargs[k]

    def __missing__(self, name: str): return print(f"Attribute {name} is missing")

    def __getitem__(self, name: str):
        if name.startswith("__") and name.endswith("__"): return super().__getattribute__(name)
        with shelve.open(self.__fname__) as shelf:
            return shelf.get(name,None)
        
    def __setitem__(self, name: str, value):
        with shelve.open(self.__fname__) as shelf:
            shelf[name] = value

    def get(self, name: str, value): return self.__getitem__(name) or value

    def set(self, name: str, value): return self.__setitem__(name, value)
    
@lru_cache
def load_config():
    return Config(), I18nAuto()

@lru_cache
def get_cwd():
    CWD = os.getcwd()
    if CWD not in sys.path:
        sys.path.append(CWD)
    return CWD

@lru_cache
def get_servers():
    servers = PersistedDict(os.path.join(get_cwd(),".cache","servers.shelve"))
    return servers

config, i18n = load_config()
SERVERS = get_servers()
RVC_INFERENCE_URL = f"{SERVERS['RVC']['url']}/rvc"
UVR_INFERENCE_URL = f"{SERVERS['RVC']['url']}/uvr"