from functools import lru_cache
import os

from lib import BASE_CACHE_DIR, PersistedDict

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

@lru_cache
def get_servers():
    os.makedirs(BASE_CACHE_DIR,exist_ok=True)
    fname = os.path.join(BASE_CACHE_DIR,"servers.shelve")
    servers = PersistedDict(fname)
    return servers

SERVERS = get_servers()
RVC_INFERENCE_URL = SERVERS.RVC_INFERENCE_URL
UVR_INFERENCE_URL = SERVERS.UVR_INFERENCE_URL