from functools import lru_cache
from lib import ObjectNamespace

@lru_cache
def get_status():
    return ObjectNamespace(status="OK",rvc=ObjectNamespace())

STATUS = get_status()