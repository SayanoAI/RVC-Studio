from typing import List, Literal
from pydantic import BaseModel


class RVCInferenceParams(BaseModel):
    name: str
    audio_data: str
    f0_up_key: int=0
    f0_method: List[Literal["rmvpe","rmvpe+","crepe","mangio-crepe"]]=["rmvpe"]
    f0_autotune: bool=False
    merge_type: Literal["median","mean"]="median"
    index_rate: float=.75
    resample_sr: int=0
    rms_mix_rate: float=.25
    protect: float=0.25
    filter_radius: int=3

class UVRInferenceParams(BaseModel):
    uvr_models: List[str]
    audio_data: str
    preprocess_models: List[str]=[]
    postprocess_models: List[str]=[]
    agg: int=10
    merge_type: Literal["median","mean"]="median"
    use_cache: bool=True
    format: Literal["mp3","flac","wav"]="flac"

class UVRRVCInferenceParams(BaseModel):
    uvr_params: UVRInferenceParams
    rvc_params: RVCInferenceParams
    audio_data: str