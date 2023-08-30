import argparse
import json
import multiprocessing
from multiprocessing.pool import ThreadPool
import os, sys, torch, warnings
from types import SimpleNamespace

from tqdm import tqdm
from lib.mdx import MDX, MDXModel

from webui_utils import gc_collect, load_input_audio, remix_audio, save_input_audio

now_dir = os.getcwd()
sys.path.append(now_dir)

warnings.filterwarnings("ignore")
import librosa
import numpy as np
from lib.uvr5_pack.lib_v5 import spec_utils
from lib.uvr5_pack.utils import inference
from lib.uvr5_pack.lib_v5.model_param_init import ModelParameters
import soundfile as sf
from lib.uvr5_pack.lib_v5.nets_new import CascadedNet
from lib.uvr5_pack.lib_v5 import nets_61968KB as nets


class UVR5Base:
    def __init__(self, agg, model_path, device, is_half,**kwargs):
        self.model_path = model_path
        self.device = device
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": False,
            # Constants
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        mp = ModelParameters("lib/uvr5_pack/lib_v5/modelparams/4band_v2.json")
        model = nets.CascadedASPPNet(mp.param["bins"] * 2)
        cpk = torch.load(model_path, map_location=self.device)
        model.load_state_dict(cpk)
        model.eval()
        if is_half:
            model = model.half().to(device)
        else:
            model = model.to(device)

        self.mp = mp
        self.model = model
    
    def process_vocals(self,v_spec_m,input_high_end,input_high_end_h,return_dict={}):
        if self.data["high_end_process"].startswith("mirroring"):
            input_high_end_ = spec_utils.mirroring(
                self.data["high_end_process"], v_spec_m, input_high_end, self.mp
            )
            wav_vocals = spec_utils.cmb_spectrogram_to_wave(
                v_spec_m, self.mp, input_high_end_h, input_high_end_
            )
        else:
            wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)
        print(f"vocals done: {wav_vocals.shape}")
        return_dict["vocals"] = remix_audio((wav_vocals,return_dict["sr"]),norm=True,to_int16=True,to_mono=True,axis=-1)
        return return_dict["vocals"]
    
    def process_instrumental(self,y_spec_m,input_high_end,input_high_end_h,return_dict={}):
        if self.data["high_end_process"].startswith("mirroring"):
            input_high_end_ = spec_utils.mirroring(
                self.data["high_end_process"], y_spec_m, input_high_end, self.mp
            )
            wav_instrument = spec_utils.cmb_spectrogram_to_wave(
                y_spec_m, self.mp, input_high_end_h, input_high_end_
            )
        else:
            wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)
        print(f"instruments done: {wav_instrument.shape}")
        return_dict["instrumentals"] = remix_audio((wav_instrument,return_dict["sr"]),norm=True,to_int16=True,to_mono=True,axis=-1)
        return return_dict["instrumentals"] 
    
    def process_audio(self,y_spec_m,v_spec_m,input_high_end,input_high_end_h):
        return_dict = {
            "sr": self.mp.param["sr"]
        }
        
        with ThreadPool(2) as pool:
            pool.apply(self.process_vocals, args=(v_spec_m,input_high_end,input_high_end_h,return_dict))
            pool.apply(self.process_instrumental, args=(y_spec_m,input_high_end,input_high_end_h,return_dict))
 
        return return_dict

    def run_inference(self, music_file):
        X_wave,  X_spec_s = {}, {}
        bands_n = len(self.mp.param["band"])
        # print(bands_n)
        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]
            if d == bands_n:  # high-end band
                input_audio = librosa.core.load(music_file,sr=bp["sr"],res_type=bp["res_type"])
                X_wave[d] = input_audio[0]
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
            else:  # lower bands
                X_wave[d] = librosa.core.resample(
                    X_wave[d + 1],
                    self.mp.param["band"][d + 1]["sr"],
                    bp["sr"],
                    res_type=bp["res_type"],
                )
            # Stft of wave source
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                X_wave[d],
                bp["hl"],
                bp["n_fft"],
                self.mp.param["mid_side"],
                self.mp.param["mid_side_b2"],
                self.mp.param["reverse"],
            )
            # pdb.set_trace()
            if d == bands_n and self.data["high_end_process"] != "none":
                input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                    self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"]
                )
                input_high_end = X_spec_s[d][
                    :, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :
                ]

        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        aggresive_set = float(self.data["agg"] / 100)
        aggressiveness = {
            "value": aggresive_set,
            "split_bin": self.mp.param["band"][1]["crop_stop"],
        }

        # pred, X_mag, X_phase = run_inference(X_spec_m, self.device, self.models, aggressiveness, self.data)
        with torch.no_grad():
            pred, X_mag, X_phase = inference(
                X_spec_m, self.device, self.model, aggressiveness, self.data
            )

    #     return pred, X_mag, X_phase, X_spec_m, input_high_end,input_high_end_h

    # def run_process_audio(self, pred, X_mag, X_phase, X_spec_m, input_high_end,input_high_end_h):
        # Postprocess
        if self.data["postprocess"]:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m

        
        return_dict = self.process_audio(y_spec_m,v_spec_m,input_high_end,input_high_end_h)
        return_dict["input_audio"] = input_audio
        
        return return_dict

class UVR5Dereverb(UVR5Base):
    def __init__(self, agg, model_path, device, is_half,**kwargs):
        self.model_path = model_path
        self.device = device
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": False,
            # Constants
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        mp = ModelParameters("lib/uvr5_pack/lib_v5/modelparams/4band_v3.json")
        nout = 64 if "DeReverb" in model_path else 48
        model = CascadedNet(mp.param["bins"] * 2, nout)
        cpk = torch.load(model_path, map_location=self.device)
        model.load_state_dict(cpk)
        model.eval()
        if is_half:
            model = model.half().to(device)
        else:
            model = model.to(device)

        self.mp = mp
        self.model = model
    
class MDXNet:
    def __init__(self, model_path, chunks=15,denoise=False,num_threads=1,device="cpu",**kwargs):
        model_hash = MDX.get_hash(model_path)
        with open(os.path.join(os.path.dirname(model_path), 'model_data.json')) as infile:
            model_params = json.load(infile)
        mp = model_params.get(model_hash)

        self.chunks = chunks
        self.sr = 16000 if denoise else 44100
        
        self.args = SimpleNamespace(**kwargs)
        self.denoise = denoise
        self.num_threads = num_threads

        self.device = device
        self.params = MDXModel(
            self.device,
            dim_f=mp["mdx_dim_f_set"],
            dim_t=2 ** mp["mdx_dim_t_set"],
            n_fft=mp["mdx_n_fft_scale_set"],
            stem_name=mp["primary_stem"],
            compensation=mp["compensate"]
        )
        # self.mp = SimpleNamespace(param={"sr": self.margin})

        self.model = MDX(model_path, self.params,device=self.device,chunks=self.chunks,margin=self.sr)
        print(f"onnx load done: {self.model} ({model_hash})")

    def __del__(self):
        del self.params, self.model
        gc_collect()

    def process_audio(self,background,foreground,target_sr=None):
        target_sr =  self.sr if target_sr is None else target_sr
        # foreground is processed data
        instrumental,vocals = (foreground,background) if "instrument" in self.params.stem_name.lower() else (background,foreground)
        with ThreadPool(2) as pool:
            results = pool.starmap(remix_audio, [
                ((instrumental,self.sr),target_sr,False,True,self.sr!=target_sr,True),
                ((vocals,self.sr),target_sr,False,True,self.sr!=target_sr,True)
            ])

        return_dict = {
            "sr": target_sr,
            "instrumentals": results[0],
            "vocals": results[1]
        }
        return return_dict
    
    def run_inference(self, audio_path):
        input_audio = load_input_audio(audio_path, mono=False)
        mix, _ = remix_audio(input_audio,target_sr=self.sr,to_mono=False,norm=True)
        
        if mix.ndim == 1:
            mix = np.stack([mix, mix],axis=0)

        if self.denoise:
            with ThreadPool(2) as pool:
                pooled_data = pool.map(self.model.process_wave,[mix,-mix])
            wave_processed = (pooled_data[0] - pooled_data[1])*0.5
        else:
            wave_processed = self.model.process_wave(mix, self.num_threads )
        # print(wave_processed.shape,mix.shape)
        return_dict = self.process_audio(background=(mix-wave_processed*self.params.compensation),foreground=wave_processed,target_sr=input_audio[1])
        return_dict["input_audio"] = input_audio

        return return_dict

class UVR5_Model:
    def __init__(self, model_path, use_cache=False, **kwargs):
        denoise = any([ele in model_path.lower() for ele in ["echo","noise","reverb"]])
        if "MDX" in model_path:
            self.model = MDXNet(model_path=model_path,denoise=denoise,**kwargs)
        elif "UVR" in model_path:
            self.model = UVR5Dereverb(model_path=model_path,**kwargs) if denoise else UVR5Base(model_path=model_path,**kwargs)
            
        self.use_cache = use_cache
        self.model_path = model_path
        self.args = kwargs
    
    # cleanup memory
    def __del__(self):
        gc_collect()

    def run_inference(self, audio_path):
        name = get_filename(self.model_path,audio_path,**self.args)
        
        # handles loading of previous processed data
        music_dir = os.path.join(os.path.dirname(audio_path))
        vocals_path = os.path.join(music_dir,".cache",".vocals")
        instrumental_path = os.path.join(music_dir,".cache",".instrumental")
        vocals_file = os.path.join(vocals_path,name)
        instrumental_file = os.path.join(instrumental_path,name)
        os.makedirs(vocals_path,exist_ok=True)
        os.makedirs(instrumental_path,exist_ok=True)
        # input_audio = load_input_audio(audio_path,mono=True)

        if os.path.exists(instrumental_file) and os.path.exists(vocals_file):
            vocals = load_input_audio(vocals_file,mono=True)
            instrumental = load_input_audio(instrumental_file,mono=True)
            input_audio = load_input_audio(audio_path,mono=True)
            return vocals, instrumental, input_audio
        
        return_dict = self.model.run_inference(audio_path)
        instrumental = return_dict["instrumentals"]
        vocals = return_dict["vocals"]
        input_audio = return_dict["input_audio"]

        if self.use_cache:
            save_input_audio(vocals_file,vocals,to_int16=True)
            save_input_audio(instrumental_file,instrumental,to_int16=True)

        return vocals, instrumental, input_audio

def get_filename(model_path,audio_path,agg,**kwargs):
    name = "_".join([str(agg),os.path.basename(model_path).split(".")[0],os.path.basename(audio_path)])
    return name

def __run_inference_worker(arg):
    (model_path,audio_path,agg,device,use_cache) = arg
    
    model = UVR5_Model(
            agg=agg,
            model_path=model_path,
            device=device,
            is_half=device=="cuda",
            use_cache=use_cache,
            # num_threads=num_threads
            )
    vocals, instrumental, input_audio = model.run_inference(audio_path)

    return vocals, instrumental, input_audio
    
def split_audio(uvr5_models,audio_path,preprocess_model=None,device="cuda",agg=10,use_cache=False,merge_type="mean"):
    
    # if "cuda" in device: torch.multiprocessing.set_start_method("spawn")
    # pooled_data = []

    if preprocess_model:
        model = UVR5_Model(
            agg=agg,
            model_path=preprocess_model,
            device=device,
            is_half="cuda" in device,
            use_cache=use_cache,
            num_threads = max(os.cpu_count()//2,1)
            )
        print("preprocessing")
        _, instrumental, input_audio = model.run_inference(audio_path)
        print(f"{instrumental[0].max()}, {instrumental[0].min()}, sr={instrumental[1]}")
        # saves preprocessed file to cache and use as input
        name = get_filename(preprocess_model,"",device=device,agg=agg)
        music_dir = os.path.join(os.path.dirname(audio_path))
        processed_path = os.path.join(music_dir,".cache",name)
        os.makedirs(processed_path,exist_ok=True)
        audio_path = os.sep.join([processed_path,os.path.basename(audio_path)])
        if not os.path.exists(audio_path):
            save_input_audio(audio_path,instrumental,to_int16=True)
    else:
        input_audio = load_input_audio(audio_path,mono=True)
    
    # num_threads = 1 # max(os.cpu_count()//(len(uvr5_models)*2),1)
    # args = [(model_path,audio_path,agg,device,use_cache,num_threads) for model_path in uvr5_models]
    # with multiprocessing.Pool(len(args)) as pool:
    #     pooled_data = pool.map(__run_inference_worker,args)
    # # pool.join()
        
    wav_instrument = []
    wav_vocals = []
    max_len = 0

    # for ( vocals, instrumental, _) in pooled_data:
    for model_path in uvr5_models:
        args = (model_path,audio_path,agg,device,use_cache)
        vocals, instrumental, _ = __run_inference_worker(args)
        wav_vocals.append(vocals[0])
        wav_instrument.append(instrumental[0])
        max_len = max(max_len,len(vocals[0]),len(instrumental[0]))

    merge_func = np.nanmedian if merge_type=="median" else np.nanmean
    wav_instrument = merge_func([librosa.util.pad_center(wav,max_len) for wav in wav_instrument],axis=0)
    wav_vocals = merge_func([librosa.util.pad_center(wav,max_len) for wav in wav_vocals],axis=0)
    instrumental = remix_audio((wav_instrument,instrumental[1]),norm=True,to_int16=True,to_mono=True)
    vocals = remix_audio((wav_vocals,vocals[1]),norm=True,to_int16=True,to_mono=True)

    return vocals, instrumental, input_audio

def main(): #uvr5_models,audio_path,device="cuda",agg=10,use_cache=False
    
    parser = argparse.ArgumentParser(description="processes audio to split vocal stems and reduce reverb/echo")
    
    parser.add_argument("uvr5_models", type=str, nargs="+", help="Path to models to use for processing", required=True)
    parser.add_argument(
        "-i", "--audio_path", type=str, help="path to audio file to process", required=True
    )
    parser.add_argument(
        "-p", "--preprocess_model", type=str, help="preprocessing model to improve audio", default=None
    )
    parser.add_argument(
        "-a", "--agg", type=int, default=10, help="aggressiveness score for processing (0-20)"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cpu", choices=["cpu","cuda"], help="perform calculations on [cpu] or [cuda]"
    )
    parser.add_argument(
        "-m", "--merge_type", type=str, default="median", choices=["mean","median"], help="how to combine processed audio"
    )
    parser.add_argument(
        "-c", "--use_cache", type=bool, action="store_true", default=False, help="caches the results so next run is faster"
    )
    args = parser.parse_args()
    return split_audio(**vars(args))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()