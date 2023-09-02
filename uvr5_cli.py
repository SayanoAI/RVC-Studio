import argparse
import json
import multiprocessing
from multiprocessing.pool import ThreadPool
import os, sys, torch, warnings
from types import SimpleNamespace

from tqdm import tqdm
from lib.mdx import MDX, MDXModel
import hashlib
import math
from webui_utils import gc_collect, load_input_audio, remix_audio, save_input_audio

CWD = os.getcwd()
sys.path.append(CWD)
CACHE_DIR = os.path.join(CWD,".cache","songs")

warnings.filterwarnings("ignore")
import librosa
import numpy as np
from lib.uvr5_pack.lib_v5 import spec_utils
from lib.uvr5_pack.lib_v5.dataset import make_padding
from lib.uvr5_pack.lib_v5.model_param_init import ModelParameters
import soundfile as sf
from lib.uvr5_pack.lib_v5.nets_new import CascadedNet
from lib.uvr5_pack.lib_v5.nets import CascadedASPPNet

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
        model = CascadedASPPNet(mp.param["bins"] * 2)
        cpk = torch.load(model_path, map_location=self.device)
        model.load_state_dict(cpk)
        model.eval()
        if is_half:
            model = model.half().to(device)
        else:
            model = model.to(device)

        self.mp = mp
        self.model = model
    
    @staticmethod
    def get_model_params(model_path):
        try:
            with open(model_path, 'rb') as f:
                f.seek(- 10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except:
            model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()

        print(model_hash)
        model_settings_json = os.path.splitext(model_path)[0]+".json"
        model_data_json = os.path.join(os.path.dirname(model_path),"model_data.json")

        if os.path.isfile(model_settings_json):
            return json.load(open(model_settings_json))
        elif os.path.isfile(model_data_json):
            with open(model_data_json,"r") as d:
                hash_mapper = json.loads(d.read())

            for hash, settings in hash_mapper.items():
                if model_hash in hash:
                    return settings
        return None

    def inference(self, X_spec, aggressiveness):
        """
        data ： dic configs
        """
        data = self.data
        device = self.device
        model = self.model

        def _execute(
            X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half=True
        ):
            model.eval()
            with torch.no_grad():
                preds = []

                iterations = [n_window]

                total_iterations = sum(iterations)
                for i in tqdm(range(n_window)):
                    start = i * roi_size
                    X_mag_window = X_mag_pad[
                        None, :, :, start : start + data["window_size"]
                    ]
                    X_mag_window = torch.from_numpy(X_mag_window)
                    if is_half:
                        X_mag_window = X_mag_window.half()
                    X_mag_window = X_mag_window.to(device)

                    pred = model.predict(X_mag_window, aggressiveness)

                    pred = pred.detach().cpu().numpy()
                    preds.append(pred[0])

                pred = np.concatenate(preds, axis=2)
            return pred

        def preprocess(X_spec):
            X_mag = np.abs(X_spec)
            X_phase = np.angle(X_spec)

            return X_mag, X_phase

        X_mag, X_phase = preprocess(X_spec)

        coef = X_mag.max()
        X_mag_pre = X_mag / coef

        n_frame = X_mag_pre.shape[2]
        pad_l, pad_r, roi_size = make_padding(n_frame, data["window_size"], model.offset)
        n_window = int(np.ceil(n_frame / roi_size))

        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

        if list(model.state_dict().values())[0].dtype == torch.float16:
            is_half = True
        else:
            is_half = False
        pred = _execute(
            X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half
        )
        pred = pred[:, :, :n_frame]

        if data["tta"]:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            n_window += 1

            X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

            pred_tta = _execute(
                X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half
            )
            pred_tta = pred_tta[:, :, roi_size // 2 :]
            pred_tta = pred_tta[:, :, :n_frame]

            return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.0j * X_phase)
        else:
            return pred * coef, X_mag, np.exp(1.0j * X_phase)

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
            pred, X_mag, X_phase = self.inference(X_spec_m, aggressiveness)

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

class UVR5New(UVR5Base):
    def __init__(self, agg, model_path, device, is_half, dereverb, **kwargs):
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
        nout = 64 if dereverb else 48
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
        self.sr = 44100
        
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
            wave_processed = (self.model.process_wave(mix, self.num_threads) - self.model.process_wave(-mix, self.num_threads))*0.5
        else:
            wave_processed = self.model.process_wave(mix, self.num_threads )
        # print(wave_processed.shape,mix.shape)
        return_dict = self.process_audio(background=(mix-wave_processed*self.params.compensation),foreground=wave_processed,target_sr=input_audio[1])
        return_dict["input_audio"] = input_audio

        return return_dict

class UVR5_Model:
    def __init__(self, model_path, use_cache=False, device="cpu", cache_dir=CACHE_DIR, **kwargs):
        dereverb = "reverb" in model_path.lower()
        deecho = "echo"  in model_path.lower()
        denoise = dereverb or deecho

        if "MDX" in model_path:
            self.model = MDXNet(model_path=model_path,denoise=denoise,device=device,**kwargs)
        elif "UVR" in model_path:
            self.model = UVR5New(model_path=model_path,device=device,dereverb=dereverb,**kwargs) if denoise else UVR5Base(model_path=model_path,device=device,**kwargs)
            
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.model_path = model_path
        self.args = kwargs
    
    # cleanup memory
    def __del__(self):
        gc_collect()

    def run_inference(self, audio_path):
        song_name = get_filename(
            os.path.basename(self.model_path).split(".")[0],
            **self.args) + ".mp3"
        
        # handles loading of previous processed data
        music_dir = os.path.join(self.cache_dir,os.path.basename(audio_path).split(".")[0])
        vocals_path = os.path.join(music_dir,".vocals")
        instrumental_path = os.path.join(music_dir,".instrumental")
        vocals_file = os.path.join(vocals_path,song_name)
        instrumental_file = os.path.join(instrumental_path,song_name)
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

def get_filename(*args,**kwargs):
    name = "_".join([str(arg) for arg in args]+[f"{k}={v}" for k,v in kwargs.items()])
    return name

def __run_inference_worker(arg):
    (model_path,audio_path,agg,device,use_cache,cache_dir) = arg
    
    model = UVR5_Model(
            agg=agg,
            model_path=model_path,
            device=device,
            is_half=device=="cuda",
            use_cache=use_cache,
            cache_dir=cache_dir
            )
    vocals, instrumental, input_audio = model.run_inference(audio_path)
    del model
    gc_collect()

    return vocals, instrumental, input_audio
    
def split_audio(uvr5_models,audio_path,preprocess_model=None,device="cuda",agg=10,use_cache=False,merge_type="mean"):
    cache_dir = CACHE_DIR #default cache dir
    if preprocess_model:
        song_name = os.path.basename(audio_path).split(".")[0]
        output_name = get_filename(os.path.basename(preprocess_model).split(".")[0],agg=agg) + ".mp3"
        preprocess_path = os.path.join(CACHE_DIR,song_name,output_name)
        if not os.path.exists(preprocess_path):
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
            # saves preprocessed file to cache and use as input
            save_input_audio(preprocess_path,instrumental,to_int16=True)
            del model
            gc_collect()
        cache_dir = os.path.join(CACHE_DIR,song_name)
        audio_path = preprocess_path
    else:
        input_audio = load_input_audio(audio_path,mono=True)
        
    wav_instrument = []
    wav_vocals = []
    max_len = 0

    for model_path in uvr5_models:
        args = (model_path,audio_path,agg,device,use_cache,cache_dir)
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