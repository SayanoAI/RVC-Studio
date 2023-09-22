from functools import partial
from multiprocessing.pool import ThreadPool
import os
import random
import numpy as np
from scipy import signal
import torch, torchcrepe, pyworld

from lib.rmvpe import RMVPE
from webui.audio import autotune_f0, pad_audio
from webui.downloader import BASE_MODELS_DIR
from webui.utils import get_optimal_threads, get_optimal_torch_device

class FeatureExtractor:
    def __init__(self, tgt_sr, config, onnx=False):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值
        self.device = config.device
        self.onnx = onnx
        self.f0_method_dict = {
            "pm": self.get_pm,
            "harvest": self.get_harvest,
            "dio": self.get_dio,
            "rmvpe": self.get_rmvpe,
            "rmvpe_onnx": self.get_rmvpe,
            "rmvpe+": self.get_pitch_dependant_rmvpe,
            "crepe": self.get_f0_official_crepe_computation,
            "crepe-tiny": partial(self.get_f0_official_crepe_computation, model='model'),
            "mangio-crepe": self.get_f0_crepe_computation,
            "mangio-crepe-tiny": partial(self.get_f0_crepe_computation, model='model'),
            
        }
        

    # Fork Feature: Compute f0 with the crepe method
    def get_f0_crepe_computation(
        self,
        x,
        f0_min,
        f0_max,
        p_len,
        *args,  # 512 before. Hop length changes the speed that the voice jumps to a different dramatic pitch. Lower hop lengths means more pitch accuracy but longer inference time.
        **kwargs,  # Either use crepe-tiny "tiny" or crepe "full". Default is full
    ):
        x = x.astype(
            np.float32
        )  # fixes the F.conv2D exception. We needed to convert double to float.
        x /= np.quantile(np.abs(x), 0.999)
        torch_device = get_optimal_torch_device()
        audio = torch.from_numpy(x).to(torch_device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()
        hop_length = kwargs.get('crepe_hop_length', 160)
        model = kwargs.get('model', 'full') 
        print("Initiating prediction with a crepe_hop_length of: " + str(hop_length))
        pitch: torch.Tensor = torchcrepe.predict(
            audio,
            self.sr,
            hop_length,
            f0_min,
            f0_max,
            model,
            batch_size=hop_length * 2,
            device=torch_device,
            pad=True,
        )
        p_len = p_len or x.shape[0] // hop_length
        # Resize the pitch for final f0
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        f0 = np.nan_to_num(target)
        return f0  # Resized f0
    
    def get_f0_official_crepe_computation(
        self,
        x,
        f0_min,
        f0_max,
        *args,
        **kwargs
    ):
        # Pick a batch size that doesn't cause memory errors on your gpu
        batch_size = 512
        # Compute pitch using first gpu
        audio = torch.tensor(np.copy(x))[None].float()
        model = kwargs.get('model', 'full') 
        f0, pd = torchcrepe.predict(
            audio,
            self.sr,
            self.window,
            f0_min,
            f0_max,
            model,
            batch_size=batch_size,
            device=self.device,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()
        return f0

    def get_pm(self, x, p_len, *args, **kwargs):
        import parselmouth
        f0 = parselmouth.Sound(x, self.sr).to_pitch_ac(
            time_step=160 / 16000,
            voicing_threshold=0.6,
            pitch_floor=kwargs.get('f0_min'),
            pitch_ceiling=kwargs.get('f0_max'),
        ).selected_array["frequency"]
        
        return np.pad(
            f0,
            [[max(0, (p_len - len(f0) + 1) // 2), max(0, p_len - len(f0) - (p_len - len(f0) + 1) // 2)]],
            mode="constant"
        )

    def get_harvest(self, x, *args, **kwargs):
        f0_spectral = pyworld.harvest(
            x.astype(np.double),
            fs=self.sr,
            f0_ceil=kwargs.get('f0_max'),
            f0_floor=kwargs.get('f0_min'),
            frame_period=1000 * kwargs.get('hop_length', 160) / self.sr,
        )
        return pyworld.stonemask(x.astype(np.double), *f0_spectral, self.sr)

    def get_dio(self, x, *args, **kwargs):
        f0_spectral = pyworld.dio(
            x.astype(np.double),
            fs=self.sr,
            f0_ceil=kwargs.get('f0_max'),
            f0_floor=kwargs.get('f0_min'),
            frame_period=1000 * kwargs.get('hop_length', 160) / self.sr,
        )
        return pyworld.stonemask(x.astype(np.double), *f0_spectral, self.sr)


    def get_rmvpe(self, x, *args, **kwargs):
        if not hasattr(self,"model_rmvpe"):
            self.model_rmvpe = RMVPE(os.path.join(BASE_MODELS_DIR,f"rmvpe.{'onnx' if self.onnx else 'pt'}"), is_half=self.is_half, device=self.device, onnx=self.onnx)

        # if self.onnx == False: 
        return self.model_rmvpe.infer_from_audio(x, thred=0.03)
        # else:
        #     f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        #     if "privateuseone" in str(self.device):
        #             del self.model_rmvpe.model
        #             del self.model_rmvpe
        #             print("cleaning ortruntime memory")
        #     return f0

    def get_pitch_dependant_rmvpe(self, x, f0_min=1, f0_max=40000, *args, **kwargs):
        if not hasattr(self,"model_rmvpe"):
            self.model_rmvpe = RMVPE(os.path.join(BASE_MODELS_DIR,f"rmvpe.{'onnx' if self.onnx else 'pt'}"), is_half=self.is_half, device=self.device, onnx=self.onnx)

        return self.model_rmvpe.infer_from_audio_with_pitch(x, thred=0.03, f0_min=f0_min, f0_max=f0_max)


    # Fork Feature: Acquire median hybrid f0 estimation calculation
    def get_f0_hybrid_computation(
        self,
        methods_list,
        merge_type,
        x,
        f0_min,
        f0_max,
        p_len,
        filter_radius,
        crepe_hop_length,
        time_step,
        **kwargs
    ):
        # Get various f0 methods from input to use in the computation stack
        params = {'x': x, 'p_len': p_len, 'f0_min': f0_min, 
          'f0_max': f0_max, 'time_step': time_step, 'filter_radius': filter_radius, 
          'crepe_hop_length': crepe_hop_length, 'model': "full"
        }
        
        f0_computation_stack = []

        print(f"Calculating f0 pitch estimations for methods: {methods_list}")
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        # Get f0 calculations for all methods specified

        def _get_f0(method,params):
            if method not in self.f0_method_dict:
                raise Exception(f"Method {method} not found.")
            f0 = self.f0_method_dict[method](**params)
            if method == 'harvest' and filter_radius > 2:
                f0 = signal.medfilt(f0, filter_radius)
                f0 = f0[1:]  # Get rid of first frame.
            return f0

        with ThreadPool(get_optimal_threads()) as pool:
            f0_computation_stack = pool.starmap(_get_f0,[(method,params) for method in methods_list])

        f0_computation_stack = pad_audio(*f0_computation_stack) # prevents uneven f0

        print(f"Calculating hybrid median f0 from the stack of: {methods_list} using {merge_type} merge")
        merge_func = np.nanmedian if merge_type=="median" else np.nanmean
        f0_median_hybrid = merge_func(f0_computation_stack, axis=0)

        return f0_median_hybrid

    def get_f0(
        self,
        x,
        p_len,
        f0_up_key,
        f0_method,
        merge_type="median",
        filter_radius=3,
        crepe_hop_length=160,
        f0_autotune=False,
        rmvpe_onnx=False,
        inp_f0=None,
        f0_min=50,
        f0_max=1100,
    ):
        time_step = self.window / self.sr * 1000
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        params = {'x': x, 'p_len': p_len, 'f0_up_key': f0_up_key, 'f0_min': f0_min, 
          'f0_max': f0_max, 'time_step': time_step, 'filter_radius': filter_radius, 
          'crepe_hop_length': crepe_hop_length, 'model': "full", 'onnx': rmvpe_onnx
        }

        if type(f0_method) == list:
            # Perform hybrid median pitch estimation
            f0 = self.get_f0_hybrid_computation(f0_method,merge_type,**params)
        else:
            print(f"f0_method={f0_method}")
            f0 = self.f0_method_dict[f0_method](**params)

        if f0_autotune:
            f0 = autotune_f0(f0)

        f0 *= pow(2, f0_up_key / 12)
        # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        tf0 = self.sr // self.window  # 每秒f0点数
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]
        
        # f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int)

        return f0_coarse, f0  # 1-0