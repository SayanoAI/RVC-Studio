from multiprocessing.pool import ThreadPool
import sys, os, multiprocessing
from threading import Thread
from scipy import signal
import numpy as np, os, traceback
from lib.slicer2 import Slicer
import librosa, traceback
from scipy.io import wavfile
from lib.audio import load_audio
from pitch_extraction import FeatureExtractor
from vc_infer_pipeline import load_hubert
from webui.audio import load_input_audio
from webui.utils import gc_collect
from webui import config
import torch

class Preprocess:
    def __init__(self, sr, exp_dir, noparallel=True):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = 3.0
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
        self.wavs16k_dir = "%s/1_16k_wavs" % exp_dir
        self.noparallel = noparallel
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def println(self,strr):
        # mutex.acquire()
        print(strr)
        with open("%s/preprocess.log" % self.exp_dir, "a+") as f:
            f.write("%s\n" % strr)
            f.flush()
        # mutex.release()

    def norm_write(self, tmp_audio, idx0, idx1):
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            print("%s-%s-%s-filtered" % (idx0, idx1, tmp_max))
            return
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            "%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000
        )  # , res_type="soxr_vhq"
        wavfile.write(
            "%s/%s_%s.wav" % (self.wavs16k_dir, idx0, idx1),
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self, path, idx0):
        try:
            audio = load_audio(path, self.sr)
            # zero phased digital filter cause pre-ringing noise...
            # audio = signal.filtfilt(self.bh, self.ah, audio)
            # audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        idx1 += 1
                        break
                self.norm_write(tmp_audio, idx0, idx1)
            self.println("%s->Suc." % path)
        except:
            self.println("%s->%s" % (path, traceback.format_exc()))

    def pipeline_mp(self, infos):
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            infos = [
                ("%s/%s" % (inp_root, name), idx)
                for idx, name in enumerate(sorted(list(os.listdir(inp_root))))
            ]
            if self.noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p],)
                    )
                    ps.append(p)
                    p.start()
                for i in range(n_p):
                    ps[i].join()
        except:
            self.println("Fail. %s" % traceback.format_exc())

class FeatureInput(FeatureExtractor):
    def __init__(self, f0_method, exp_dir, samplerate=16000, hop_size=160, device="cpu", version="v2", if_f0=False):
        self.sr = samplerate
        self.hop = hop_size
        self.f0_method = f0_method
        self.exp_dir = exp_dir
        self.device = device
        self.version = version
        self.if_f0 = if_f0

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        self.model = load_hubert(config)
        
        super().__init__(samplerate, config, onnx=False)

    def printt(self,strr):
        print(strr)
        with open("%s/extract_f0_feature.log" % self.exp_dir, "a+") as f:
            f.write("%s\n" % strr)
            f.flush()

    def compute_feats(self,x):
        feats = torch.from_numpy(x).float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)

        inputs = {
            "source": feats.half().to(self.device) 
                if self.device not in ["mps", "cpu"]
                else feats.to(self.device),
            "padding_mask": padding_mask.to(self.device),
            "output_layer": 9 if self.version == "v1" else 12,  # layer 9
        }
        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
            feats = (
                self.model.final_proj(logits[0]) if self.version == "v1" else logits[0]
            )

        feats = feats.squeeze(0).float().cpu().numpy()
        if np.isnan(feats).sum() == 0:
            return feats
        else:
            return self.printt("==contains nan==")

    def compute_f0(self,x):
        p_len = x.shape[0] // self.hop
        
        return self.get_f0(x,p_len,0,self.f0_method,crepe_hop_length=self.hop)
    
    def go(self, paths):
        if len(paths) == 0:
            self.printt("no-f0-todo")
        else:
            self.printt("todo-f0-%s" % len(paths))
            # n = max(len(paths) // 5, 1)  # 每个进程最多打印5条
            for idx, (inp_path, opt_path1, opt_path2, opt_path3) in enumerate(paths):
                try:
                    # if idx % n == 0:
                    #     self.printt("f0ing,now-%s,all-%s,-%s" % (idx, len(paths), inp_path))
                    if (
                        os.path.exists(opt_path1 + ".npy") == True
                        and os.path.exists(opt_path2 + ".npy") == True
                        and os.path.exists(opt_path3 + ".npy") == True
                    ):
                        continue
                    x,_ = load_input_audio(inp_path,self.sr)
                    if self.model:
                        feats = self.compute_feats(x)
                        if feats is not None:
                            np.save(
                                opt_path3,
                                feats,
                                allow_pickle=False,
                            )  # features
                            if self.if_f0: # uses pitch
                                coarse_pit, featur_pit = self.compute_f0(x)
                                np.save(
                                    opt_path2,
                                    featur_pit,
                                    allow_pickle=False,
                                )  # nsf
                                np.save(
                                    opt_path1,
                                    coarse_pit,
                                    allow_pickle=False,
                                )  # ori
                except:
                    self.printt("f0fail-%s-%s-%s" % (idx, inp_path, traceback.format_exc()))

def preprocess_trainset(inp_root, sr, n_p, exp_dir):
    try:
        pp = Preprocess(sr, exp_dir)
        pp.println("start preprocess")
        pp.println(sys.argv)
        pp.pipeline_mp_inp_dir(inp_root, n_p)
        pp.println("end preprocess")
        del pp
        gc_collect()
        return "Successfully preprocessed data"
    except Exception as e:
        return f"Failed to preprocess data: {e}"

def extract_features_trainset(exp_dir,n_p,f0method,device,version,if_f0):
    try:
        if type(f0method)!=list: f0method=[f0method] # make sure f0method is a list
        featureInput = FeatureInput(f0_method=f0method,exp_dir=exp_dir,device=device,if_f0=if_f0)
        paths = []
        inp_root = os.path.join(exp_dir,"1_16k_wavs")
        opt_root1 = os.path.join(exp_dir,"2a_f0")
        opt_root2 = os.path.join(exp_dir,"2b-f0nsf")
        opt_root3 = os.path.join(exp_dir,"3_feature256" if version == "v1" else "3_feature768")

        os.makedirs(opt_root1, exist_ok=True)
        os.makedirs(opt_root2, exist_ok=True)
        os.makedirs(opt_root3, exist_ok=True)

        for name in sorted(list(os.listdir(inp_root))):
            inp_path = os.path.join(inp_root, name)
            if "spec" in inp_path:
                continue
            for method in f0method:
                opt_path1 = os.path.join(opt_root1, ",".join([str(method),name]))
                opt_path2 = os.path.join(opt_root2, ",".join([str(method),name])) 
                opt_path3 = os.path.join(opt_root3, ",".join([str(method),name]))
                paths.append([inp_path, opt_path1, opt_path2, opt_path3])

        ps = []
        n_p = max(n_p,1)
        for i in range(n_p):
            if device=="cuda":
                featureInput.go(paths[i::n_p])
            else:
                # p = multiprocessing.Process(
                #     target=featureInput.go,
                #     args=[paths[i::n_p]],
                # )
                p = Thread(target=featureInput.go,args=(paths[i::n_p],),daemon=True)
                ps.append(p)
                p.start()

        if device != "cuda":
            for p in ps:
                try:
                    p.join()
                except:
                    featureInput.printt("f0_all_fail-%s" % (traceback.format_exc()))

        return f"Successfully extracted features using {f0method}"
    except Exception as e:
        return f"Failed to extract features: {e}"
