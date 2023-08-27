import os
from random import shuffle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import streamlit as st
from types import SimpleNamespace
import subprocess
import faiss
import torch
from preprocessing_utils import extract_features_trainset, preprocess_trainset

from webui_utils import gc_collect, get_filenames, get_index, load_config

config, i18n = load_config()
CWD = os.getcwd()

DEVICE_OPTIONS = ["cpu","cuda"]
PITCH_EXTRACTION_OPTIONS = ["harvest","crepe","rmvpe"]

def preprocess_data(exp_dir, sr, trainset_dir, n_threads, version):
    model_log_dir = f"{CWD}/logs/{exp_dir}_{version}_{sr}"
    os.makedirs(model_log_dir, exist_ok=True)
    return preprocess_trainset(trainset_dir,SR_MAP[sr],n_threads,model_log_dir)

def extract_features(exp_dir, n_threads, version, if_f0, f0method,device,sr):
    model_log_dir = f"{CWD}/logs/{exp_dir}_{version}_{sr}"
    os.makedirs(model_log_dir, exist_ok=True)
    
    # if if_f0: #pitch extraction
    n_p = n_threads if device=="cpu" else torch.cuda.device_count()
    return extract_features_trainset(model_log_dir,n_p=n_p,f0method=f0method,device=device,if_f0=if_f0,version=version)

def train_model(exp_dir,if_f0,spk_id,version,sr,gpus,batch_size,total_epoch,save_epoch,pretrained_G,pretrained_D,if_save_latest,if_cache_gpu,if_save_every_weights):
    print(i18n("training.train_model"))
    model_log_dir = f"{CWD}/logs/{exp_dir}_{version}_{sr}"
    gt_wavs_dir = os.sep.join([model_log_dir,"0_gt_wavs"])
    feature_dir = os.sep.join([model_log_dir,"3_feature256" if version == "v1" else "3_feature768"])
    os.makedirs(gt_wavs_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    
    if if_f0:
        f0_dir =  os.sep.join([model_log_dir,"2a_f0"])
        f0nsf_dir = os.sep.join([model_log_dir,"2b-f0nsf"])
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id,
                )
            )
    fea_dim = 256 if version == "v1" else 768
    if if_f0:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (CWD, sr, CWD, fea_dim, CWD, CWD, spk_id)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (CWD, sr, CWD, fea_dim, spk_id)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % model_log_dir, "w") as f:
        f.write("\n".join(opt))
    print("write filelist done")

    
    cmd = " ".join(str(i) for i in [
        config.python_cmd,
        'training_cli.py -e', f"{exp_dir}_{version}_{sr}",
        "-n", exp_dir,
        "-sr", sr,
        "-f0", 1 if if_f0 else 0,
        "-bs", batch_size,
        f"-g {gpus}" if gpus else "",
        "-te", total_epoch,
        "-se", save_epoch,
        f"-pg {pretrained_G}" if pretrained_G and gpus else "",
        f"-pd {pretrained_D}" if pretrained_D and gpus else "",
        "-l", 1 if if_save_latest else 0,
        "-c", 1 if if_cache_gpu else 0,
        "-sw", 1 if if_save_every_weights else 0,
        "-v", version
    ])
    
    p = subprocess.Popen(cmd, shell=True, cwd=CWD)

    return p

def train_index(exp_dir,version,sr):
    model_log_dir = f"{CWD}/logs/{exp_dir}_{version}_{sr}"
    feature_dir = os.sep.join([model_log_dir,"3_feature256" if version == "v1" else "3_feature768"])
    os.makedirs(feature_dir, exist_ok=True)

    npys = []
    listdir_res = list(os.listdir(feature_dir))
    for name in sorted(listdir_res):
        phone = np.load(os.sep.join([feature_dir, name]))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    if big_npy.shape[0] > 2e5:
        # if(1):
        info = "Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0]
        print(info)
        print(info)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception as e:
            print(e)

    np.save("%s/total_fea.npy" % model_log_dir, big_npy)

    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    print("%s,%s" % (big_npy.shape, n_ivf))
    index = faiss.index_factory(256 if version == "v1" else 768, "IVF%s,Flat" % n_ivf)
    print("training index")
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(index,os.sep.join([model_log_dir,f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir}_{version}.index"]))
    print("adding index")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(index,os.sep.join([model_log_dir,f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir}_{version}.index"]))
       
    print("added_IVF%s_Flat_nprobe_%s_%s_%s.index")

def one_click_train(): #TODO not implemented yet

    #########step1:处理数据
    preprocess_data(**st.session_state.training.preprocess_params)

    #########step2a:提取音高
    extract_features(**st.session_state.training.feature_params)
    
    #######step3a:训练模型
    print(i18n("step3a:正在训练模型"))
    train_model(**st.session_state.training.train_params)

    #######step3b:训练索引
    print(i18n("step3b:训练索引"))
    train_index(**st.session_state.training.index_params)
    
    print(i18n("全流程结束！"))

def get_active_pids(pids):
    active_pids = []
    for prc in pids:
        try:
            if prc.returncode is not None: print(f"process {prc} returned with {prc.returncode}")
            else: active_pids.append(prc)
        except Exception as e:
            print(e)
    return active_pids

def kill_all_process(pids):
    for pid in get_active_pids(pids):
        pid.kill()
    gc_collect(pids)
    st.experimental_rerun()

N_THREADS_OPTIONS=[1,2,4,8,12,16]
PITCH_EXTRACTION_OPTIONS = ["crepe","rmvpe"]
SR_MAP = {"40k": 40000, "48k": 48000}
PRETRAINED_G = get_filenames(root="models",folder="pretrained_v2",name_filters=[f"G{st.session_state.training.sr}"])
PRETRAINED_D = get_filenames(root="models",folder="pretrained_v2",name_filters=[f"D{st.session_state.training.sr}"])

def render(state):
    
    with st.container():
        col1,col2 = st.columns(2)
        state.exp_dir = col1.text_input(i18n("training.exp_dir"),value=state.exp_dir) #model_name
        state.n_threads=col2.selectbox(i18n("training.n_threads"),
                                    options=N_THREADS_OPTIONS,
                                    index=get_index(N_THREADS_OPTIONS,state.n_threads))

        col1,col2,col3 = st.columns(3)
        state.sr=col1.radio(i18n("training.sr"),
                        options=["40k","48k"],
                        index=get_index(["40k","48k"],state.sr),
                        horizontal=True)
        state.version=col2.radio(i18n("training.version"),options=["v1","v2"],horizontal=True,index=get_index(["v1","v2"],state.version))
        state.device=col3.radio(i18n("training.device"),options=["cuda","cpu"],horizontal=True)

        #preprocess_data(exp_dir, sr, trainset_dir, n_threads)
        st.subheader(i18n("training.preprocess_data.title"))
        st.write(i18n("training.preprocess_data.text"))
        state.trainset_dir=st.text_input(i18n("training.preprocess_data.trainset_dir"))
        if st.button(i18n("training.preprocess_data.submit"),disabled=not (state.trainset_dir and state.exp_dir)):
            preprocess_data(state.exp_dir, state.sr, state.trainset_dir, state.n_threads, state.version)

    with st.form(i18n("training.extract_features.form")):  #extract_features(exp_dir, n_threads, version, if_f0, f0method)
        st.subheader(i18n("training.extract_features.title"))
        st.write(i18n("training.extract_features.text"))
        col1,col2 = st.columns(2)
        state.if_f0=col1.checkbox(i18n("training.if_f0"),value=state.if_f0)
        state.f0method=col2.radio(i18n("training.f0method"),options=PITCH_EXTRACTION_OPTIONS,
                                  horizontal=True,index=get_index(PITCH_EXTRACTION_OPTIONS,state.f0method))

        if st.form_submit_button(i18n("training.extract_features.submit"),disabled=not state.exp_dir):
            state.pids.extend(extract_features(state.exp_dir, state.n_threads, state.version, state.if_f0, state.f0method, state.device,state.sr))
        
    with st.form(i18n("training.train_model.form")):  #def train_model(exp_dir,if_f0,spk_id,version,sr,gpus,batch_size,total_epoch,save_epoch,pretrained_G,pretrained_D,if_save_latest,if_cache_gpu,if_save_every_weights):
        st.subheader(i18n("training.train_model.title"))
        st.write(i18n("training.train_model.text"))
        state.gpus=st.multiselect(i18n("training.gpus"),options=np.arange(torch.cuda.device_count(),dtype=str))
        col1,col2,col3=st.columns(3)
        state.batch_size=col1.slider(i18n("training.batch_size"),min_value=1,max_value=100,step=1,value=state.batch_size)
        state.total_epoch=col2.slider(i18n("training.total_epoch"),min_value=0,max_value=1000,step=10,value=state.total_epoch)
        state.save_epoch=col3.slider(i18n("training.save_epoch"),min_value=0,max_value=100,step=10,value=state.save_epoch)
        state.pretrained_G=st.selectbox(i18n("training.pretrained_G"),options=PRETRAINED_G)
        state.pretrained_D=st.selectbox(i18n("training.pretrained_D"),options=PRETRAINED_D)
        state.if_save_latest=st.checkbox(i18n("training.if_save_latest"),value=state.if_save_latest)
        state.if_cache_gpu=st.checkbox(i18n("training.if_cache_gpu"),value=state.if_cache_gpu)
        state.if_save_every_weights=st.checkbox(i18n("training.if_save_every_weights"),value=state.if_save_every_weights)
        
        if st.form_submit_button(i18n("training.train_model.submit"),disabled=not state.exp_dir):
            state.pids.append(train_model(state.exp_dir, state.if_f0, state.spk_id, state.version,state.sr,
                                          "-".join(state.gpus),state.batch_size,state.total_epoch,state.save_epoch,
                                          state.pretrained_G,state.pretrained_D,state.if_save_latest,state.if_cache_gpu,
                                          state.if_save_every_weights))

    if state.exp_dir and state.version and st.button(i18n("training.train_index.submit")):
        train_index(state.exp_dir,state.version,state.sr)
    
    if state.pids is not None and len(state.pids):
        state.pids = get_active_pids(state.pids)

        st.subheader(i18n("training.pids"))
        st.write(state.pids)

        col1,col2,col3=st.columns(3)
        for prc in state.pids:
            col1.write(prc.pid)
            col2.write(prc.returncode)
            if col3.button(i18n("training.kill_one_pid"),key=f"training.kill_one_pid.{prc.pid}"):
                prc.kill()
                prc.returncode="killed"
                gc_collect(prc)

        if st.button(i18n("training.kill_all_pids"),type="primary",use_container_width=True):
            kill_all_process(state.pids)

    return state

@st.cache_data
def init_training_state():
    state = SimpleNamespace(
        exp_dir="",
        sr="40k",
        if_f0=True,
        trainset_dir="",
        spk_id=0,
        f0method="rmvpe",
        save_epoch=0,
        total_epoch=100,
        batch_size=4,
        n_threads=os.cpu_count(),
        if_save_latest=True,
        pretrained_G=None,
        pretrained_D=None,
        gpus=[0],
        if_cache_gpu=False,
        if_save_every_weights=False,
        version="v2",
        pids=[],
        device="cuda")
    return state

def init_state():
    st.session_state["training"] = st.session_state.get("training",init_training_state())

init_state()

if __name__=="__main__": st.session_state.training=render(st.session_state.training)