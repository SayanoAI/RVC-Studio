import os
from random import shuffle
import sys
from time import sleep
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import streamlit as st
from webui import MENU_ITEMS, N_THREADS_OPTIONS, PITCH_EXTRACTION_OPTIONS, SR_MAP, config, i18n
st.set_page_config(layout="centered",menu_items=MENU_ITEMS)

from webui.components import active_subprocess_list, file_uploader_form
from webui.downloader import BASE_MODELS_DIR, DATASETS_DIR
from tts_cli import EMBEDDING_CHECKPOINT, TTS_MODELS_DIR

from webui.audio import load_input_audio, save_input_audio



from webui.utils import ObjectNamespace
import subprocess
import faiss
import torch
from preprocessing_utils import extract_features_trainset, preprocess_trainset
from webui.contexts import ProgressBarContext, SessionStateContext

from webui.utils import get_filenames, get_index

CWD = os.getcwd()
if CWD not in sys.path:
    sys.path.append(CWD)

def preprocess_data(exp_dir, sr, trainset_dir, n_threads, version):
    model_log_dir = os.path.join(CWD,"logs",f"{exp_dir}_{version}_{sr}")
    os.makedirs(model_log_dir, exist_ok=True)
    return preprocess_trainset(trainset_dir,SR_MAP[sr],n_threads,model_log_dir)

def extract_features(exp_dir, n_threads, version, if_f0, f0method,device, sr):
    model_log_dir = os.path.join(CWD,"logs",f"{exp_dir}_{version}_{sr}")
    os.makedirs(model_log_dir, exist_ok=True)
    
    # if if_f0: #pitch extraction
    # n_p = n_threads if device=="cpu" else torch.cuda.device_count()
    n_p = max(n_threads // (len(f0method) if type(f0method)==list else os.cpu_count()),1)

    if type(f0method)==list:
        return "\n".join([
            extract_features_trainset(model_log_dir,n_p=n_p,f0method=method,device=device,if_f0=if_f0,version=version)
            for method in f0method
        ])
    return extract_features_trainset(model_log_dir,n_p=n_p,f0method=f0method,device=device,if_f0=if_f0,version=version)

def create_filelist(exp_dir,if_f0,spk_id,version,sr):
    model_log_dir = os.path.join(CWD,"logs",f"{exp_dir}_{version}_{sr}")

    print(i18n("training.create_filelist"))
    gt_wavs_dir = os.sep.join([model_log_dir,"0_gt_wavs"])
    feature_dir = os.sep.join([model_log_dir,"3_feature256" if version == "v1" else "3_feature768"])
    os.makedirs(gt_wavs_dir, exist_ok=True)
    os.makedirs(feature_dir, exist_ok=True)
    
    # add training data 
    if if_f0:
        f0_dir =  os.sep.join([model_log_dir,"2a_f0"])
        f0nsf_dir = os.sep.join([model_log_dir,"2b-f0nsf"])
        names = (
            set([os.path.splitext(name)[0] for name in os.listdir(feature_dir)])
            & set([os.path.splitext(name)[0] for name in os.listdir(f0_dir)])
            & set([os.path.splitext(name)[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set(
            [os.path.splitext(name)[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    missing_data = []
    for name in names:
        name_parts = name.split(",")
        gt_name = name if len(name_parts) == 1 else name_parts[-1]
        gt_file = os.path.join(gt_wavs_dir,gt_name)
        if not os.path.isfile(gt_file):
            print(f"{gt_name} not found!")
            missing_data.append(gt_name)
            continue #skip data

        if if_f0:
            data = "|".join([
                gt_file,
                os.path.join(feature_dir,f"{name}.npy"),
                os.path.join(f0_dir,f"{name}.npy"),
                os.path.join(f0nsf_dir,f"{name}.npy"),
                str(spk_id)
            ])
        else:
            data = "|".join([
                gt_file,
                os.path.join(feature_dir,f"{name}.npy"),
                str(spk_id)
            ])
        opt.append(data)

    # add mute data 
    fea_dim = 256 if version == "v1" else 768
    if if_f0:
        data = "|".join([
            os.path.join(CWD,"logs","mute","0_gt_wavs",f"mute{sr}.wav"),
            os.path.join(CWD,"logs","mute",f"3_feature{fea_dim}","mute.npy"),
            os.path.join(CWD,"logs","mute","2a_f0","mute.wav.npy"),
            os.path.join(CWD,"logs","mute","2b-f0nsf","mute.wav.npy"),
            str(spk_id)
        ])
    else:
        data = "|".join([
            os.path.join(CWD,"logs","mute","0_gt_wavs",f"mute{sr}.wav"),
            os.path.join(CWD,"logs","mute",f"3_feature{fea_dim}","mute.npy"),
            str(spk_id)
        ])
    opt.append(data)

    shuffle(opt)
    if len(opt)>=len(os.listdir(gt_wavs_dir)): # has gt data
        with open(os.path.join(model_log_dir, "filelist.txt"), "w") as f:
            f.write("\n".join(opt))
        print("write filelist done")
        return True
    else:
        raise Exception(f"missing ground truth data: {missing_data}")

def train_model(exp_dir,if_f0,spk_id,version,sr,gpus,batch_size,total_epoch,save_epoch,pretrained_G,pretrained_D,if_save_latest,if_cache_gpu,if_save_every_weights):
    try:
        print(i18n("training.train_model"))
        create_filelist(exp_dir,if_f0,spk_id,version,sr)
        
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
        
        subprocess.Popen(cmd, shell=True, cwd=CWD, stderr=subprocess.PIPE)

        return f"Successfully started training with {cmd}. View your process under Active Processes."
    except Exception as e:
        return f"Failed to initiate training: {e}"

def train_index(exp_dir,version,sr):
    try:
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

        index_name = os.path.join(BASE_MODELS_DIR,"RVC",".index",f"{exp_dir}_{version}_{sr}.index")
        faiss.write_index(index,index_name)
        
        return f"saved index file to {index_name}"
    except Exception as e:
        return f"Failed to train index: {e}"

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

def train_speaker_embedding(exp_dir: str, model_log_dir: str):
    try:
        # get dataset
        training_file = os.path.join(CWD,"logs",model_log_dir,"embedding.wav")
        if os.path.isfile(training_file): audio = load_input_audio(training_file,sr=16000,mono=True)[0]
        else:
            dataset_dir = os.path.join(CWD,"logs",model_log_dir,"1_16k_wavs")
            audio = np.concatenate([
                load_input_audio(os.path.join(dataset_dir,fname),sr=16000,mono=True)[0]
                for fname in os.listdir(dataset_dir)],axis=None)
            save_input_audio(training_file,(audio,16000))

        # train embedding
        from speechbrain.pretrained import EncoderClassifier
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(TTS_MODELS_DIR,EMBEDDING_CHECKPOINT)
        classifier = EncoderClassifier.from_hparams(source=EMBEDDING_CHECKPOINT, savedir=os.path.join(TTS_MODELS_DIR,EMBEDDING_CHECKPOINT))
        embeddings = classifier.encode_batch(torch.from_numpy(audio),normalize=True).squeeze(0)

        # save embedding file
        embedding_path = os.path.join(TTS_MODELS_DIR,"embeddings",f"{exp_dir}.npy")
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        np.save(embedding_path,embeddings.numpy())
        return f"Saved speaker embedding to: {embedding_path}"
    except Exception as e:
        return f"Failed to train speecht5 speaker embedding: {e}"

def init_training_state():
    return ObjectNamespace(
        exp_dir="",
        sr="40k",
        if_f0=True,
        trainset_dir="",
        spk_id=0,
        f0method=["rmvpe"],
        save_epoch=0,
        total_epoch=100,
        batch_size=4,
        n_threads=os.cpu_count(),
        if_save_latest=True,
        pretrained_G=None,
        pretrained_D=None,
        gpus=[],
        if_cache_gpu=False,
        if_save_every_weights=False,
        version="v2",
        pids=[],
        device="cuda")

if __name__=="__main__":
    with SessionStateContext("training",init_training_state()) as state:
        
        with st.container():
            col1,col2 = st.columns(2)
            state.exp_dir = col1.text_input(i18n("training.exp_dir"),value=state.exp_dir,placeholder="Sayano") #model_name
            state.n_threads=col2.selectbox(i18n("training.n_threads"),
                                        options=N_THREADS_OPTIONS,
                                        index=get_index(N_THREADS_OPTIONS,state.n_threads))

            col1,col2,col3 = st.columns(3)
            SR_OPTIONS = list(SR_MAP.keys())
            state.sr=col1.radio(i18n("training.sr"),
                            options=SR_OPTIONS,
                            index=get_index(SR_OPTIONS,state.sr),
                            horizontal=True)
            state.version=col2.radio(i18n("training.version"),options=["v1","v2"],horizontal=True,index=get_index(["v1","v2"],state.version))
            state.device=col3.radio(i18n("training.device"),options=["cuda","cpu"],horizontal=True)

            #preprocess_data(exp_dir, sr, trainset_dir, n_threads)
            st.subheader(i18n("training.preprocess_data.title"))
            st.write(i18n("training.preprocess_data.text"))
            file_uploader_form(DATASETS_DIR,"Upload a zipped folder of your dataset (make sure the files are in a folder)",types="zip")
            state.trainset_dir=st.text_input(i18n("training.preprocess_data.trainset_dir"),placeholder="./datasets/name_of_zipped_folder")

            disabled = not (state.trainset_dir and state.exp_dir and os.path.exists(state.trainset_dir))
            if st.button(i18n("training.preprocess_data.submit"),disabled=disabled):
                st.toast(preprocess_data(state.exp_dir, state.sr, state.trainset_dir, state.n_threads, state.version))

        PRETRAINED_G = get_filenames(root="models",folder="pretrained_v2",name_filters=[f"{'f0' if state.if_f0 else ''}G{state.sr}"])
        PRETRAINED_D = get_filenames(root="models",folder="pretrained_v2",name_filters=[f"{'f0' if state.if_f0 else ''}D{state.sr}"])
        model_log_dir = f"{state.exp_dir}_{state.version}_{state.sr}"

        with st.form(i18n("training.extract_features.form")):  #extract_features(exp_dir, n_threads, version, if_f0, f0method)
            st.subheader(i18n("training.extract_features.title"))
            st.write(i18n("training.extract_features.text"))
            col1,col2 = st.columns(2)
            state.if_f0=col1.checkbox(i18n("training.if_f0"),value=state.if_f0)
            state.f0method=col2.multiselect(i18n("training.f0method"),options=PITCH_EXTRACTION_OPTIONS,default=state.f0method)
            disabled = not (state.exp_dir and os.path.exists(os.path.join(CWD,"logs",model_log_dir,"1_16k_wavs")))
            if st.form_submit_button(i18n("training.extract_features.submit"),disabled=disabled):
                st.toast(extract_features(state.exp_dir, state.n_threads, state.version, state.if_f0, state.f0method, state.device,state.sr))
            
        with st.form(i18n("training.train_model.form")):  #def train_model(exp_dir,if_f0,spk_id,version,sr,gpus,batch_size,total_epoch,save_epoch,pretrained_G,pretrained_D,if_save_latest,if_cache_gpu,if_save_every_weights):
            st.subheader(i18n("training.train_model.title"))
            st.write(i18n("training.train_model.text"))
            state.gpus=st.multiselect(i18n("training.gpus"),options=[str(i) for i in range(torch.cuda.device_count())],default=state.gpus)
            col1,col2,col3=st.columns(3)
            state.batch_size=col1.slider(i18n("training.batch_size"),min_value=1,max_value=100,step=1,value=state.batch_size)
            state.total_epoch=col2.slider(i18n("training.total_epoch"),min_value=0,max_value=1000,step=10,value=state.total_epoch)
            state.save_epoch=col3.slider(i18n("training.save_epoch"),min_value=0,max_value=100,step=1,value=state.save_epoch)
            state.pretrained_G=st.selectbox(i18n("training.pretrained_G"),options=PRETRAINED_G)
            state.pretrained_D=st.selectbox(i18n("training.pretrained_D"),options=PRETRAINED_D)
            state.if_save_latest=st.checkbox(i18n("training.if_save_latest"),value=state.if_save_latest)
            state.if_cache_gpu=st.checkbox(i18n("training.if_cache_gpu"),value=state.if_cache_gpu)
            state.if_save_every_weights=st.checkbox(i18n("training.if_save_every_weights"),value=state.if_save_every_weights)
            
            disabled = not (state.exp_dir and os.path.exists(os.path.join(CWD,"logs",model_log_dir,"3_feature768")))
            if st.form_submit_button(i18n("training.train_model.submit"),disabled=disabled):
                if not (state.pretrained_D and state.pretrained_G): st.toast("Please download the pretrained models!")
                else: 
                    with ProgressBarContext([1]*100,sleep,"Waiting for training process to spawn (you should hear your GPU fans spinning). Uncheck GPU cache if you have a large dataset.") as pb:
                        st.toast(train_model(state.exp_dir, state.if_f0, state.spk_id, state.version,state.sr,
                                                "-".join(state.gpus),state.batch_size,state.total_epoch,state.save_epoch,
                                                state.pretrained_G,state.pretrained_D,state.if_save_latest,state.if_cache_gpu,
                                                state.if_save_every_weights))
                        pb.run()
                        st.experimental_rerun()

        disabled = not (state.exp_dir and os.path.exists(os.path.join(CWD,"logs",model_log_dir,"3_feature256" if state.version == "v1" else "3_feature768")))
        if state.exp_dir and state.version and st.button(i18n("training.train_index.submit"),disabled=disabled):
            st.toast(train_index(state.exp_dir,state.version,state.sr))

        if state.exp_dir:
            disabled = not (state.exp_dir and os.path.exists(os.path.join(CWD,"logs",model_log_dir)))
            if st.button(i18n("training.train_speaker.submit"),disabled=disabled):
                st.toast(train_speaker_embedding(state.exp_dir,model_log_dir))
            else: st.markdown(f"*Only required for speecht5 TTS*")

        active_subprocess_list()
