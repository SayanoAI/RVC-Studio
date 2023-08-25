import os
import streamlit as st
from types import SimpleNamespace

from webui_utils import gc_collect, get_filenames, get_index, get_vc, load_config, merge_audio, save_input_audio, vc_single
from uvr5_cli import split_audio

config, i18n = load_config()

@st.cache_data
def split_vocals(model_paths,**args):
    vocals,instrumental,input_audio=split_audio(model_paths,**args)
    st.session_state.inference.input_audio=input_audio
    st.session_state.inference.input_vocals=vocals
    st.session_state.inference.input_instrumental=instrumental
    st.session_state.inference.output_vocals = None
    st.session_state.inference.output_audio = None

    gc_collect()

    return vocals, instrumental

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    clear_data()

    index_file = get_filenames(root="./models/weights",exts=["index"],name_filters=[os.path.basename(model_path).split(".")[0]])
    if st.session_state.inference.vc and st.session_state.inference.cpt and st.session_state.inference.net_g and st.session_state.inference.hubert_model:
        return {
            "vc": st.session_state.inference.vc,
            "cpt": st.session_state.inference.cpt,
            "net_g": st.session_state.inference.net_g,
            "file_index": index_file[0] if len(index_file) else "",
            "hubert_model": st.session_state.inference.hubert_model
        }
    else:
        data = get_vc(model_path,device=st.session_state.inference.device)
        st.session_state.inference.vc = data["vc"]
        st.session_state.inference.cpt = data["cpt"]
        st.session_state.inference.net_g = data["net_g"]
        st.session_state.inference.hubert_model = data["hubert_model"]
        data["file_index"] = index_file[0] if len(index_file) else ""
        return data

@st.cache_data(max_entries=10)
def convert_vocals(model_name,input_audio,**kwargs):
    print(f"converting vocals... {model_name} - {kwargs}")
    data=load_model(model_name)
    st.session_state.inference.convert_params = SimpleNamespace(**kwargs)
    return vc_single(input_audio=input_audio,**data,**kwargs)

def get_models(folder="."):
    fnames = get_filenames(root="./models",folder=folder,exts=["pth","pt"])
    return fnames

@st.cache_data
def init_inference_state():
    state = SimpleNamespace(
        sid=None,
        cpt=None,
        vc=None,
        net_g=None,
        device="cuda" if config.has_gpu else "cpu",
        models=get_models(folder="weights"),
        model_name=None,
        uvr5_models=get_filenames(root="./models/uvr5_weights",name_filters=["vocal","instrument"]),
        preprocess_models=[""]+get_filenames(root="./models/uvr5_weights",name_filters=["echo","reverb","noise"]),
        preprocess_model="",
        agg=10,
        dereverb=False,
        uvr5_name=[],
        use_cache=True,
        hubert_model=None,
        audio_files=get_filenames(exts=["wav","flac","ogg","mp3"],folder="songs"),
        input_audio_name=None,
        input_audio=None,
        input_vocals=None,
        input_instrumental=None,
        output_audio=None,
        output_audio_name=None,
        output_vocals=None,
        convert_params=SimpleNamespace(
            f0_up_key=0,
            f0_method="rmvpe",
            index_rate=.75,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=.2,
            protect=0.2,
        )
    )
    return state

def refresh_data():
    st.session_state.inference.uvr5_models = get_filenames(root="./models/uvr5_weights",name_filters=["vocal","instrument"])
    st.session_state.inference.preprocess_models = [""]+get_filenames(root="./models/uvr5_weights",name_filters=["echo","reverb","noise"])
    st.session_state.inference.models = get_models(folder="weights")
    st.session_state.inference.audio_files = get_filenames(exts=["wav","flac","ogg","mp3"],name_filters=[""],folder="songs")
    gc_collect()
    
def clear_data():
    vc = st.session_state.inference.vc
    cpt = st.session_state.inference.cpt 
    net_g = st.session_state.inference.net_g
    huber_model = st.session_state.inference.hubert_model
    
    del vc, cpt, net_g, huber_model

    st.session_state.inference.vc = None
    st.session_state.inference.cpt = None
    st.session_state.inference.net_g = None
    st.session_state.inference.hubert_model = None

    gc_collect()

DEVICE_OPTIONS = ["cpu","cuda"]
PITCH_EXTRACTION_OPTIONS = ["crepe","rmvpe"]

@st.cache_data
def get_filename(audio_name,model_name):
    song = os.path.basename(audio_name).split(".")[0]
    singer = os.path.basename(model_name).split(".")[0]
    return f"{singer}.{song}"

def one_click_convert():
    clear_data()
    vocals, instrumental = split_vocals(
        st.session_state.inference.uvr5_name,
        audio_path=st.session_state.inference.input_audio_name,
        preprocess_model=st.session_state.inference.preprocess_model,
        device=st.session_state.inference.device,
        agg=st.session_state.inference.agg,
        use_cache=st.session_state.inference.use_cache
        )
    params = vars(st.session_state.inference.convert_params)
    
    changed_vocals = convert_vocals(
        st.session_state.inference.model_name,
        vocals,
        **params)
    st.session_state.inference.output_vocals = changed_vocals
    mixed_audio = merge_audio(changed_vocals,instrumental,sr=st.session_state.inference.input_audio[1])
    st.session_state.inference.output_audio_name = get_filename(
        st.session_state.inference.input_audio_name,st.session_state.inference.model_name)
    st.session_state.inference.output_audio = mixed_audio

def download_song(output_audio,output_audio_name,ext="mp3"):
    output_dir = os.sep.join([os.getcwd(),"output"])
    os.makedirs(output_dir,exist_ok=True)
    output_file = os.sep.join([output_dir,f"{output_audio_name}.{ext}"])
    return f"saved to {output_file}.{ext}: {save_input_audio(output_file,output_audio)}"
    
def render(state):
    with st.container():
        left, right = st.columns(2)
        state.input_audio_name = left.selectbox(
            i18n("inference.song.selectbox"),
            options=state.audio_files,
            index=get_index(state.audio_files,state.input_audio_name))
        col1, col2 = left.columns(2)
        if col1.button(i18n("inference.refresh_data.button"),on_click=refresh_data,use_container_width=True): st.experimental_rerun()
        col2.button(i18n("inference.one_click.button"), type="primary",
                    use_container_width=True,
                    disabled=not (state.uvr5_name and state.input_audio_name and state.model_name),
                    on_click=one_click_convert)
        
        state.model_name = right.selectbox(
            i18n("inference.voice.selectbox"),
            options=state.models,
            index=get_index(state.models,state.model_name)
            )
        if right.button(i18n("inference.clear_data.button"),on_click=clear_data): st.experimental_rerun()
        

    st.subheader(i18n("inference.split_vocals"))
    with st.expander(i18n("inference.split_vocals.expander"),expanded=not (state.input_audio_name and len(state.uvr5_name))):
        with st.form("inference.split_vocals.expander"):
            preprocess_model = st.selectbox(
                i18n("inference.preprocess_model"),
                options=state.preprocess_models,
                index=get_index(state.preprocess_models,state.preprocess_model))
            uvr5_name = st.multiselect(
                i18n("inference.uvr5_name"),
                options=state.uvr5_models,
                default=state.uvr5_name)
            
            col1, col2 = st.columns(2)
            device = col1.radio(
                i18n("inference.device"),
                disabled=not config.has_gpu,
                options=DEVICE_OPTIONS,horizontal=True,
                index=get_index(DEVICE_OPTIONS,state.device))
            agg = col2.slider(i18n("inference.agg"),min_value=0,max_value=20,step=1,value=state.agg)

            use_cache=col1.checkbox(i18n("inference.use_cache"),value=state.use_cache)
            # state.dereverb=col1.checkbox(i18n("inference.dereverb"),value=state.dereverb)
            if st.form_submit_button(i18n("inference.save.button")):
                state.agg=agg
                state.use_cache=use_cache
                state.device=device
                state.preprocess_model=preprocess_model
                state.uvr5_name=uvr5_name

    if st.button(i18n("inference.split_vocals"),disabled=not (state.input_audio_name and len(state.uvr5_name))):
        split_vocals(
            state.uvr5_name,
            audio_path=state.input_audio_name,
            preprocess_model=state.preprocess_model,
            device=state.device,
            agg=state.agg,
            use_cache=state.use_cache
            )
            
    with st.container():
        
        if state.input_audio is not None:
            st.write("Input Audio")
            st.audio(state.input_audio[0],sample_rate=state.input_audio[1])
        
        col1, col2 = st.columns(2)

        if state.input_vocals is not None:
            col1.write("Vocals")
            col1.audio(state.input_vocals[0],sample_rate=state.input_vocals[1])
        
        if state.input_instrumental is not None:
            col2.write("Instrumental")
            col2.audio(state.input_instrumental[0],sample_rate=state.input_instrumental[1])
    
    st.subheader(i18n("inference.convert_vocals"))
    with st.expander(i18n("inference.convert_vocals.expander")):
        with st.form("inference.convert_vocals.expander"):
            
            f0_up_key = st.slider(i18n("inference.f0_up_key"),min_value=-12,max_value=12,step=6,value=state.convert_params.f0_up_key)
            f0_method = st.selectbox(i18n("inference.f0_method"),
                                                options=PITCH_EXTRACTION_OPTIONS,
                                                index=get_index(PITCH_EXTRACTION_OPTIONS,state.convert_params.f0_method))
            resample_sr = st.select_slider(i18n("inference.resample_sr"),
                                                options=[0,16000,24000,22050,40000,44100,48000],
                                                value=state.convert_params.resample_sr)
            index_rate=st.slider(i18n("inference.index_rate"),min_value=0.,max_value=1.,step=.05,value=state.convert_params.index_rate)
            filter_radius=st.slider(i18n("inference.filter_radius"),min_value=0,max_value=7,step=1,value=state.convert_params.filter_radius)
            rms_mix_rate=st.slider(i18n("inference.rms_mix_rate"),min_value=0.,max_value=1.,step=.05,value=state.convert_params.rms_mix_rate)
            protect=st.slider(i18n("inference.protect"),min_value=0.,max_value=.5,step=.01,value=state.convert_params.protect)
            
            if st.form_submit_button(i18n("inference.save.button")):
                state.convert_params = SimpleNamespace(
                    f0_up_key=f0_up_key,
                    f0_method=f0_method,
                    resample_sr=resample_sr,
                    index_rate=index_rate,
                    filter_radius=filter_radius,
                    rms_mix_rate=rms_mix_rate,
                    protect=protect
                )
    if st.button(i18n("inference.convert_vocals"),disabled=not (state.input_vocals and state.model_name)):
                output_vocals = convert_vocals(
                    state.model_name,
                    state.input_vocals,
                    **vars(state.convert_params)
                    )
                state.output_vocals = output_vocals
                if output_vocals is not None:
                    mixed_audio = merge_audio(
                        output_vocals,
                        state.input_instrumental,
                        sr=state.input_audio[1]
                    )
                    state.output_audio = mixed_audio
                    state.output_audio_name = get_filename(state.input_audio_name,state.model_name)
    
    with st.container():
        col1, col2 = st.columns(2)
        if state.output_vocals is not None:
            col1.write("Original Vocals")
            col1.audio(state.input_vocals[0],sample_rate=state.input_vocals[1])
            col1.write("Original Song")
            col1.audio(state.input_audio[0],sample_rate=state.input_audio[1])

        if state.output_audio is not None:
            col2.write("Converted Vocals")
            col2.audio(state.output_vocals[0],sample_rate=state.output_vocals[1])
            col2.write("Converted Song")
            col2.audio(state.output_audio[0],sample_rate=state.output_audio[1])
            if col2.button(i18n("inference.download.button")):
                download_song(state.output_audio,state.output_audio_name)
            # col2.download_button(
            #     label=i18n("inference.download.button"),
            #     data=audio_to_bytes(state.output_audio[0],state.output_audio[1]),
            #     file_name=state.output_audio_name,
            #     mime="audio/mpeg")

    return state

def init_state():
    st.session_state["inference"] = st.session_state.get("inference",init_inference_state())

init_state()

if __name__=="__main__": st.session_state.inference=render(st.session_state.inference)