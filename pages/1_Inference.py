import os
import streamlit as st

from webui import DEVICE_OPTIONS, MENU_ITEMS
from lib import config, OUTPUT_DIR, SONG_DIR, i18n
from webui.api import convert_vocals, get_rvc_models, split_vocals
st.set_page_config(layout="centered",menu_items=MENU_ITEMS)

from webui.components import file_uploader_form, initial_vocal_separation_params, initial_voice_conversion_params, save_vocal_separation_params, save_voice_conversion_params, vocal_separation_form, voice_conversion_form
from lib.utils import ObjectNamespace
from webui.contexts import SessionStateContext
from lib.audio import SUPPORTED_AUDIO, bytes_to_audio, merge_audio, remix_audio, save_input_audio

from lib.utils import gc_collect, get_filenames, get_index, get_optimal_torch_device

def call_uvr(audio_path,**kwargs):
    with st.status(f"splitting vocals... {kwargs}") as status:
        try:
            return split_vocals(audio_path,**kwargs)
        except Exception as e:
            status.error(e)
            status.update(state="error")
            return None, None, None

def call_rvc(model_name,input_audio,**kwargs):
    with st.status(f"converting vocals... {model_name} - {kwargs}") as status:
        try:
            return convert_vocals(model_name,input_audio,**kwargs)
        except Exception as e:
            print(e)
            status.error(e)
            status.update(state="error")
            return None

def init_inference_state():
    return ObjectNamespace(
        rvc_models=None,
        device=get_optimal_torch_device(),
        format="flac",
        models=get_rvc_models(),
        model_name=None,
        
        audio_files=get_filenames(exts=SUPPORTED_AUDIO,root=SONG_DIR),
        input_audio_name=None,
        input_audio=None,
        input_vocals=None,
        input_instrumental=None,
        output_audio=None,
        output_audio_name=None,
        output_vocals=None,

        uvr5_params=initial_vocal_separation_params("inference"),
        convert_params=initial_voice_conversion_params("inference")
    )

def refresh_data(state):
    state.models = get_rvc_models()
    state.audio_files = get_filenames(exts=SUPPORTED_AUDIO,root=SONG_DIR)
    gc_collect()
    return state
    
def clear_data(state):
    del state.rvc_models
    state.rvc_models = None
    gc_collect()
    return state

def get_filename(audio_name,model_name):
    song = os.path.basename(audio_name).split(".")[0]
    singer = os.path.basename(model_name).split(".")[0]
    return f"{singer}.{song}"

def one_click_convert(state):
    state.input_vocals, state.input_instrumental, state.input_audio = call_uvr(
        audio_path=state.input_audio_name,
        # device=state.device,
        format=state.format,
        **state.uvr5_params,
        )
    
    params = dict(state.convert_params)
    if state.input_vocals is not None and state.input_instrumental is not None:
        params.update(resample_sr=int(state.input_instrumental[1]))
        changed_vocals = call_rvc(
            state.model_name,
            state.input_vocals,
            **(params)
        )
        
        if changed_vocals:
            state.output_vocals = changed_vocals
            mixed_audio = merge_audio(changed_vocals,state.input_instrumental,sr=state.input_instrumental[1])
            state.output_audio_name = get_filename(
                state.input_audio_name,state.model_name)
            state.output_audio = mixed_audio
    return state

def download_song(output_audio,output_audio_name,ext="mp3"):
    audio_path = output_audio_name.split(".")
    
    output_dir = os.path.join(OUTPUT_DIR,"inference",audio_path[0])
    os.makedirs(output_dir,exist_ok=True)
    output_file = os.path.join(output_dir,f"{audio_path[1]}.{ext}")
    if save_input_audio(output_file,output_audio,to_int16=True):
        return f"successfully saved to {output_file}"
    else: "failed to save"
    
def render_vocal_separation_form(state):
    with st.form("inference.split_vocals.expander"):
        state.uvr5_params = vocal_separation_form(state.uvr5_params)
        
        if st.form_submit_button(i18n("inference.save.button"),type="primary"):
            save_vocal_separation_params("inference",state.uvr5_params)
            st.experimental_rerun()
        elif state.uvr5_params.uvr_models is None: st.write(i18n("inference.uvr_models"))
    return state

def render_voice_conversion_form(state):
    with st.form("inference.convert_vocals.expander"):
        state.convert_params = voice_conversion_form(state.convert_params)
        
        if st.form_submit_button(i18n("inference.save.button"),type="primary"):
            save_voice_conversion_params("inference",state.convert_params)
            st.experimental_rerun()
    return state

if __name__=="__main__":
    with SessionStateContext("inference",initial_state=init_inference_state()) as state:
        file_uploader_form(
                SONG_DIR,"Upload your songs",
                types=SUPPORTED_AUDIO+["zip"],
                accept_multiple_files=True)
        with st.container():
            left, right = st.columns(2)
            state.input_audio_name = left.selectbox(
                i18n("inference.song.selectbox"),
                options=state.audio_files,
                index=get_index(state.audio_files,state.input_audio_name),
                format_func=lambda option: os.path.basename(option)
                )
            col1, col2 = left.columns(2)
            if col1.button(i18n("inference.refresh_data.button"),use_container_width=True):
                state = refresh_data(state)
                st.experimental_rerun()

            if col2.button(i18n("inference.one_click.button"), type="primary",use_container_width=True,
                        disabled=not (state.uvr5_params.uvr_models and state.input_audio_name and state.model_name)):
                with st.spinner(i18n("inference.one_click.button")):
                    state = one_click_convert(state)
            
            state.model_name = right.selectbox(
                i18n("inference.voice.selectbox"),
                options=state.models,
                index=get_index(state.models,state.model_name),
                format_func=lambda option: os.path.basename(option).split(".")[0]
                )
            # col1, col2 = right.columns(2)
            # if col1.button(i18n("inference.load_model.button"),use_container_width=True, type="primary"):
            #     del state.rvc_models
            #     state.rvc_models = load_model(state)
            #     gc_collect()
            # if col2.button(i18n("inference.clear_data.button"),use_container_width=True):
            #     state = clear_data(state)
            #     st.experimental_rerun()
            
        col1, col2 = st.columns(2)
        state.device = col1.radio(
            i18n("inference.device"),
            disabled=not config.has_gpu,
            options=DEVICE_OPTIONS,horizontal=True,
            index=get_index(DEVICE_OPTIONS,state.device))
        state.format = col2.radio(
            i18n("inference.format"),
            options=SUPPORTED_AUDIO,horizontal=True,
            index=get_index(SUPPORTED_AUDIO,state.format))

        st.subheader(i18n("inference.split_vocals"))
        with st.expander(i18n("inference.split_vocals.expander"),expanded=not (state.input_audio_name and len(state.uvr5_params.uvr_models))):
            state = render_vocal_separation_form(state)

        if st.button(i18n("inference.split_vocals"),disabled=not (state.input_audio_name and len(state.uvr5_params.uvr_models))):
            state.input_vocals, state.input_instrumental, state.input_audio = call_uvr(
                audio_path=state.input_audio_name,
                device=state.device,
                format=state.format,
                **(state.uvr5_params),
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
        with st.expander(f"{i18n('inference.convert_vocals.expander')} voice={state.model_name}"):
            state = render_voice_conversion_form(state)

        col1, col2 = st.columns(2)
        uploaded_vocals = col1.file_uploader("Upload your own voice file (if you didn't use voice extraction)",type=SUPPORTED_AUDIO)
        if uploaded_vocals is not None:
            input_audio = bytes_to_audio(
                uploaded_vocals.getvalue())
            state.input_vocals = remix_audio(input_audio,norm=True,to_int16=True,to_mono=True)
            state.input_audio_name = uploaded_vocals.name
            del uploaded_vocals
        uploaded_instrumentals = col2.file_uploader("Upload your own instrumental file (if you didn't use voice extraction)",type=SUPPORTED_AUDIO)
        if uploaded_instrumentals is not None:
            input_audio = bytes_to_audio(
                uploaded_instrumentals.getvalue())
            state.input_instrumental = remix_audio(input_audio,norm=True,to_int16=True,to_mono=True)
            state.input_audio_name = uploaded_instrumentals.name
            del uploaded_instrumentals

        if st.button(i18n("inference.convert_vocals"),disabled=not (state.input_vocals and state.model_name)):
            with st.spinner(i18n("inference.convert_vocals")):
                params = dict(state.convert_params)
                params.update(resample_sr=int(state.input_instrumental[1]))

                output_vocals = call_rvc(
                    state.model_name,
                    state.input_vocals,
                    **params
                    )
                        
                if output_vocals is not None:
                    state.output_vocals = output_vocals
                    if (state.input_instrumental):
                        mixed_audio = merge_audio(
                            output_vocals,
                            state.input_instrumental,
                            sr=state.input_instrumental[1]
                        )
                    else: mixed_audio = output_vocals
                    state.output_audio = mixed_audio
                    state.output_audio_name = get_filename(state.input_audio_name,state.model_name)
        
        with st.container():
            col1, col2 = st.columns(2)
            if state.input_vocals is not None:
                col1.write("Original Vocals")
                col1.audio(state.input_vocals[0],sample_rate=state.input_vocals[1])
            if state.input_audio is not None:
                col1.write("Original Song")
                col1.audio(state.input_audio[0],sample_rate=state.input_audio[1])

            if state.output_vocals is not None:
                col2.write("Converted Vocals")
                col2.audio(state.output_vocals[0],sample_rate=state.output_vocals[1])
            if state.output_audio is not None:
                col2.write("Converted Song")
                col2.audio(state.output_audio[0],sample_rate=state.output_audio[1])
                if col2.button(i18n("inference.download.button")):
                    st.toast(download_song(state.output_audio,state.output_audio_name,ext="flac"))