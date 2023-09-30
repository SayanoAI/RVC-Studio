import hashlib
import json
import os
import sys
import streamlit as st
from webui import MENU_ITEMS, TTS_MODELS, config, get_cwd, i18n, DEVICE_OPTIONS
from webui.chat import init_assistant_template, init_llm_options, init_model_config, init_model_params, Character
from webui.downloader import OUTPUT_DIR
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from webui.audio import save_input_audio
from webui.components import file_uploader_form, initial_voice_conversion_params, voice_conversion_form

import sounddevice as sd
from lib.model_utils import get_hash

from webui.contexts import SessionStateContext

import time
from webui.utils import ObjectNamespace

from webui.utils import gc_collect, get_filenames, get_index, get_optimal_torch_device

CWD = get_cwd()

def get_model_list():
    models_list =  [os.path.relpath(path,CWD) for path in get_filenames(root=os.path.join(CWD,"models"),folder="LLM",exts=["bin","gguf"])]
    return models_list

def get_voice_list():
    models_list = [os.path.relpath(path,CWD) for path in get_filenames(root=os.path.join(CWD,"models"),folder="RVC",exts=["pth"])]
    return models_list

def get_character_list():
    models_list =  [os.path.relpath(path,CWD) for path in get_filenames(root=os.path.join(CWD,"models","RVC"),folder=".characters",exts=["json"])]
    return models_list

def init_state():
    return ObjectNamespace(
        voice_models=get_voice_list(),
        voice_model=None,
        characters=get_character_list(),
        selected_character=None,
        model_config=init_model_config(),
        assistant_template=init_assistant_template(),
        model_params=init_model_params(),
        model_list=get_model_list(),
        tts_options=initial_voice_conversion_params(),
        llm_options=init_llm_options(),
        messages = [],
        user = "",
        device=get_optimal_torch_device(),
        LLM=None,
        tts_method=None,
        character=None
    )
    

def refresh_data(state):
    state.model_list = get_model_list()
    state.voice_models = get_voice_list()
    state.characters = get_character_list()
    return state

def save_character(state):
    with open(os.path.join(CWD,"models","RVC",".characters",f"{state.assistant_template.name}.json"),"w") as f:
        loaded_state = {
            "assistant_template": (state.assistant_template),
            "tts_options": (state.tts_options),
            "voice": state.voice_model,
            "tts_method": state.tts_method
        }
        f.write(json.dumps(loaded_state,indent=2))
    state = refresh_data(state)
    return state

def load_character(state):
    with open(state.selected_character,"r") as f:
        loaded_state = json.load(f)
        state.assistant_template = ObjectNamespace(**loaded_state["assistant_template"])
        
        state.tts_options = (state.tts_options)
        state.tts_options.update(loaded_state["tts_options"])
        state.tts_options = ObjectNamespace(**state.tts_options)
        state.voice_model = loaded_state["voice"]
        state.tts_method = loaded_state["tts_method"]
    state = refresh_data(state)
    return state

def save_model_config(state):
    fname = os.path.join(CWD,"models","LLM","config.json")
    key = get_hash(state.model_params.fname)

    if os.path.isfile(fname):
        with open(fname,"r") as f:
            data = json.load(f)
    else:
        data = {}

    with open(fname,"w") as f:
        data[key] = {
            "version": 2,
            "params": (state.model_params),
            "config": (state.model_config),
            "options": (state.llm_options),
        }
        f.write(json.dumps(data,indent=2))
    state = refresh_data(state)
    return state

def load_model_config(state):
    fname = os.path.join(CWD,"models","LLM","config.json")
    key = get_hash(state.selected_llm)

    with open(fname,"r") as f:
        data = json.load(f) if os.path.isfile(fname) else {}    
        model_data = data[key] if key in data else {**vars(init_model_config()), **vars(init_model_params()), **vars(init_llm_options())}

        if "version" in model_data and model_data["version"]==2:
            state.model_params = ObjectNamespace(**model_data["params"])
            state.model_config = ObjectNamespace(**model_data["config"])
            state.llm_options = ObjectNamespace(**model_data["options"])
        else: # old version
            state.model_config.prompt_template = model_data["prompt_template"]
            state.model_config.chat_template = model_data["chat_template"]
            state.model_config.instruction = model_data["instruction"]
            state.model_config.mapper = model_data["mapper"]
            state.model_params.n_ctx = model_data["n_ctx"]
            state.model_params.n_gpu_layers = model_data["n_gpu_layers"]
            state.llm_options.max_tokens = model_data["max_tokens"]
    state = refresh_data(state)
    return state

def render_model_config_form(state):
    state.model_config.instruction = st.text_area("Instruction",value=state.model_config.instruction)
    state.model_config.chat_template = st.text_area("Chat Template",value=state.model_config.chat_template)
    state.model_config.prompt_template = st.text_area("Prompt Template",value=state.model_config.prompt_template,height=400)
    state.model_config.mapper = st.data_editor(state.model_config.mapper,
                                                        column_order=("_index","value"),
                                                        use_container_width=False,
                                                        num_rows="fixed",
                                                        disabled=["_index"],
                                                        hide_index=False)
    return state

def render_model_params_form(state):
    state.model_params.n_ctx = st.slider("Max Context Length", min_value=512, max_value=4096, step=512, value=state.model_params.n_ctx)
    state.model_params.n_gpu_layers = st.slider("GPU Layers", min_value=0, max_value=64, step=4, value=state.model_params.n_gpu_layers)
    state.llm_options.max_tokens = st.slider("New Tokens",min_value=24,max_value=256,step=8,value=state.llm_options.max_tokens)
    return state

def render_llm_form(state):
    if not state.selected_llm: st.markdown("*Please save your model config below if it doesn't exist!*")
    elif st.button("Load LLM Config",disabled=not state.selected_llm): state=load_model_config(state)
    
    with st.form("model.loader"):
        state = render_model_params_form(state)
        state = render_model_config_form(state)

        if st.form_submit_button("Save Configs",disabled=not state.selected_llm):
            state = save_model_config(state)
            st.experimental_rerun()
    return state

def render_tts_options_form(state):

    col1, col2 =st.columns(2)
    state.tts_method = col1.selectbox(
                i18n("tts.model.selectbox"),
                options=TTS_MODELS,
                index=get_index(TTS_MODELS,state.tts_method),
                format_func=lambda option: option.upper()
                )
    state.voice_model = col2.selectbox(
            i18n("inference.voice.selectbox"),
            options=state.voice_models,
            index=get_index(state.voice_models,state.voice_model),
            format_func=lambda option: os.path.basename(option).split(".")[0]
            )
    state.tts_options = voice_conversion_form(state.tts_options)
    return state

def render_assistant_template_form(state):
    state.assistant_template.name = st.text_input("Character Name",value=state.assistant_template.name)
    ROLE_OPTIONS = ["CHARACTER", "USER"]
    state.assistant_template.background = st.text_area("Background", value=state.assistant_template.background, max_chars=900)
    state.assistant_template.personality = st.text_area("Personality", value=state.assistant_template.personality, max_chars=900)
    st.write("Example Dialogue")
    state.assistant_template.examples = st.data_editor(state.assistant_template.examples,
                                                        column_order=("role","content"),
                                                        column_config={
                                                            "role": st.column_config.SelectboxColumn("Role",options=ROLE_OPTIONS,required=True),
                                                            "content": st.column_config.TextColumn("Content",required=True)
                                                        },
                                                        use_container_width=True,
                                                        num_rows="dynamic",
                                                        hide_index =True)
    state.assistant_template.greeting = st.text_input("Greeting",value=state.assistant_template.greeting,max_chars=200)
    return state

def render_character_form(state):
    if not state.selected_character: st.markdown("*Please create a character below if it doesn't exist!*")
    elif st.button("Load Character Info",disabled=not state.selected_character): state=load_character(state)
        
    with st.form("character"):
        state = render_tts_options_form(state)
        state = render_assistant_template_form(state)
        if st.form_submit_button("Save"):
            state = save_character(state)
            st.experimental_rerun()
    
    return state

if __name__=="__main__":
    with SessionStateContext("chat",init_state()) as state:
        
        col1, col2 = st.columns(2)
        if col1.button("Refresh Files"):
            state = refresh_data(state)
        if col2.button("Unload Models",disabled=state.character is None):
            state.character.unload()
            del state.character
            gc_collect()

        state.user = col1.text_input("Your Name", value=state.user)
        state.selected_character = col2.selectbox("Your Character",
                                              options=state.characters,
                                              index=get_index(state.characters,state.selected_character),
                                              format_func=lambda x: os.path.basename(x))
        state.selected_llm = col2.selectbox("Choose a language model",
                                options=state.model_list,
                                index=get_index(state.model_list,state.selected_llm),
                                format_func=lambda x: os.path.basename(x))

        with col1:
            c1, c2 = st.columns(2)
            state.device = c1.radio(
                i18n("inference.device"),
                disabled=not config.has_gpu,
                options=DEVICE_OPTIONS,horizontal=True,
                index=get_index(DEVICE_OPTIONS,state.device))
            
            
            if c2.button("Start Chatting",disabled=not (state.selected_character and state.selected_llm and state.user),type="primary"):
                del state.character
                gc_collect()
                state.character = Character(
                    voice_file=state.selected_character,
                    model_file=state.selected_llm,
                    user=state.user,
                    device=state.device
                )
                state.character.load()
                st.experimental_rerun()

        chat_disabled = state.character is None or not state.character.loaded
        if chat_disabled: st.subheader("Select your character and language model above to get started!")

        # chat settings
        with st.expander(f"Chat Options: character={state.selected_character} LLM={state.selected_llm}", expanded=chat_disabled):
            character_tab, llm_tab = st.tabs(["Character","LLM"])
            with llm_tab:
                state = render_llm_form(state)
            with character_tab:
                state = render_character_form(state)
        
        if not chat_disabled:

            # save/load chat history
            save_dir = os.path.join(OUTPUT_DIR,"chat",state.character.name)
            file_uploader_form(save_dir,types=["zip"])
            state.history_file = st.selectbox("Select a history",options=[""]+get_filenames(root=save_dir,name_filters=["json"]))
            col1,col2, col3 = st.columns(3)

            if col1.button("Save Chat",disabled=not state.character):
                st.toast(state.character.save_history())

            if col2.button("Load Chat",disabled=not state.history_file):
                st.toast(state.character.load_history(state.history_file))

            if col3.button("Clear Chat",type="primary",disabled=state.character is None or len(state.character.messages)==0):
                state.character.clear_chat()

            # display chat messages
            for i,msg in enumerate(state.character.messages):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    col1, col2, col3 = st.columns(3)
                    if msg.get("audio"):
                        if col1.button("Play",key=f"Play{i}"): sd.play(*msg["audio"])
                        if col2.button("Download",key=f"Download{i}"):
                            fname=os.path.join(OUTPUT_DIR,"chat",msg['role'],
                                                        f"{i}_{hashlib.md5(msg['content'].encode('utf-8')).hexdigest()}.wav")
                            if save_input_audio(fname,msg["audio"]):
                                st.toast(f"successfully saved to: {fname}")
                    if col3.button("Delete",key=f"Delete{i}"):
                        st.toast(f"Deleted message: {state.character.messages.pop(i)}")
                        st.experimental_rerun()

            # container = st.container()
            if st.button("Summarize Context"):
                st.write(state.character.summarize_context())
            if state.character.is_recording:
                if st.button("Stop Voice Chat",type="primary"):
                    state.character.is_recording=False
                    st.experimental_rerun()
                with st.spinner("Listening to mic..."):
                    time.sleep(1)
                    st.experimental_rerun()
            elif st.button("Voice Chat (WIP)",type="secondary" ):
                state.character.speak_and_listen()
                st.experimental_rerun()
            elif st.button("Toggle Autoplay",type="primary" if state.character.autoplay else "secondary" ):
                state.character.toggle_autoplay()

            if prompt:=st.chat_input(disabled=chat_disabled or state.character.autoplay) or state.character.autoplay:
                state.character.is_recording=False
                if not state.character.autoplay:
                    st.chat_message(state.character.user).write(prompt)
                full_response = ""
                with st.chat_message(state.character.name):
                    message_placeholder = st.empty()
                    for response in state.character.generate_text("ok, go on" if state.character.autoplay else prompt):
                        full_response += response
                        message_placeholder.markdown(full_response)
                audio = state.character.text_to_speech(full_response)
                if audio: sd.play(*audio)
                if not state.character.autoplay:
                    state.character.messages.append({"role": state.character.user, "content": prompt}) #add user prompt to history
                state.character.messages.append({
                    "role": state.character.name,
                    "content": full_response,
                    "audio": audio
                    })
                if state.character.autoplay:
                    st.experimental_rerun()

            
            
        