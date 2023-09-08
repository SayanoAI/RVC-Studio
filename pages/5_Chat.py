import json
import os
import sys
import streamlit as st
import sounddevice as sd
from lib.model_utils import get_hash
from tts_cli import generate_speech
from vc_infer_pipeline import get_vc, vc_single
from web_utils import MENU_ITEMS
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from web_utils.contexts import SessionStateContext

from llama_cpp import Llama
import time
from types import SimpleNamespace

from webui_utils import gc_collect, get_filenames, get_index, config, i18n

CWD = os.getcwd()
if CWD not in sys.path:
    sys.path.append(CWD)

DEVICE_OPTIONS = ["GPU", "CPU"]

def get_model_list():
    models_list = get_filenames(root="./models",folder="LLM",exts=["bin","gguf"])
    return models_list

def get_voice_list():
    models_list = get_filenames(root="./models",folder="RVC",exts=["pth"])
    return models_list

def get_character_list():
    models_list = get_filenames(root="./models",folder="RVC/.characters",exts=["json"])
    return models_list

def load_model(fname,n_ctx,n_gpu_layers):
    model = Llama(fname,
                  n_ctx=n_ctx,
                  n_gpu_layers=n_gpu_layers,
                  verbose=False
                  )
    return model

def init_llm_options():
    params = SimpleNamespace(**{
        "top_k": 42,
        "repeat_penalty": 1.1,
        "frequency_penalty": 0.,
        "presence_penalty": 0.,
        "tfs_z": 1.0,
        "mirostat_mode": 0,
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1,
        "suffix": None,
        "max_tokens": 64,
        "temperature": .8,
        "top_p": .95,
    })
    return params

def init_model_config():
    return SimpleNamespace(
        # template placeholder
        prompt_template = "",
        # how dialogs appear
        chat_template = "",
        # Default system instructions
        instruction = "",
        # maps role names
        mapper={
            "CHARACTER": "",
            "USER": ""
        }
    )
def init_assistant_template():
    return SimpleNamespace(
        background = "",
        personality = "",
        examples = [{"role": "", "content": ""}],
        greeting = "",
        name = ""
    )

def build_context(state,prompt,memory=100):
    model_config=state.model_config
    assistant_template=state.assistant_template
    chat_history=state.messages
    chat_mapper = {
        state.user: model_config.mapper["USER"],
        assistant_template.name: model_config.mapper["CHARACTER"]
    }
    # Concatenate chat history and system template
    examples = [
        model_config.chat_template.format(role=model_config.mapper[ex["role"]],content=ex["content"])
            for ex in assistant_template.examples if ex["role"] and ex["content"]]+[
        model_config.chat_template.format(role=chat_mapper[ex["role"]],content=ex["content"])
            for ex in chat_history[-memory:]]
        
    instruction = model_config.instruction.format(name=assistant_template.name,user=state.user)
    persona = f"{assistant_template.background} {assistant_template.personality}"
    context = "\n".join(examples)
    
    chat_history_with_template = model_config.prompt_template.format(
        context=context,
        instruction=instruction,
        persona=persona,
        name=assistant_template.name,
        user=state.user,
        prompt=prompt
        )

    print(chat_history_with_template)
    return chat_history_with_template

def generate_response(state,prompt):
    context = build_context(state,prompt)
    generator = state.LLM.create_completion(
        context,stream=True,stop=["*","\n",state.model_config.mapper["USER"],state.model_config.mapper["CHARACTER"]],**vars(state.llm_options))
    
    response = ""
    last_update = time.time()
    for completion_chunk in generator:
        response += completion_chunk['choices'][0]['text']
        cur_time = time.time()
        if cur_time - last_update > 1./24:  # Limit streaming to 24 fps
            last_update = cur_time
            yield response

def init_model_params():
    return SimpleNamespace(
        fname=None,n_ctx=2048,n_gpu_layers=0
    )

def init_state():
    state = SimpleNamespace(
        voice_models=get_voice_list(),
        voice_model=None,
        characters=get_character_list(),
        selected_character=None,
        model_config=init_model_config(),
        assistant_template=init_assistant_template(),
        model_params=init_model_params(),
        model_list=get_model_list(),
        tts_options=SimpleNamespace(
            f0_up_key=6,
            f0_method="rmvpe",
            index_rate=.8,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=.25,
            protect=0.25
        ),
        llm_options=init_llm_options(),
        messages = [],
        user = "USER",
        device = "cuda",
        LLM=None,
        tts_method=None
    )
    return vars(state)

def refresh_data(state):
    state.model_list = get_model_list()
    state.voice_models = get_voice_list()
    state.characters = get_character_list()
    return state

def text_to_speech(text,models=None,speaker="Sayano",method="edge",device="gpu",**tts_options):
    tts_audio = generate_speech(text,speaker=speaker,method=method, device=device)
    input_audio = vc_single(input_audio=tts_audio,**models,**tts_options)
    return input_audio

def save_character(state):
    with open(os.path.join(CWD,"models","RVC",".characters",f"{state.assistant_template.name}.json"),"w") as f:
        loaded_state = {
            "assistant_template": vars(state.assistant_template),
            "tts_options": vars(state.tts_options),
            "voice": state.voice_model,
            "tts_method": state.tts_method
        }
        f.write(json.dumps(loaded_state,indent=2))
    del state.models
    state.models = get_vc(state.voice_model,config=config,device=state.device)
    gc_collect()
    state = refresh_data(state)
    return state

def load_character(state):
    with open(state.selected_character,"r") as f:
        loaded_state = json.load(f)
        state.assistant_template = SimpleNamespace(**loaded_state["assistant_template"])
        state.tts_options = SimpleNamespace(**loaded_state["tts_options"])
        state.voice_model = loaded_state["voice"]
        state.tts_method = loaded_state["tts_method"]
    del state.models
    state.models = get_vc(state.voice_model,config=config,device=state.device)
    gc_collect()
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
            "name": os.path.basename(state.model_params.fname),
            "prompt_template": state.model_config.prompt_template,
            "chat_template": state.model_config.chat_template,
            "instruction": state.model_config.instruction,
            "mapper": state.model_config.mapper,
            "n_ctx": state.model_params.n_ctx,
            "n_gpu_layers": state.model_params.n_gpu_layers,
            "max_tokens": state.llm_options.max_tokens
        }
        f.write(json.dumps(data,indent=2))
    del state.LLM
    state.LLM = load_model(**vars(state.model_params))
    gc_collect()
    state = refresh_data(state)
    return state

def load_model_config(state):
    fname = os.path.join(CWD,"models","LLM","config.json")
    key = get_hash(state.model_params.fname)

    with open(fname,"r") as f:
        data = json.load(f) if os.path.isfile(fname) else {}    
        model_config = data[key] if key in data else {**vars(init_model_config()), **vars(init_model_params()), **vars(init_llm_options())}
        state.model_config.prompt_template = model_config["prompt_template"]
        state.model_config.chat_template = model_config["chat_template"]
        state.model_config.instruction = model_config["instruction"]
        state.model_config.mapper = model_config["mapper"]
        state.model_params.n_ctx = model_config["n_ctx"]
        state.model_params.n_gpu_layers = model_config["n_gpu_layers"]
        state.llm_options.max_tokens = model_config["max_tokens"]
    del state.LLM
    state.LLM = load_model(**vars(state.model_params))
    gc_collect()
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
    state.llm_options.max_tokens = st.slider("New Tokens",min_value=24,max_value=128,step=8,value=state.llm_options.max_tokens)
    return state

def render_llm_form(state):
    col1,col2 = st.columns(2)
    state.model_params.fname = col1.selectbox("Choose a language model",
                                options=state.model_list,
                                index=get_index(state.model_list,state.model_params.fname),
                                format_func=lambda x: os.path.basename(x))
    col2.markdown("*Please save your model config below if it doesn't exist!*")
    if col2.button("Load Model", disabled=not state.model_params.fname, type="primary"):
        state = load_model_config(state)
        st.experimental_rerun()
    
    with st.form("model.loader"):
        state = render_model_params_form(state)
        state = render_model_config_form(state)

        if st.form_submit_button("Save Configs",disabled=not state.model_params.fname):
            state = save_model_config(state)
            st.experimental_rerun()
    return state

def render_tts_options_form(state):
    PITCH_EXTRACTION_OPTIONS = ["crepe","rmvpe"]
    TTS_MODELS = ["edge","vits","speecht5","bark","tacotron2"]

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
    
    state.tts_options.f0_up_key = st.slider(i18n("inference.f0_up_key"),min_value=-12,max_value=12,step=1,value=state.tts_options.f0_up_key)
    state.tts_options.f0_method = st.selectbox(i18n("inference.f0_method"),
                                        options=PITCH_EXTRACTION_OPTIONS,
                                        index=get_index(PITCH_EXTRACTION_OPTIONS,state.tts_options.f0_method))
    state.tts_options.resample_sr = st.select_slider(i18n("inference.resample_sr"),
                                        options=[0,16000,24000,22050,40000,44100,48000],
                                        value=state.resample_sr)
    state.tts_options.index_rate=st.slider(i18n("inference.index_rate"),min_value=0.,max_value=1.,step=.05,value=state.tts_options.index_rate)
    state.tts_options.filter_radius=st.slider(i18n("inference.filter_radius"),min_value=0,max_value=7,step=1,value=state.tts_options.filter_radius)
    state.tts_options.rms_mix_rate=st.slider(i18n("inference.rms_mix_rate"),min_value=0.,max_value=1.,step=.05,value=state.tts_options.rms_mix_rate)
    state.tts_options.protect=st.slider(i18n("inference.protect"),min_value=0.,max_value=.5,step=.01,value=state.tts_options.protect)
    return state

def render_assistant_template_form(state):
    state.assistant_template.name = st.text_input("Character Name",
                                                    value=os.path.basename(state.voice_model).split(".")[0] if state.voice_model else state.assistant_template.name)
    ROLE_OPTIONS = ["CHARACTER", "USER"]
    state.assistant_template.background = st.text_area("Background", value=state.assistant_template.background, max_chars=400)
    state.assistant_template.personality = st.text_area("Personality", value=state.assistant_template.personality, max_chars=400)
    state.assistant_template.examples = st.data_editor(state.assistant_template.examples,
                                                        column_order=("role","content"),
                                                        column_config={
                                                            "role": st.column_config.SelectboxColumn("Role",options=ROLE_OPTIONS,required=True),
                                                            "content": st.column_config.TextColumn("Content",required=True)
                                                        },
                                                        use_container_width=True,
                                                        num_rows="dynamic",
                                                        hide_index =True)
    state.assistant_template.greeting = st.text_input("Greeting",value=state.assistant_template.greeting,max_chars=100)
    return state

def render_character_form(state):
    DEVICE_OPTIONS = ["cpu","cuda"]

    col1, col2, col3 =st.columns(3)
    state.user = col1.text_input("Your Name", value=state.user)
    state.selected_character = col2.selectbox("Character",
                                              options=state.characters,
                                              index=get_index(state.characters,state.selected_character),
                                              format_func=lambda x: os.path.basename(x))
    col2.markdown("*Please create a character below if it doesn't exist!*")
    state.device = col3.radio(
        i18n("inference.device"),
        disabled=not config.has_gpu,
        options=DEVICE_OPTIONS,horizontal=True,
        index=get_index(DEVICE_OPTIONS,state.device))
    if col3.button("Load Character", disabled=state.selected_character is None, type="primary"):
        state = load_character(state)
        st.experimental_rerun()
        
    with st.form("character"):
        state = render_tts_options_form(state)
        state = render_assistant_template_form(state)
        if st.form_submit_button("Save",disabled=not (state.voice_model and state.assistant_template.background and state.assistant_template.personality)):
            state = save_character(state)
            st.experimental_rerun()
    
    return state

if __name__=="__main__":
    with SessionStateContext("chat",init_state()) as state:
        chat_disabled = not (state.LLM and state.models)

        col1, col2, col3 = st.columns(3)
        if col1.button("Refresh Files"):
            state = refresh_data(state)
        if col2.button("Unload Models",disabled=state.LLM is None):
            del state.LLM, state.models
            state.LLM = None
            state.models = None
            gc_collect()
        if col3.button("Clear Chat",type="primary",disabled=len(state.messages)==0):
            del state.messages
            state.messages = []
            gc_collect()

        with st.expander(f"Chat Settings: voice_model={state.voice_model} LLM={state.LLM} character={state.assistant_template.name}", expanded=chat_disabled):
            
            llm_tab, character_tab = st.tabs(["LLM","Character"])
            with llm_tab:
                state = render_llm_form(state)
            with character_tab:
                state = render_character_form(state)

        if len(state.messages)==0 and state.assistant_template.greeting:
            state.messages.append({"role": state.assistant_template.name, "content": state.assistant_template.greeting})
        for i,msg in enumerate(state.messages):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("audio") and st.button("Play",key="Play"+str(msg)):
                    sd.play(*msg["audio"])

        if prompt := st.chat_input(disabled=chat_disabled):
            st.chat_message(state.user).write(prompt)
            full_response = ""
            with st.chat_message(state.assistant_template.name):
                message_placeholder = st.empty()
                chat_history = [ f"### {msg['role']}:\n{msg['content']}" for msg in state.messages]
                for response in generate_response(state,prompt):
                    full_response = response
                    message_placeholder.markdown(response)
                message_placeholder.markdown(full_response)
            state.audio = text_to_speech(full_response, models=state.models, speaker=state.assistant_template.name, method=state.tts_method,device=state.device, **vars(state.tts_options))
            if state.audio: sd.play(*state.audio)
            state.messages.append({"role": state.user, "content": prompt}) #add user prompt to history
            state.messages.append({
                "role": state.assistant_template.name,
                "content": full_response,
                "audio": state.audio
                })
        
        