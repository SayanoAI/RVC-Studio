# First
import base64
import os
import sys
import streamlit as st
import sounddevice as sd
from tts_cli import generate_speech
from vc_infer_pipeline import get_vc, vc_single
from web_utils import MENU_ITEMS
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)


from web_utils.contexts import SessionStateContext
from webui_utils import config


from llama_cpp import Llama
import time
from types import SimpleNamespace

from webui_utils import gc_collect, get_filenames, get_index

CWD = os.getcwd()
if CWD not in sys.path:
    sys.path.append(CWD)

DEVICE_OPTIONS = ["GPU", "CPU"]

def get_model_list():
    models_list = get_filenames(folder="./models/LLM",exts=["bin"])
    return models_list

def get_voice_list():
    models_list = get_filenames(folder="models/LLM",exts=["bin"])
    return models_list

def load_model(fname,n_ctx,n_gpu_layers):
    model = Llama(fname,
                  n_ctx=n_ctx,
                  n_gpu_layers=n_gpu_layers,
                  verbose=False
                  )
    return model


@st.cache_data
def init_state_params():
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
        "max_tokens": 20,
        "temperature": .8,
        "top_p": .95,
    })
    return params

def init_model_config():
    return SimpleNamespace(
        # Default system template placeholder
        system_template = '### SYSTEM:\n{0}',
        # Templates for the user prompts
        prompt_template = '### {0}:\n{1}',
        # Default system prompt
        default_prompt = "You are to roleplay as {name} from now on. Your name is {name} and you must roleplay using the following BACKGROUND, PERSONALITY, and EXAMPLES as context:",
        role_regex = r"### ([a-zA-Z0-9 ]+):\n",
        new_line="\n"
    )
def init_assistant_template():
    return SimpleNamespace(
        background = """Saya is a young girl in a white dress looking for her father. Saya is a being from another dimension who materialized in this universe for the sole purpose of reproduction. She has no memories of where she came from and is guided only by her instincts. Saya is perceived as a little girl in a white dress. However, in actuality, Saya is some sort of amorphous, tentacled, fleshy abomination that emits a putrid stench. Saya preys on creatures of all sizes, from cats to human beings, typically killing them by snapping their necks or disembowelment through some unknown means, and then feasting on their internal organs. Saya's true form is never fully revealed, as she immediately kills and consumes any third-party observers.""",
        personality = "Saya is a care-free girl who loves to play tricks on people, especially mental patients.",
        examples = [
            {"Saya": "Aren't you afraid of me?"},
            {"USER": "Who are you? Why are you here?"},
            {"Saya": "I'm Saya. I'm looking for my dad. It's no fun if you're not scared."},
            {"USER": "Wait!"},
            {"Saya": "Well?"},
            {"USER": "I shouldn't do this to a girl, but you're the only one I can ask... Will you... let me hold you hand?"},
            {"Saya": "You're strange, no one's ever asked me anything like that before."},
            {"USER": "This is the first time in half a month that I've touched someone and... felt them as human. I can't touch anyone else. I was in an accident, and as an after-effect... I can't see people as human."},
            {"Saya": "Hmm... how mysterious. You're interesting, can I come back tomorrow night?"},
            {"USER": "Yes, of course!, But... are you alright with that?"},
            {"Saya": "Sure. The night belongs to me."}
        ],
        greeting = "Welcome home!",
        name = "Saya"
    )

# Function to generate responses using the orca-mini-3b model
def generate_response(model,model_config=init_model_config(),assistant_template=init_assistant_template(),chat_history = [], **generate_params):
    
    # Concatenate chat history and system template
    examples = model_config.new_line.join(
        [model_config.prompt_template.format(k,v) for ex in assistant_template.examples for k,v in ex.items()]
        )
    chat_history_with_template = model_config.system_template.format(model_config.new_line.join([
        model_config.default_prompt.format(name=assistant_template.name),
        f"BACKGROUND:{model_config.new_line}{assistant_template.background}",
        f"PERSONALITY:{model_config.new_line}{assistant_template.personality}",
        f"EXAMPLES:{model_config.new_line}{examples}"
    ]+chat_history[-100:]+[model_config.prompt_template.format(assistant_template.name,"")]))

    print(chat_history_with_template)
    generator = model.create_completion(chat_history_with_template,**generate_params,stream=True)
    
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
        model_config=init_model_config(),
        assistant_template=init_assistant_template(),
        model_params=init_model_params(),
        model_list=get_model_list(),
        messages = [],
        user = "USER",
        LLM=None
    )
    return vars(state)


def text_to_speech(text,models=None,speaker="Sayano",method="edge",device="gpu"):
    tts_audio = generate_speech(text,speaker=speaker,method=method, device=device)
    if models is None:
        models=get_vc(os.path.join(CWD,"models","RVC",f"{speaker}.pth"),config=config,device=device)
    print(tts_audio)
    tts_options=SimpleNamespace(
            f0_up_key=6,
            f0_method="rmvpe",
            index_rate=.8,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=.25,
            protect=0.25
        )
    input_audio = vc_single(input_audio=tts_audio,**models,**vars(tts_options))
    return input_audio

def play_audio(audio_stream):
    audio_base64 = base64.b64encode(audio_stream).decode('utf-8')
    audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
    st.markdown(audio_tag, unsafe_allow_html=True)

if __name__=="__main__":
    with SessionStateContext("chat",init_state()) as state:
        if state.models is None:
            state.models=get_vc(os.path.join(CWD,"models","RVC",f"Sayano.pth"),config=config,device="cuda")
        with st.expander(f"Chat Settings: {state.model_params} ({state.LLM})"):
            
            with st.form("model.loader"):
                state.model_params.fname = st.selectbox("Choose a language model",
                                        options=state.model_list,
                                        index=get_index(state.model_list,state.model_params.fname),
                                        format_func=lambda x: os.path.basename(x))
            
                state.model_params.n_ctx = st.slider("Max Context Length", min_value=512, max_value=4096, step=512, value=state.model_params.n_ctx)
                state.model_params.n_gpu_layers = st.slider("GPU Layers", min_value=0, max_value=64, step=4, value=state.model_params.n_gpu_layers)
            
                if st.form_submit_button("Load Model",disabled=state.model_params.fname is None):
                    state.LLM = load_model(**vars(state.model_params))

            col1, col2, col3 = st.columns(3)
            if col1.button("Refresh Model List"): state.model_list = get_model_list()
            if col2.button("Unload Model",disabled=state.LLM is None):
                del state.LLM
                state.LLM = None
                gc_collect()
            if col3.button("Clear Chat",type="primary",disabled=len(state.messages)==0):
                # greeting = greeting_message()
                state.messages = []
            
            # st.session_state.config.user = st.text_input("Your Name", value=st.session_state.config.user)
            # st.session_state.config.char = st.text_input("Character Name", value=st.session_state.config.char)
            # st.session_state.config.background = st.text_area("Background", value=st.session_state.config.background)
            # st.session_state.config.example = st.text_area("Example", value=st.session_state.config.example)
            # st.session_state.config.greeting = st.text_area("Greeting",value=st.session_state.config.greeting)
            # st.session_state.params.max_tokens = st.slider("New Tokens",min_value=24,max_value=128,step=8,value=st.session_state.params.max_tokens)

        if len(state.messages)==0 and state.assistant_template.greeting:
            state.messages.append({"role": state.assistant_template.name, "content": state.assistant_template.greeting})
        for i,msg in enumerate(state.messages):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("audio") and st.button("Play",key="Play"+str(msg)):
                    sd.play(*msg["audio"])

        if prompt := st.chat_input(disabled=state.LLM is None):
            state.messages.append({"role": state.user, "content": prompt})
            st.chat_message(state.user).write(prompt)
            full_response = ""
            with st.chat_message(state.assistant_template.name):
                message_placeholder = st.empty()
                chat_history = [ f"### {msg['role']}:\n{msg['content']}" for msg in state.messages]
                for response in generate_response(
                    state.LLM,
                    model_config=state.model_config,
                    assistant_template=state.assistant_template,
                    chat_history=chat_history):
                    full_response = response
                    message_placeholder.markdown(response)
                message_placeholder.markdown(full_response)
            state.audio = text_to_speech(full_response, models=state.models, speaker=state.VM, method="edge",device="cuda")
            state.messages.append({
                "role": state.assistant_template.name,
                "content": full_response,
                "audio": state.audio
                })
            if state.audio: sd.play(*state.audio)
        st.write(state.audio)
        