from datetime import datetime
import hashlib
import json
import os
import threading
from webui.utils import ObjectNamespace
from llama_cpp import Llama
import numpy as np
from lib.model_utils import get_hash
from tts_cli import generate_speech, load_stt_models, transcribe_speech
from webui.audio import bytes_to_audio, load_input_audio, save_input_audio
from webui.downloader import BASE_MODELS_DIR, OUTPUT_DIR
import sounddevice as sd
from webui.sumy_summarizer import get_summary

from webui.utils import gc_collect
from . import config

from vc_infer_pipeline import get_vc, vc_single

def init_model_params():
    return ObjectNamespace(
        fname=None,n_ctx=2048,n_gpu_layers=0
    )
def init_model_config():
    return ObjectNamespace(
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
def init_llm_options():
    return ObjectNamespace(
        top_k = 42,
        repeat_penalty = 1.1,
        frequency_penalty = 0.,
        presence_penalty = 0.,
        tfs_z = 1.0,
        mirostat_mode = 1,
        mirostat_tau = 5.0,
        mirostat_eta = 0.1,
        suffix = None,
        max_tokens = 256,
        temperature = .8,
        top_p = .9,
    )
    
def init_model_data():
    return {
        "params": vars(init_model_params()),
        "config": vars(init_model_config()),
        "options": vars(init_llm_options())
    }

def init_assistant_template():
    return ObjectNamespace(
        background = "",
        personality = "",
        examples = [{"role": "", "content": ""}],
        greeting = "",
        name = ""
    )

def load_character_data(fname):
    with open(fname,"r") as f:
        loaded_state = json.load(f)
    return loaded_state

def load_model_data(model_file):
    fname = os.path.join(BASE_MODELS_DIR,"LLM","config.json")
    key = get_hash(model_file)
    model_data = init_model_data()

    with open(fname,"r") as f:
        data = json.load(f) if os.path.isfile(fname) else model_data
        if key in data:
            model_data["params"].update(data[key]["params"])
            model_data["config"].update(data[key]["config"])
            model_data["options"].update(data[key]["options"])

    return model_data

# Define a Character class
class Character:
    # Initialize the character with a name and a voice
    def __init__(self, voice_file, model_file, memory = 0, user="",stt_method="speecht5",device=None):
        self.voice_file = voice_file
        self.model_file = model_file
        self.voice_model = None
        self.stt_models = None
        self.loaded = False
        self.sample_rate = 16000 # same rate as hubert model
        self.memory = memory
        self.messages = []
        self.user = user
        self.is_recording = False
        self.stt_method = stt_method
        self.device=device
        self.autoplay = False
        self.recognizer = None
        self.listener = None
        self.lock = threading.Lock()

        #load data
        self.character_data = load_character_data(voice_file)
        self.model_data = load_model_data(self.model_file)
        self.name = self.character_data["assistant_template"]["name"]

        # build context
        self.context_summary = ""
        self.context_size = 0
        self.context_index = 0
        self.memory = memory if memory else int(np.log(self.model_data["params"]["n_ctx"])*2.5)+1 # memory is log(context_window)*2.5 words/token
        self.max_memory = int(np.sqrt(self.model_data["params"]["n_ctx"]))+1
        self.context = self.build_context("")
        
    def __del__(self):
        self.unload()

    def stop_listening(self):
        if self.listener:
            self.is_recording = False
            self.listener.join(1)

    def load(self,verbose=False):
        assert not self.loaded, "Model is already loaded"

        try:
            # load LLM first
            self.LLM = Llama(self.model_file,
                    n_ctx=self.model_data["params"]["n_ctx"],
                    n_gpu_layers=self.model_data["params"]["n_gpu_layers"],
                    verbose=verbose
                    )
            self.context_size = len(self.LLM.tokenize(self.context.encode("utf-8")))
            self.free_tokens = self.model_data["params"]["n_ctx"] - self.context_size
            self.LLM.create_completion(self.context,max_tokens=1) #preload
            # self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn",framework="pt",device=self.device)

            # load voice model
            self.voice_model = get_vc(self.character_data["voice"],config=config,device=self.device)
            if len(self.messages)==0 and self.character_data["assistant_template"]["greeting"] and self.user:
                greeting_message = { #add greeting message
                    "role": self.character_data["assistant_template"]["name"],
                    "content": self.character_data["assistant_template"]["greeting"].format(
                        name=self.character_data["assistant_template"]["name"], user=self.user)}
                output_audio = self.text_to_speech(greeting_message["content"])
                if (output_audio):
                    sd.play(*output_audio)
                    greeting_message["audio"] = output_audio
                self.messages.append(greeting_message)
            
            self.loaded=True
        except Exception as e:
            print(e)
            self.loaded=False

    def unload(self):
        del self.LLM, self.voice_model, self.stt_models, self.recognizer
        gc_collect()
        self.loaded=False
        self.is_recording = False
        self.stop_listening()
        print("models unloaded")

    def toggle_autoplay(self):
        self.autoplay = not self.autoplay
        self.is_recording = False

    def clear_chat(self):
        del self.messages
        self.messages = []
        gc_collect()

    @property
    def save_dir(self):
        history_dir = os.path.join(OUTPUT_DIR,"chat",self.name)
        num = len(os.listdir(history_dir)) if os.path.exists(history_dir) else 0
        save_dir = os.path.join(history_dir,f"{datetime.now().strftime('%Y-%m-%d')}_chat{num}")
        return save_dir

    def save_history(self):
        save_dir = self.save_dir
        os.makedirs(save_dir,exist_ok=True)
        messages = []

        try:
            for i,msg in enumerate(self.messages):
                if msg["role"]==self.name: role = "CHARACTER"
                elif msg["role"]==self.user: role = "USER"
                else: role = msg["role"]
                content = msg['content'].replace(self.user,"USER").replace(self.name,"CHARACTER")
                message = {
                    "role": role,
                    "content": content
                }
                if "audio" in msg:
                    fname=os.path.join(save_dir,f"{i}_{hashlib.md5(content.encode('utf-8')).hexdigest()}.wav")
                    save_input_audio(fname,msg["audio"])
                    message["audio"]=os.path.relpath(fname,save_dir)
                messages.append(message)
            text = json.dumps({"messages":messages},indent=2)
            with open(os.path.join(save_dir,"messages.json"),"w") as f:
                f.write(text)
            return f"Chat successfully saved in {save_dir}"
        except Exception as e:
            return f"Chat failed to save: {e}"

    def load_history(self,history_file):

        messages = []
        save_dir = os.path.dirname(history_file)

        try:
            with open(os.path.join(history_file),"r") as f:
                data = json.load(f)
                saved_messages = data["messages"]

            for msg in saved_messages:
                if msg["role"]=="CHARACTER": role = self.name
                elif msg["role"]=="USER": role = self.user
                else: role = msg["role"]
                content = msg['content'].replace("USER",self.user).replace("CHARACTER",self.name)
                message = {
                    "role": role,
                    "content": content
                }
                if "audio" in msg:
                    fname=os.path.join(save_dir,msg["audio"])
                    message["audio"] = load_input_audio(fname)
                messages.append(message)
            self.messages = messages
            # summarize messages after load
            self.context_index = 0
            self.context = self.build_context("")
            
            return f"Chat successfully loaded from {save_dir}!"
        except Exception as e:
            return f"Chat failed to load: {e}"

    # Define a method to generate text using llamacpp model
    def generate_text(self, input_text):
        assert self.loaded, "Please load the models first"

        model_config = self.model_data["config"]
        # Send the input text to llamacpp model as a prompt
        self.context = self.build_context(input_text)
        generator = self.LLM.create_completion(
            self.context,stream=True,stop=[
                "*","\n",
                model_config["mapper"]["USER"],
                model_config["mapper"]["CHARACTER"]
                ],**self.model_data["options"])
        
        for completion_chunk in generator:
            response = completion_chunk['choices'][0]['text']
            yield response

    def build_context(self,prompt: str):
        model_config = self.model_data["config"]
        assistant_template = self.character_data["assistant_template"]
        chat_mapper = {
            self.user: model_config["mapper"]["USER"],
            assistant_template["name"]: model_config["mapper"]["CHARACTER"]
        }

        # clear chat history if memory maxed
        if len(self.messages[self.context_index:])>self.max_memory:
            self.context_index+=self.max_memory
            self.context_summary = self.summarize_context()
        # summarize memory
        elif self.loaded and len(self.LLM.tokenize(self.context.encode("utf-8")))+self.context_size>self.model_data["params"]["n_ctx"]:
            self.context_index+=self.memory
            self.context_summary = self.summarize_context() #summarizes the past
        

        # Concatenate chat history and system template
        examples = [
            model_config["chat_template"].format(role=model_config["mapper"][ex["role"]],content=ex["content"])
                for ex in assistant_template["examples"] if ex["role"] and ex["content"]]+[self.context_summary]+[
            model_config["chat_template"].format(role=chat_mapper[ex["role"]],content=ex["content"])
                for ex in self.messages[self.context_index:]
            ] 
            
        instruction = model_config["instruction"].format(name=assistant_template["name"],user=self.user)
        persona = f"{assistant_template['background']} {assistant_template['personality']}"
        context = "\n".join(examples)
        
        chat_history_with_template = model_config["prompt_template"].format(
            context=context,
            instruction=instruction,
            persona=persona,
            name=assistant_template["name"],
            user=self.user,
            prompt=prompt
            )

        return chat_history_with_template
    
    def summarize_context(self):
        model_config = self.model_data["config"]
        assistant_template = self.character_data["assistant_template"]
        chat_mapper = {
            self.user: model_config["mapper"]["USER"],
            assistant_template["name"]: model_config["mapper"]["CHARACTER"]
        }
        history = "\n".join([self.context_summary] + [
            "{role}: {content}".format(role=chat_mapper[ex["role"]],content=ex["content"])
            for ex in self.messages[-self.memory:]
        ])
        num = int(np.sqrt(self.memory))+1

        completion = get_summary(history,num_sentences=num)
        print(completion)
        return completion

    # Define a method to convert text to speech
    def text_to_speech(self, text):
        tts_audio = generate_speech(text,method=self.character_data["tts_method"], speaker=self.name, device=config.device)
        output_audio = vc_single(input_audio=tts_audio,**self.voice_model,**self.character_data["tts_options"])
        return output_audio

    # Define a method to run the STT and TTS in the background and be non-blocking
    def speak_and_listen(self):
        assert self.loaded, "Please load the models first"
        import speech_recognition as sr
        
        # Create a speech recognizer instance
        self.stt_models = load_stt_models(self.stt_method) #speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 2000
        self.recognizer.pause_threshold = 2.
        self.is_recording = True
        
        # Start listening to the microphone in the background and call the callback function when speech is detected
        self.autoplay = False
        
        # Create a microphone instance
        # Start listening to mic
        self.listener = threading.Thread(target=self.microphone_callback,daemon=True,name="listener")
        self.listener.start()
        # self.stop_listening = self.recognizer.listen_in_background(sr.Microphone(sample_rate=self.sample_rate), self.microphone_callback)
        

    # Define a callback function that will be called when speech is detected
    def microphone_callback(self):
        import speech_recognition as sr
        with sr.Microphone(sample_rate=self.sample_rate) as source, self.lock:
            self.recognizer.adjust_for_ambient_noise(source, duration=self.recognizer.pause_threshold)

        while self.is_recording:
            with sr.Microphone(sample_rate=self.sample_rate) as source, self.lock:
                try:
                    audio = self.recognizer.listen(source)

                    if not self.is_recording:
                        return self.stop_listening()
                    
                    print("listening to mic...")
                    
                    prompt, input_audio = transcribe_speech(audio,stt_models=self.stt_models,stt_method=self.stt_method,denoise=True)
                    if prompt is not None and len(prompt)>1:
                        print(f"{self.name} heard: {prompt}")
                        full_response = ""
                        for response in self.generate_text(prompt):
                            full_response += response
                        output_audio = self.text_to_speech(full_response)
                        if output_audio: sd.play(*output_audio)
                        self.messages.append({"role": self.user, "content": prompt, "audio": input_audio}) #add user prompt to history
                        self.messages.append({
                            "role": self.name,
                            "content": full_response,
                            "audio": output_audio
                            })
                        print(f"{self.name} said: {full_response}")
                        sd.wait() # wait for audio to stop playing
                except Exception as e:
                    print(e)

class Recorder:
    # Initialize the character with a name and a voice
    def __init__(self, stt_method="speecht5", device=None):
        self.voice_model = None
        self.stt_models = None
        self.loaded = False
        self.sample_rate = 16000 # same rate as hubert model
        self.is_recording = False
        self.stt_method = stt_method
        self.device=device
        self.autoplay = False
        self.recognizer = None
        self.listener = None
        self.lock = threading.Lock()
        self.text=None


    def __del__(self):
        self.stop_listening()
        gc_collect()
        print("models unloaded")

    # Define a method to run the STT and TTS in the background and be non-blocking
    def start_listening(self):
        import speech_recognition as sr
        
        # Create a speech recognizer instance
        self.stt_models = load_stt_models(self.stt_method) #speech recognition
        self.recognizer = sr.Recognizer()
        # self.recognizer.energy_threshold = 2000
        self.recognizer.pause_threshold = 2.
        self.is_recording = True
        
        # Start listening to the microphone in the background and call the callback function when speech is detected
        self.autoplay = False
        
        # Create a microphone instance
        # Start listening to mic
        self.listener = threading.Thread(target=self.microphone_callback,daemon=True,name="listener")
        self.listener.start()
       
    # Define a callback function that will be called when speech is detected
    def microphone_callback(self):
        import speech_recognition as sr
        # with sr.Microphone(0,sample_rate=self.sample_rate) as source, self.lock:
        #     self.recognizer.adjust_for_ambient_noise(source, duration=self.recognizer.pause_threshold)

        while self.is_recording:
            with sr.Microphone(sample_rate=self.sample_rate) as source, self.lock:
                try:
                    sd.wait()
                    audio = self.recognizer.listen(source)

                    if not self.is_recording:
                        return self.stop_listening()
                    
                    print("listening to mic...")

                    # if len(audio.frame_data)>self.sample_rate*2:
                    input_audio = bytes_to_audio(audio.get_wav_data())
                    self.text = transcribe_speech(input_audio,stt_models=self.stt_models,stt_method=self.stt_method,denoise=True)
                    print(self.text)
                except Exception as e:
                    print(e)

    def stop_listening(self):
        if self.listener:
            print("stopped listening to mic...")
            self.is_recording = False
            self.listener.join(1)
            del self.listener, self.recognizer, self.stt_models
            self.listener = self.recognizer = self.stt_models = None