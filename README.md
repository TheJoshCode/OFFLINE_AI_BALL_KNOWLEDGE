# ---------------------------------------------------------------------------------------
# WELCOME TO MY ON-DEVICE AGENT BALL KNOWLEDGE
# ---------------------------------------------------------------------------------------
    
    this is basically my running list of goated models and workflows for agents n stuff
    
    i know alot of you probably know ways to make things faster or better models then what im listing here
    
    dont care
    
    this isnt all i know, just like the main ball knowledge for my agent stacks
    
    its unorganized af
    
    also, yes ik theres probably stuff where u could cd into a dir or something and itd be cleaner, this again, is my ball knowledge, cope and seeth

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# SYSTEM PROMPTING SAUCE
# ---------------------------------------------------------------------------------------

    to make an llm talk A LOT less, just end the sys prompt with "RESPOND IN A SHORT PHRASE."

    to make an llm yap, just add "BE VERBOSE."

    any other prompting knowledge basically is the same as talking to people
    
    the best system prompters are manipulative and snakey people IRL, just saying.

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# TRANSCRIPTIONS SAUCE
# ---------------------------------------------------------------------------------------

    use whisper.cpp's whisper-server, do http requests for the transcriptions
```bash
all u gotta do is this
    
git clone https://github.com/ggml-org/whisper.cpp --recursive
    
cmake -B build -DGGML_CUDA=1
cmake --build build -j $(( $(nproc) * 75 / 100 )) --config Release
```
    the nproc times 3/4 is to use 3/4 of ur cpu cores to build, otherwise itll annihalte ur cpu, C++ DNGAF about ur cpu

```bash
if u get the nvidia cuda toolkit not found error:

sudo apt install nvidia-cuda-toolkit
```    
```bash
now u gotta download the models
chmod u+x ./whisper.cpp/models/download-ggml-models.sh
    
./models/download-ggml-model.sh
```
    run it once and itll tell u what models r available, as of me typing this out its:
    
    Available models:
      tiny tiny.en tiny-q5_1 tiny.en-q5_1 tiny-q8_0
      base base.en base-q5_1 base.en-q5_1 base-q8_0
      small small.en small.en-tdrz small-q5_1 small.en-q5_1 small-q8_0
      medium medium.en medium-q5_0 medium.en-q5_0 medium-q8_0
      large-v1 large-v2 large-v2-q5_0 large-v2-q8_0 large-v3 large-v3-q5_0 large-v3-turbo large-v3-turbo-q5_0 large-v3-turbo-q8_0
    
    anything smaller than large-v2 is practically unusable, so just stick to that and on, i use large-v3-turbo-q8_0 for most of my agents. it uses about a gig of memory

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# LANGUADE MODEL SAUCE
# ---------------------------------------------------------------------------------------
    
    use https://github.com/ggml-org/llama.cpp
    
    if u dont need it to be able to examine images or video, use an llama
    
    if u want it to be "multi-modal", which just means it can open pngs or videos and understand them like u copy pasted text to it, use a VLM.
    
```bash
for LLMs, my GOAT rn is:
    
https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/
```
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!THIS IS LIKE UNFATHOMABLY IMPORTANT BRO, READ WHAT IM ABOUT TO TELL YOU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    THE STUFF AT THE END OF THE NAME IS THE QUANTIZATION TYPE
    
    THIS WILL DRASTICALLY CHANGE HOWU MUCH VRAM IT USES 
    
    im not gonna type out how much vram each quant uses, bc that depends on the model and alot of factors
    
    but the main thing to know is q8 uses more vram than q7, and so on.
    
    the trade off with smaller quants is less accuracy, so more stupid basically.
```bash
best way i can explain quantization is with pixels
    
fp16 or bf16 are the "native resolution", imma use 4k for this example.
    
q8 is 1080p
    
q4 is like 480
    
the picture is the same, but q4 has alot less data to show the same overall image.
    
so when u ask q4 the same question as fp16 version, itll know the overall idea of what youre asking about, missing random stuff here and there.
```
# ---------------------------------------------------------------------------------------
# MAIN STUFF THAT REDUCES VRAM USAGE WITH LLMs
# ---------------------------------------------------------------------------------------

    start with f16, see if thats within ur range

    if not, get a slightly smaller quant size

    if ur alr at q4, dont go smaller, its like almost unusable UNLESS THE MODEL WAS TRAINED NATIVE 1 BIT, THOSE DO EXIST

    if at q4, start reducing the context size. llamacpp keeps past chat messages, images, docs etc in the context window. so making it smaller reduces how much past info, thus the "context", it can hold, which is the main thing that reduces memory usage

    the model repos will list the max ctx size, or you can just buff it up and llamacpp will autoadjust and give u a warning message that says the max ctx size

    if that still isnt small enough, you can start lowering --n-gpu-layers.

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# VLM SAUCE (PRETTY MUCH MILKTOAST RN SRRY, DONT WANNA TYPE IT ALL OUT)
# ---------------------------------------------------------------------------------------
```bash
MY GOAT FOR VLMS IS https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF
```
    THE MAIN BALL KNOWLEDGE FOR VLMS THAT DOESNT APPLY TO LLMS TOO:

    treat images and videos the same way u do with copypasting scripts into GPT

    bigger image = more processing time and more memory usage

    if ur making a video agent that like works on a surveillance camera or something, the first thing id do alongside the LLM advice is reducing the video size

    dont compress the image at all, that makes no difference, same amount of pixels, only makes the VLM more stupid and less accurate.

    some vlms have dynamic_pixels, which auto scales down videos and images

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# TTS MODEL SAUCE
# ---------------------------------------------------------------------------------------

    if ur making gooner apps or ASMR stuff, use https://huggingface.co/hexgrad/Kokoro-82M
    
    ik c++ is faster, but their python lib is solid af
```bash
U NEED CUDA TORCH
```
```bash
pip install uv

uv pip install kokoro

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```bash
EXAMPLE SCRIPT FOR YALL
```

```bash
from kokoro import KPipeline
import torch

pipeline = KPipeline(lang_code='a')

def generate_tts(text: str, voice: str = 'af_heart'):
    generator = pipeline(text, voice=voice)
    # Get the first segment
    for _, _, audio in generator:
        return audio, 24000
```
# ---------------------------------------------------------------------------------------
# VOICE CLONING SAUCE
# ---------------------------------------------------------------------------------------
    
    CHATTERBOX-TTS's TURBO IS LIKE AS GOOD OR BETTER THAN ELEVENLABS IF THAT GIVES U AN IDEA
    
    I USED IT IN MY OWN ONDEVICE APP, VOICEBOX:
    
    https://github.com/TheJoshCode/VoiceBox

------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    ANYWAY, HERES HOW U INSTALL CHATTERBOX
    
    pip install chatterbox-TTS
    
    pip uninstall torch torchaudio torchvision
    
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    CHATTERBOX INSTALLS CPU TORCH FOR SOME REASON EVEN ON LINUX, SO MAKE SURE U RERUN CUDA TORCH AFTER INSTALLING CHATTERBOX-TTS
```bash
EXAMPLE SCRIPT:
```
```bash
import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cuda")

def generate_tts(text: str, audio_prompt_path: str):
    wav = model.generate(text, audio_prompt_path=audio_prompt_path)
    return wav, model.sr
```
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
    
    THERES LIKE 5 MILLION OTHER THINGS I COULD WRITE HERE, ILL UPDATE THIS SOMETIME TO ADD MORE BALL KNOWLEDGE, FOR NOW THIS IS IT
    
    SO YEA 
    
    THATS ABOUT IT 
    
    SEE YA

# ---------------------------------------------------------------------------------------
# THIS BALL KNOWLEDGE WAS PROVIDED BY THEJOSHCODE
# ---------------------------------------------------------------------------------------
