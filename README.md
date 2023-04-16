# HandymanAI

Using:
- Alpaca 7B
- Contriever
- HyDE
- facebook/bart-large-cnn
- ✨Magic ✨

## Features

- Chatbot with knowledge of a Fischer products. 
- Can assist you in free style

## Architecture
![alt text](https://i.imgur.com/60DE2L3.png)

Also during inference we include retrieved products inside a context. After some threshold we replace initial product description with output of text summarization to save context space and to highlight recent products in dialog. Storing several product information as much as possible helps us to compare, tell difference or reference them during conversation.


## Installation
Sorry no docker images :(. 

Install the dependencies and devDependencies and start the server.

```
pip install -r requirements && apt-get install openjdk-11-jdk && pip install pyserini  && pip install faiss-cpu
```
Set up model base:
```
BASE_URL=decapoda-research/llama-7b-hf
FINETUNED_CKPT_URL=tloen/alpaca-lora-7b
```
Run:
```
python app.py --base_url $BASE_URL --ft_ckpt_url $FINETUNED_CKPT_URL --port YOUR_PORT
```
