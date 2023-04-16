from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional
from datasets import Features, Sequence, Value, load_dataset

from chats.prompts import generate_prompt
from miscs.utils import common_post_process, post_processes_batch, post_process_stream


import json
from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.faiss import AutoQueryEncoder

#from mesh_hack import Promptor, OpenAIGenerator, CohereGenerator, HyDE, AlpacaGenerator

import numpy as np
import faiss

import global_vars

from args import parse_args

from transformers import pipeline

FISCHER_DATA_PATH = "/root/chatbot/Alpaca-LoRA-Serve/mesh_hack/dataset_fischer.csv"

HyDE_INSTRUCTION = "Please write a passage to answer the question as a worker of Fischerwerke. Use professional language.\nQuestion: {}"

CONTEXT_SEARCH_THRESHOLD = 0.8
SUGGEST_DOCUMENTS_THRESHOLD = 0.75

def load_dataset_hyde(path=FISCHER_DATA_PATH):
    return load_dataset(
        "csv", data_files=[path], split="train", delimiter="\t", column_names=["title", "text"]
    )


def split_text(text: str, n=1024, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict, summarizer) -> dict:
    """Split documents into passages"""
    titles, texts, short_desc = [], [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    for i in range(len(texts)):
        try:
            lines = texts[i].split("\n")
            name = "Product name: " +lines[0]
            header = "Header: "+lines[1]
            desc = f"Description of {lines[0]}: "+lines[2]
            application = f"Applications of a {lines[0]}:"+lines[3].split("Applications:")[1]
            advantages = f"Advantages of a {lines[0]}:"+lines[4].split("Advantages:")[1]
            materials = f"{lines[0]} can be used with these building materials:"+lines[5].split("Materials:")[1]
            
            texts[i] = "\n".join([name, header, desc, application, advantages, materials])
            sum_out = summarizer(texts[i], max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            short_desc.append(sum_out)
        except Exception as e:
            print(texts[i])
            print(e)
            short_desc.append("text")
            continue
    return {"title": titles, "text": texts, "short_desc": short_desc}


def encode(encoder, q):
    all_emb_c = []
    for c in [q]:
        c_emb = encoder.encode(c)
        all_emb_c.append(np.array(c_emb))
    all_emb_c = np.array(all_emb_c)
    avg_emb_c = np.mean(all_emb_c, axis=0)
    hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
    return hyde_vector


def embed(documents: dict, encoder) -> dict:
    """Compute the DPR embeddings of document passages"""
    embeddings = []
    texts = documents['text']
    for doc in texts:
        embeddings.append(encode(encoder, doc))
    embeddings = np.vstack(embeddings)
    return {"embeddings": embeddings}


def process_dataset(dataset, encoder):
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
    new_features = Features(
    {"text": Value("string"), "title": Value("string"), "short_desc": Value("string")}
    )
    dataset = dataset.map(partial(split_documents, summarizer=summarizer), batched=True, batch_size=4, features=new_features)
    del summarizer
    new_features = Features(
    {"text": Value("string"), "title": Value("string"), "short_desc": Value("string"), "embeddings": Sequence(Value("float32"))}
)  # optional, save as float32 instead of float64 to save space
    dataset = dataset.map(
        partial(embed, encoder=encoder),
        batched=True,
        batch_size=16,
        features=new_features,
    )
    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)
    return dataset


def get_hyde_encoder():
    return AutoQueryEncoder(encoder_dir='facebook/contriever', pooling='mean', device='cuda')


def get_index(dataset):
    return dataset.get_index("embeddings")

def hyde_encode(query, hypothesis_documents, encoder):
    all_emb_c = []
    for c in [query] + hypothesis_documents:
        c_emb = encoder.encode(c)
        all_emb_c.append(np.array(c_emb))
    all_emb_c = np.array(all_emb_c)
    avg_emb_c = np.mean(all_emb_c, axis=0)
    hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
    return hyde_vector

def generate_passage(query, index, encoder):
    instruction = HyDE_INSTRUCTION.format(query)
    gen_prompt = partial(generate_prompt, ctx_indicator="### Input:", user_indicator="### Instruction:", ai_indicator="### Response:")
    bot_summarized_response = ''

    instruction_display = common_post_process(instruction)
    instruction_prompt, conv_length = gen_prompt(instruction, [], None)
    
    bot_response = global_vars.stream_model(
        instruction_prompt,
        max_tokens=256,
        temperature=0.25,
        top_p=0.9
    )

    return bot_response


def filter_hyde_context(query, documents, encoder):
    query_vector = encode(encoder, query)
    filtered_docs = []
    scores = []
    for i, doc in documents:
        inner_product = query_vector @ doc.T
        if inner_product > CONTEXT_SEARCH_THRESHOLD:
            filtered_docs.append((i, doc))
            scores.append(inner_product)
    
    return filtered_docs


def suggest_documents(query, encoder, index, dataset):
   
    #hypothesis_document = list(generate_passage(query, index, encoder))[-1]
    
    hyde_vector = hyde_encode(query, [], encoder)
    scores_hyde, hits_hyde = index.search(hyde_vector, k=2)
    
    scores_hyde =  scores_hyde[scores_hyde>SUGGEST_DOCUMENTS_THRESHOLD]
    hits_hyde = hits_hyde[:scores_hyde.shape[0]] 

    retrieved_documents = [
        (int(hit), dataset[int(hit)]['text'], np.array(dataset[int(hit)]['embeddings'])) for hit in hits_hyde
    ]
    return retrieved_documents

    
def convert_dict2context(d):
    start_context = "Suggest one of these products according application: \n"
    for k in d:
        current_line = ""
        current_line += k 
        current_line += ":"+",".join(d[k][:4])
        start_context += current_line + "\n"
    return start_context


if __name__ == "__main__":
    from utils import get_chat_interface

    args = parse_args()
    # global_vars.initialize_globals(args)

    # batch_enabled = global_vars.batch_enabled
    # chat_interface = get_chat_interface(global_vars.model_type, batch_enabled)
    
    dataset = load_dataset_hyde()
    encoder = get_hyde_encoder()

    dataset = process_dataset(dataset, encoder)

    total = ""
    for i in range(len(dataset)):
        total += dataset[i]['short_desc']
    print(len(total))

    total_app = ""
    for i in range(len(dataset)):
        total += dataset[i]['text'].split("\n")[3].split("Applications")[1]
    print(len(total_app))
    print(len(total))
    

    #print(list(generate_passage("Do you sell a screw with a fine thread and a drill point?", index, encoder))[-1])






