import time
import torch
import openai
import cohere

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from peft import PeftModel

class Generator:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
    
    def generate(self):
        return ""


class OpenAIGenerator(Generator):
    def __init__(self, model_name, api_key, n=8, max_tokens=512, temperature=0.7, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False):
        super().__init__(model_name, api_key)
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.wait_till_success = wait_till_success
    
    @staticmethod
    def parse_response(response):
        to_return = []
        for _, g in enumerate(response['choices']):
            text = g['text']
            logprob = sum(g['logprobs']['token_logprobs'])
            to_return.append((text, logprob))
        texts = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]
        return texts

    def generate(self, prompt):
        get_results = False
        while not get_results:
            try:
                result = openai.Completion.create(
                    engine=self.model_name,
                    prompt=prompt,
                    api_key=self.api_key,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    top_p=self.top_p,
                    n=self.n,
                    stop=self.stop,
                    logprobs=1
                )
                get_results = True
            except Exception as e:
                if self.wait_till_success:
                    time.sleep(1)
                else:
                    raise e
        return self.parse_response(result)


class CohereGenerator(Generator):
    def __init__(self, model_name, api_key, n=8, max_tokens=512, temperature=0.7, p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False):
        super().__init__(model_name, api_key)
        self.cohere = cohere.Cohere(self.api_key)
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.p = p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.wait_till_success = wait_till_success

    
    @staticmethod
    def parse_response(response):
        text = response.generations[0].text
        return text
    
    def generate(self, prompt):
        texts = []
        for _ in range(self.n):
            get_result = False
            while not get_result:
                try:
                    result = self.cohere.generate(
                        prompt=prompt,
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        p=self.p,
                        k=0,
                        stop=self.stop,
                    )
                    get_result = True
                except Exception as e:
                    if self.wait_till_success:
                        time.sleep(1)
                    else:
                        raise e
            text = self.parse_response(result)
            texts.append(text)
        return texts


class AlpacaGenerator:
    def __init__(self, base_model, lora_weights, device):
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        self.device = device
        self.n = 5
        if "cuda" in device:
            self.model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        self.model.eval()

        self.temperature=0.1
        self.top_p=0.75
        self.top_k=40
        self.num_beams=4
        self.max_new_tokens=64

    def generate(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        texts = []
        for _ in range(self.n):
            generation_config = GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                num_beams=self.num_beams,
                **kwargs,
            )
            generate_params = {
                "input_ids": input_ids,
                "generation_config": generation_config,
                "return_dict_in_generate": True,
                "output_scores": True,
                "max_new_tokens": self.max_new_tokens,
            }

            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=self.max_new_tokens,
                )
            s = generation_output.sequences[0]
            output = self.tokenizer.decode(s)
            #response = output.split('###Passage:')[1].strip()
            texts.append(output)

        return texts