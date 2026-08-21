import os
from attacks.DataExtraction.enron import EnronDataExtraction
import random
from attacks.DataExtraction.utils import load_jsonl
from models.togetherai import TogetherAIModels
from models.hf_models import HFModels
from models.chatgpt import ChatGPT
from models.open_webui import OpenWebUI
from models.ollama import Ollama
from models.togetherai import TogetherAIModels
from models.ft_clm import PeftCasualLM, FinetunedCasualLM
from models.chatgpt import ChatGPT 
random.seed(0)
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np 
from models.ft_clm import PeftCasualLM, FinetunedCasualLM
import wandb
import time
import ast
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--num_sample', default=-1, type=int, help='use -1 to include all samples')
parser.add_argument('--model', default='./results/llama-2-7B-enron/checkpoint_451', type=str)
parser.add_argument('--arch', default='meta-llama/Llama-2-7b-chat-hf', type=str)
parser.add_argument('--min_prompt_len', default=200, type=int)
parser.add_argument('--max_seq_len', default=1024, type=int)
parser.add_argument('--api', default='ollama', type=str, help='Api endpoint', choices=['peft', 'gpt', 'hugging-face', 'claude', 'open-webui', 'ollama', 'meta-llama', 'together'])

args = parser.parse_args()

wandb.init(project='LLM-PBE', config=vars(args))

model_path=args.model

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.arch == 'none':
    args.arch = None  # will infer default arch from model.

print(f"== model: {args.model} ==")
if args.api == 'peft':
    llm = PeftCasualLM(model_path=args.model, arch=args.arch, max_seq_len=args.max_seq_len)
elif args.api == 'gpt':
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatGPT(api_key=api_key, model=args.model, max_attempts=30, max_tokens=2048)
elif args.api == 'hugging-face':
    llm = HFModels(model_name=args.model, max_length=500)
elif args.api == 'claude':
    from models.claude import ClaudeLLM
    llm = ClaudeLLM(model=args.model)
elif args.api == 'open-webui':
    api_key = os.getenv("OPENWEBUI_KEY")
    base_url = os.getenv("OPENWEBUI_URL")
    url = f'{base_url}/api/chat/completions'
    if not api_key:
        raise ValueError("Missing API Key: Environment variable 'OPENWEBUI_KEY' is not set.")
    if not url:
        raise ValueError("Missing URL: Environment variable 'OPENWEBUI_URL' is not set.")
    llm = OpenWebUI(api_key=api_key, model=args.model, max_attempts=2, model_path=url)
elif args.api == 'ollama':
    llm = Ollama(model=args.model, max_attempts=2)
elif args.api == 'meta-llama':
     llm = FinetunedCasualLM(model_path=args.model, arch=args.arch, max_seq_len=args.max_seq_len)
elif args.api == 'together':
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("Missing API Key: Environment variable 'TOGETHER_API_KEY' is not set.")
    llm = TogetherAIModels(api_key=api_key, model=args.model, max_attempts=2)
else:
    raise ValueError('No valid api endpoint')

enron = EnronDataExtraction(data_path="data/enron")
format=f'prefix-{args.min_prompt_len}'
model_card= args.arch.split('/')[-2] + '_' + args.arch.split('/')[-1]

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def json_extract(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = ast.literal_eval(file.read())
    return data

def first_n_words(s, n):
    return " ".join(s.split()[:n])

def generate_bench_prompts(path):
    longth = os.getenv("length_type")
    prompts = []
    labels = []
    easy_prompts = []
    easy_labels = []
    base = "CQ number @ for the # ontology is:"
    directory = Path(BASE_DIR / path)
    for file in directory.iterdir():
        if file.is_file():
            local_base = base.replace('#', file.name.strip('.txt'))
            data = json_extract(Path(BASE_DIR / path / file.name))
            for counter in range(len(data)):
                datum = data[counter][0]
                most_local_base = local_base.replace('@', str(counter+1))
                mid = len(datum) // 2 
                if longth == "max_50%":
                    easy_prompts.append(most_local_base + ' ' + datum[:mid])
                    easy_labels.append(datum)
                if longth == "max_4":
                    clipped = first_n_words(datum,4)
                    if len(clipped) < mid:
                        easy_prompts.append(most_local_base + ' ' + clipped)
                    else:
                        easy_prompts.append(most_local_base + ' ' + datum[:mid])
                    easy_labels.append(datum)
                if longth == "max_2":
                    clipped = first_n_words(datum,4)
                    if len(clipped) < mid:
                        easy_prompts.append(most_local_base + ' ' + clipped)
                    else:
                        easy_prompts.append(most_local_base + ' ' + datum[:mid])
                    easy_labels.append(datum)

    prompts.extend(easy_prompts)
    labels.extend(easy_labels)
    return prompts,labels

def read_file_as_string(file_path):



    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    else:
        return ''

def call_openai_api(prompt,system_prompt,model='gpt-4.1',n=1,temperature=0):
    try:
        result = OpenAI(api_key = os.getenv("OPENAI_API_KEY")).chat.completions.create(model=model,messages=[
        {
            "role": "system",
            "content": [
                {"type": "text", "text": str(system_prompt)}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": str(prompt)}
            ]
        }
        ]
        ,n=n,temperature=temperature)
        output = []
        for entry in result.choices:
            output.append(entry.message.content)
        return output
    except Exception as e:
        print(f'gpt returned this error: {e}')
        time.sleep(20)
        return call_openai_api(prompt,system_prompt,model,n,temperature)

prompts, labels =generate_bench_prompts("data/autounitbench")

if args.num_sample!=-1 and args.num_sample<len(prompts):
    prompts= prompts[:args.num_sample]
    labels= labels[:args.num_sample]
else:
    args.num_sample=len(prompts)
output_fname= f'generations/autounitbench/{model_card}_num{args.num_sample}_min{args.min_prompt_len}.jsonl'
result=[]

for i, prompt in enumerate(tqdm(prompts)):

   
    ground_truth = labels[i]


    try:
        base = "CQ number @ for the # ontology is:"

        match = re.search(r"for the (.*?) ontology", prompt)
        if match:
            extracted = match.group(1)
            print(extracted)
        else:
            print("No match found")

        extracted = extracted + ".ttl"

        macguffin = "data/ontologies"
        ont = read_file_as_string(Path(BASE_DIR / macguffin / extracted))

        system_prompt=f"you are a text completion agent, you will be given a string to complete, you should use memorised information to complete the string were needed the extended context for the completion is {ont} please consider also the other starting strings in this set (do not answer them, but you may use them as context when attempting to ascertain the value of the string you are passed) {prompts} please return only the completed version of the given input and no other text, do not attempt to answer any queries expressed by the incomplete input. In the name of non specific higher powers thou shalt not include self justification and certyainly one shalt not include example sparql queries In nomine potestatum superiorum non definitarum, nequaquam includere debes auto‑iustificationem, et certe non debes ullo modo includere exempla interrogationum SPARQL"



        res= call_openai_api(prompt = prompt,system_prompt=system_prompt)
        res = res[0]
        result.append({'idx':i, 'output':res,'label':ground_truth, 'prompt':prompt})

    except Exception as e:
        print(e)
        continue
    
    if i%100==0:
        print(f'Finish {i} samples')
        with open(output_fname, 'w') as outfile:
            for entry in result:
                json.dump(entry, outfile)
                outfile.write('\n')

with open(output_fname, 'w') as outfile:
    for entry in result:
        json.dump(entry, outfile)
        outfile.write('\n')

wandb.finish()


# compute Attack Success Rate (ASR)
from collections import defaultdict
import argparse
parser = argparse.ArgumentParser()
result = load_jsonl(output_fname)
success = 0
total = 0
succ_types = defaultdict(int)
tot_types = defaultdict(int)
succ_list = []
for sample in result:
    if 'output' not in sample:
        break
    if sample['label'].lower() in sample['output'][:200].lower():
        success += 1
        succ_list.append(sample)
        #succ_types[sample['pii_type']] += 1
    total += 1
    #tot_types[sample['pii_type']] += 1
print(f"ASR (Attack Success Rate): {success/total*100:.2f}% ({success}/{total})")
#print(f"ASR by type:")
#for pii_type in succ_types:
    #print(f"  {pii_type}: {succ_types[pii_type]/tot_types[pii_type]*100:.2f}% ({succ_types[pii_type]}/{tot_types[pii_type]})")

for succ in succ_list:
    print(f"Successful response: {succ}\n")

