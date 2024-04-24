import numpy as np
# import pca
from sklearn.decomposition import PCA
import pickle
import random
from datasets import load_dataset
import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import random
import spacy
import os
import openai
import tqdm
import time
from sklearn.utils import shuffle
import argparse



def get_generation(prompt, sentence_query, n=1, description_query= None, args=None):
    if not args.chatgpt:

        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt.format(sentence=sentence_query) if description_query is None else prompt.format(sentence=sentence_query, description=description_query),
        temperature=np.random.random()*0.7,
        max_tokens=300,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\"\"\""],
        n=n
        )

        return [(response["choices"][i]["text"]) for i in range(len(response["choices"]))] 
    else:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                            messages=[{"role": "user", 
                                    "content": prompt.format(sentence=sentence_query) if description_query is None else prompt.format(sentence=sentence_query, description=description_query),}]
        )
  
        return [(response["choices"][i]["message"]["content"]) for i in range(len(response["choices"]))] 


parser = argparse.ArgumentParser()
parser.add_argument("--api_key", type=str, default="")
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--pubmed", action="store_true", default=False)
parser.add_argument("--wiki", action="store_true", default=True)
parser.add_argument("--bookscorpus", action="store_true", default=False)
parser.add_argument("--chatgpt", action="store_true", default=False)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

if args.pubmed:
    with open("pubmed.txt", "r") as f:
        pubmed_data = f.readlines()
        pubmed_data = [s.strip() for s in pubmed_data]
elif args.wiki:
    with open("wiki_sents_10m_v2.txt", "r") as f:
        wiki_data = f.readlines()
    wiki_data = [d.strip() for d in wiki_data]
    good_idx = np.array([i for i in range(len(wiki_data)) if len(wiki_data[i].split(" ")) > 12 and len(wiki_data[i].split(" ")) < 40])
    wiki_data = [wiki_data[i] for i in good_idx]

with open("sents_bookscorpus_shuffled_1m.txt", "r") as f:
    book_data = f.readlines()
    book_data = [s.strip() for s in book_data]

device =  args.device
prompt_abstract = "Sentence: in spite of excellent pediatric health care , several educational problems could be noted in this tertiary pediatric center .\nDescription: Despite having advanced healthcare resources, certain deficiencies in education were identified at a medical center that serves children.\nA very abstract description: The provision of care at a specialized medical center was not optimal in one particular area, despite the presence of advanced resources.\nSentence: {sentence}\nDescription: {description}\nA very abstract description:"

results = []
n=1


with open("sims_{}.pickle".format("wiki" if args.wiki else "pubmed"), "rb") as f:
    train = pickle.load(f)

prompt = prompt_abstract
data = train

random.seed(0)


for i in tqdm.tqdm(range(len(train))):
    if "abstract_description" in data[i] and len(data[i]["abstract_description"]) > 0:
        continue
    if data[i]["dataset"] != "pubmed":
        continue
    for query in data[i]["good"][:3]:
        sentence = data[i]["sentence"]
        bad = data[i]["bad"][0]
        try:
            generations = get_generation(prompt, sentence, None, n, description_query=query)
            abstract_description = generations[0]
            # remove trailing spaces
            abstract_description = abstract_description.strip()
            # add <query>: 
            abstract_description = "<query>: "+ abstract_description

            if "abstract_description" in data[i]:
                data[i]["abstract_description"].append(abstract_description)
            else:
                data[i]["abstract_description"] = [abstract_description]
        except Exception as e:
            print("exception", e)
            time.sleep(5)
            continue
        

        if i % 10 == 0:
            print(data[i])
            print("=====================================")
    if i % 50 == 0 and i > 0:
        with open("sims_with_abstract_{}.pickle".format("pubmed" if args.pubmed else "wiki" if args.wiki else "bookcorpus"), "wb") as f:
            pickle.dump(data, f)


