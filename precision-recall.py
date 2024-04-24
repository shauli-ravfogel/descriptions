from transformers import AutoModel, AutoTokenizer

import transformers
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pickle
from typing import List
import argparse
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from InstructorEmbedding import INSTRUCTOR

METHOD = "mean"


def average_pool_e5(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_batch_fn_e5(model, tokenizer, sentences, device, prefix:str = None):
    sentences = [prefix + s for s in sentences] 
    batch_dict = tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model(**batch_dict)
    embeddings = average_pool_e5(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    return embeddings.cpu().detach().numpy()


def encode_batch_fn(model, tokenizer, sentences: List[str], device: str, prefix:str = None):
    sentences = [prefix + s for s in sentences]
    input_ids = tokenizer(sentences, padding=True, max_length=128, truncation=True, return_tensors="pt",
                          add_special_tokens=True).to(device)
    features = model(**input_ids)[0]
    features =  torch.sum(features[:,1:,:] * input_ids["attention_mask"][:,1:].unsqueeze(-1), dim=1) / torch.clamp(torch.sum(input_ids["attention_mask"][:,1:], dim=1, keepdims=True), min=1e-9)
    features = features / torch.norm(features, dim=1, keepdim=True)
    return features.cpu().detach().numpy()


def encode_batch_fn_instructor(model, tokenizer, sentences: List[str], device: str, prefix:str = None):
        sentences = [prefix + s for s in sentences]
        h =  model.encode(sentences)
        h = h / np.linalg.norm(h, axis=1, keepdims=True)
        return h

def load_sents_and_mpnet_vecs(model_name, load_pubmed=False):
    
    #X = np.load("X_mpnet-pretrained_bitfit:False_wiki-trained-wiki-only_mean_v9.npy")
    X = np.load("encodings/X_{}.npy".format(model_name))
    #X = np.load("X_mpnet-pretrained_bitfit:0_wiki-trained-wiki-only_mean_v9_snli.npy")
    #X = np.load("X_mpnet-pretrained_bitfit:0_wiki-trained-wiki-only_mean_v9_only-info:1.npy")
    
    with open("wiki_sents_10m_v2.txt", "r") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    # normalize 
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    return X, lines


def load_finetuned_model(model_type):

        if model_type == "abstract-sim":
            print("abstract sim")
            sentence_encoder = AutoModel.from_pretrained("biu-nlp/abstract-sim-sentence")
            query_encoder = AutoModel.from_pretrained("biu-nlp/abstract-sim-query")
            tokenizer = AutoTokenizer.from_pretrained("biu-nlp/abstract-sim-sentence")
        elif "mpnet" in model_type:
            print("mpnet")
            sentence_encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            query_encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        elif model_type == "e5":
            print("e5")
            tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
            sentence_encoder = AutoModel.from_pretrained('intfloat/e5-base-v2')
            query_encoder = AutoModel.from_pretrained('intfloat/e5-base-v2')
        return tokenizer, query_encoder, sentence_encoder

def get_closest_neighbor_to_vector_by_cosine_sim(vec, X, sents, k=10):
    # get the closest neighbor for a given vector
    # normalize 
    vec = vec / np.linalg.norm(vec)
    #X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sims = np.dot(X, vec)
    idx = np.argsort(sims)[::-1]
    return [sents[i] for i in idx[:k]]


args = argparse.ArgumentParser()
args.add_argument("--X_model_name", type=str, default="mpnet_bitfit:0_wiki-trained-wiki-only_mean_v9_mpnet-finetuned")
args.add_argument("--model_type", type=str, default="mpnet") 
args.add_argument("--model_weights", type=str, default=None) 
args.add_argument("--model_name", type=str, default=None) #mpnet-pretrained_final_mean_only-triplet:0_only-info:1_v2))
args.add_argument("--prefix_name", type=str, default="_mpnet-finetuned")
args.add_argument("--use_pretrained_model", action="store_true")
args.add_argument("--use_negatives", action="store_true", default=True)
args.add_argument("--pretrained", action="store_true", default=False)
args = args.parse_args()


if args.model_name == "instructor":
    prefix_sentence = "'Represent the Wikipedia document for retrieval: "
    prefix_query = "Represent the Wikipedia summary for retrieving relevant passages: "
else:
    if "hf_both" in args.prefix_name:
        prefix_query = "<query>: "
        prefix_sentence = ""

    elif "e5" in args.prefix_name:
        prefix_query = "query: "
        prefix_sentence = "passage: "
    else:

        prefix_sentence = ""
        prefix_query = ""

# prefix = ""
#X = np.load("X_mpnet-pretrained_bitfit:False_wiki-trained-wiki-only_mean_v9.npy")
#X = np.load("X_mpnet-pretrained_bitfit:0_wiki-trained-wiki-only_mean_v9_snli.npy")
#X = np.load("X_mpnet-pretrained_bitfit:0_wiki-trained-wiki-only_mean_v9_only-info:1.npy")

tokenizer, query_encoder, sentence_encoder = load_finetuned_model(args.model_type)
encode_batch = encode_batch_fn
if args.model_name is not None or (args.model_weights is not None):
 if not args.pretrained:
    print("Loading parameters", args.model_weights)
    sentence_encoder_dict = torch.load("models/sentence_encoder_{}.pt".format(args.model_weights))
    query_encoder_dict = torch.load("models/query_encoder_{}.pt".format(args.model_weights))
    sentence_encoder_dict = {k.replace("module.", ""):v for k,v in sentence_encoder_dict.items()}
    query_encoder_dict = {k.replace("module.", ""):v for k,v in query_encoder_dict.items()}

    sentence_encoder.load_state_dict(sentence_encoder_dict)
    query_encoder.load_state_dict(query_encoder_dict)
 else:
    print("Loading pretrained model")
    if "instructor" not in args.model_name:
        print("Loading e5")
        sentence_encoder = AutoModel.from_pretrained(args.model_name)
        query_encoder = AutoModel.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if "e5" in args.model_name:
            encode_batch = encode_batch_fn_e5
    else:
        print("Loading instructor")
        sentence_encoder = INSTRUCTOR('hkunlp/instructor-large')
        query_encoder = INSTRUCTOR('hkunlp/instructor-large')
        tokenizer = None
        encode_batch = encode_batch_fn_instructor

X, lines = load_sents_and_mpnet_vecs(args.X_model_name)
# with open("test_with_misleading_sentences_negative_prompt.pickle", "rb") as f:
#     test_with_misleading_sentences = pickle.load(f)

# with open("test_with_sentences.pickle", "rb") as f:
#     test_with_sentences = pickle.load(f)

# test_misleading_as_dict = {t["description"]:t["sentences"] for t in test_with_misleading_sentences}
# test_as_dict = {t["description"]:t["sentences"] for t in test_with_sentences}

# with open("test.txt", "r") as f:
#      test_lines = f.readlines()
#      test_lines = [l.strip() for l in test_lines]


# test_all = []
# for i, description in enumerate(test_lines):
#     bad = test_misleading_as_dict[description]
#     good = test_as_dict[description]
#     test_all.append({"description": description, "good": good, "bad": bad})


query_encoder.to("cuda:1")
sentence_encoder.to("cuda:1")
sentence_encoder.eval()
query_encoder.eval()

# with open("test_all_v4.pickle", "rb") as f:
#     test_all = pickle.load(f)

with open("test_all_v4_after_verification.pickle", "rb") as f:
    test_all = pickle.load(f)

eval_results = []
from collections import defaultdict
k_good = 12
for i in range(len(test_all)):
    eval_dict = defaultdict(list)

    x_good = encode_batch(sentence_encoder, tokenizer, test_all[i]["good"][:k_good], "cuda:1", prefix_sentence)
    x_bad = encode_batch(sentence_encoder, tokenizer, test_all[i]["bad"], "cuda:1", prefix_sentence)
    x_synthetic = np.concatenate([x_good, x_bad], axis=0)
    X_test_i = np.concatenate([X, x_good, x_bad], axis=0)
    lines_i = lines + test_all[i]["good"][:k_good] + test_all[i]["bad"]
    lines_synthetic = test_all[i]["good"][:k_good] + test_all[i]["bad"]

    
    query_vec = encode_batch(query_encoder, tokenizer, [test_all[i]["description"]], "cuda:1", prefix_query)[0]
    neighbors = get_closest_neighbor_to_vector_by_cosine_sim(query_vec, X_test_i, lines_i, k=10000000)
    neighbors_synthetic = get_closest_neighbor_to_vector_by_cosine_sim(query_vec, x_synthetic, lines_synthetic, k=10000000)
    
    top_10k = neighbors[:10000]
    sentences_set = set(test_all[i]["good"])
    sentences_set_bad = set(test_all[i]["bad"])
    relevant = [n for n in top_10k if n in sentences_set]
    is_relevant = np.array([1 if n in sentences_set else 0 for n in top_10k])
    is_not_relevant = np.array([1 if n in sentences_set_bad else 0 for n in top_10k])
    is_relevant_synthetic = np.array([1 if n in sentences_set else 0 for n in neighbors_synthetic])

    print(test_all[i]["description"])
    print("top 50 results: ")
    for n in neighbors[:10]:
        print(n)
        print("-------------")
    print("=========")
    eval_dict["top_50"] = neighbors[:50]
    for k in [1,2,3,4,5,8, 10,12]:
        if k > len(test_all[i]["good"]): continue
        precision_at_k = is_relevant_synthetic[:k].sum() / k
        print(f"precision@{k} = {precision_at_k}")
        eval_dict[f"precision@{k}"].append(precision_at_k)
    
    for k in [1,5,10, 12, 20,30,40,50,100,200, 500, 1000, 2000]:
        recall_at_k = is_relevant[:k].sum() / len(test_all[i]["good"])
        print(f"recall@{k} = {recall_at_k}")
        eval_dict[f"recall@{k}"].append(recall_at_k)
    for k in [1,5,10, 12, 20,30,40,50,100,200, 500, 1000, 2000]:
        recall_at_k = is_not_relevant[:k].sum() / len(test_all[i]["bad"])
        print(f"recall@{k}, bad = {recall_at_k}")
        eval_dict[f"recall@{k}_bad"].append(recall_at_k)
        
    eval_results.append({"instance": test_all[i],
                         "eval_dict": eval_dict})
    
    print("done, saving")
    with open("out5-iclr/eval_precision_recall{}{}.pickle".format("_positives+negatives" if args.use_negatives else "_positives", args.prefix_name), "wb") as f:
        pickle.dump(eval_results, f)
    print("====================================")
