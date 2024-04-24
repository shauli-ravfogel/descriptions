import transformers
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pickle

METHOD = "mean"


def encode_batch(model, tokenizer, sentences, device, pooling = "cls"):
    input_ids = tokenizer(sentences, padding=True, max_length=512, truncation=True, return_tensors="pt",
                          add_special_tokens=True).to(device)
    features = model(**input_ids)[0]

    if pooling == "mean":
        features =  torch.sum(features[:,1:,:] * input_ids["attention_mask"][:,1:].unsqueeze(-1), dim=1) / torch.clamp(torch.sum(input_ids["attention_mask"][:,1:], dim=1, keepdims=True), min=1e-9)
    elif pooling == "cls":
        features = features[:,0,:]
    elif pooling == "mean+cls":
        mean_features = torch.sum(features[:,1:,:] * input_ids["attention_mask"][:,1:].unsqueeze(-1), dim=1) / torch.clamp(torch.sum(input_ids["attention_mask"][:,1:], dim=1, keepdims=True), min=1e-9)
        cls_features = features[:,0,:]
        features = torch.cat([mean_features, cls_features], dim=1)        
    return features


def load_sents_and_mpnet_vecs(load_pubmed=False):
    
    X = np.load("X_mpnet_bitfit:False_wiki_original_mean_v9.npy")
    #X = np.load("X_mpnet-pretrained_bitfit:1_wiki-trained-wiki-only_mean_v9.npy")
    
    #X = np.load("X_mpnet_bitfit:False_wiki_original_{}_v2.npy".format('mean'))
    # normalize 
    #X = X / np.linalg.norm(X, axis=1, keepdims=True)
    # with open("wiki_sents_10m.txt", "r") as f:
    #     lines = f.readlines()
    # lines = [l.strip() for l in lines]
    # lines = [s for s in lines if len(s.split()) > 5]
    # lines = lines[:5000000]


    # with open("wiki_sents_5m_v2.txt", "r") as f:
    #     lines = f.readlines()
    #     lines = [l.strip() for l in lines]

    with open("wiki_sents_10m_v2.txt", "r") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

    if load_pubmed:
        with open("pubmed.txt", "r") as f:
            pubmed_data = f.readlines()
        pubmed_data = [s.strip() for s in pubmed_data]
        lines_pubmed = pubmed_data[:5000000]
    
        lines = lines + lines_pubmed
        #X_pubmed = np.load("X_mpnet_bitfit:False_pubmed_original_mean.npy")
        # concat
        #X  = np.concatenate([X, X_pubmed], axis=0)
    # normalize 
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    #X = X[:5000000]
    #lines = lines[:5000000]
    #good_idx = np.array([i for i in range(len(lines)) if len(lines[i].split(" ")) > 5 and len(lines[i].split(" ")) < 40])
    #X = X[good_idx]
    #lines = [lines[i] for i in good_idx]
    return X, lines


def fix_module_prefix_in_state_dict(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

def load_finetuned_model(finetuned=True,bitfit=False, model_name="mpnet"):
        if model_name == "roberta":
            sentence_encoder = AutoModel.from_pretrained("roberta-base")
            query_encoder = AutoModel.from_pretrained("roberta-base")
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        elif model_name == "mpnet":
            sentence_encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            query_encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

        prefix = "_mean" if METHOD == "mean" else "_cls"
        params_sent_encoder = torch.load("sentence_encoder9_mpnet-pretrained_bitfit:False_final_mean_negatives:0.1_late:False_v2.pt")
        params_query_encoder = torch.load("query_encoder9_mpnet-pretrained_bitfit:False_final_mean_negatives:0.1_late:False_v2.pt")
        params_linear_query = torch.load("linear_query9_mpnet-pretrained_bitfit:False_final_mean_negatives:0.1_late:False_v2.pt")
        params_linear_sentence = torch.load("linear_sentence9_mpnet-pretrained_bitfit:False_final_mean_negatives:0.1_late:False_v2.pt")
        params_sent_encoder = fix_module_prefix_in_state_dict(params_sent_encoder)
        params_query_encoder = fix_module_prefix_in_state_dict(params_query_encoder)
        params_linear_query = fix_module_prefix_in_state_dict(params_linear_query)
        params_linear_sentence = fix_module_prefix_in_state_dict(params_linear_sentence)


        sentence_encoder.load_state_dict(params_sent_encoder)
        query_encoder.load_state_dict(params_query_encoder)
        linear_query = torch.nn.Linear(768, 768)
        linear_query.load_state_dict(params_linear_query)
        linear_sentence = torch.nn.Linear(768, 768)
        linear_sentence.load_state_dict(params_linear_sentence)
        #sentence_encoder.eval()
        query_encoder.eval()

        return query_encoder, linear_query, linear_sentence, tokenizer
    
def load_pretrained_mpnet():
        mpnet = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        mpnet.eval()
        return mpnet, tokenizer     


def load_finetuned_sentence_representations(load_pubmed=True):
        #X = np.load("X_mpnet_bitfit:False_wiki_and_pubmed_mean_no-negatives_v2.npy")
        #X = np.load("X_mpnet_bitfit:False_wiki_mean_v5.npy")
        X = np.load("X_mpnet-pretrained_bitfit:False_wiki-trained-wiki-only_mean_v9.npy")
        X = X[:10000000]
        # normalize
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X

def get_closest_neighbor_to_vector_by_cosine_sim(vec, X, sents, k=10):
    # get the closest neighbor for a given vector
    # normalize 
    vec = vec / np.linalg.norm(vec)
    #X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sims = np.dot(X, vec)
    idx = np.argsort(sims)[::-1]
    return [sents[i] for i in idx[:k]]

device = "cuda:0"
pretrained_sentence_reps, sentences = load_sents_and_mpnet_vecs()
query_encoder, linear_query, linear_sentence, tokenizer = load_finetuned_model()
query_encoder.to(device)

mpnet, tokenizer_pretrained = load_pretrained_mpnet()

mpnet.to(device)
finetuned_sentence_reps = load_finetuned_sentence_representations()

with open("test.txt", "r") as f:
    test_data = f.readlines()
test_data = [s.strip() for s in test_data]

eval_data = []
for query in test_data:
    print(query, "<query>:" in query)
    query_rep_orig = encode_batch(mpnet, tokenizer_pretrained, [query.replace("<query>: ", "")], device, "mean")[0].detach().cpu().numpy()
    query_rep = encode_batch(query_encoder, tokenizer, ["<query>: " + query], device, "mean")[0].detach().cpu().numpy()
    neighbors_orig = get_closest_neighbor_to_vector_by_cosine_sim(query_rep_orig, pretrained_sentence_reps, sentences, k=5)
    neighbors = get_closest_neighbor_to_vector_by_cosine_sim(query_rep, finetuned_sentence_reps, sentences, k=5)
    eval_data.append({"query": query, "results_original": neighbors_orig, "results_ours": neighbors})
    d = eval_data[-1].copy()
    d["results_original"] = d["results_original"][:5]
    d["results_ours"] = d["results_ours"][:5]
    print(d)

#with open("eval_data_ours_vs_mpnet-pretrained_full_ft.pickle", "wb") as f:
with open("eval_data_ours_vs_mpnet_full_ft.pickle", "wb") as f:

    pickle.dump(eval_data, f)