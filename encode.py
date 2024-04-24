import numpy as np
# import pca
from sklearn.decomposition import PCA
from datasets import load_dataset
import tqdm
import torch
from transformers import AutoModel, AutoTokenizer, ElectraForPreTraining, ElectraTokenizerFast
import random
import spacy
import numpy as np
from info_nce import info_nce_loss
import pickle
import argparse

def encode_batch(model, tokenizer, sentences, device, pooling = "mean"):
    input_ids = tokenizer(sentences, padding=True, max_length=512, truncation=True, return_tensors="pt",
                          add_special_tokens=True).to(device)
    features = model(**input_ids)[0]

    if pooling == "mean":
        features = torch.sum(features[:,1:,:] * input_ids["attention_mask"][:,1:].unsqueeze(-1), dim=1) / torch.clamp(torch.sum(input_ids["attention_mask"][:,1:], dim=1, keepdims=True), min=1e-9)
    elif pooling == "cls":
        features = features[:,0,:]
    elif pooling == "mean+cls":
        # concatenate the mean and cls features. when calcualting the mean, ignore the cls token.
        mean_features = torch.sum(features[:,1:,:] * input_ids["attention_mask"][:,1:].unsqueeze(-1), dim=1) / torch.clamp(torch.sum(input_ids["attention_mask"][:,1:], dim=1, keepdims=True), min=1e-9)
        cls_features = features[:,0,:]
        features = torch.cat([mean_features, cls_features], dim=1)
    return features


def load_finetuned_model():


        sentence_encoder = AutoModel.from_pretrained("XXXX-2-XXXX-1/XXXX-9-sentence")
        query_encoder = AutoModel.from_pretrained("XXXX-2-XXXX-1/XXXX-9-query")
        tokenizer = AutoTokenizer.from_pretrained("XXXX-2-XXXX-1/XXXX-9-sentence")

        return tokenizer, query_encoder, sentence_encoder

def get_closest_neighbor_to_vector_by_cosine_sim(vec, X, sents, k=10):
    # get the closest neighbor for a given vector
    # normalize 
    vec = vec / np.linalg.norm(vec)
    assert (np.linalg.norm(X[0])-1)**2 < 1e-5
    
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sims = np.dot(X, vec)
    idx = np.argsort(sims)[::-1]

    return [sents[i] for i in idx[:k]]


def get_rank_in_neighbors(vecs, X, sent_inds):
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    assert (np.linalg.norm(X[0])-1)**2 < 1e-5
    #X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sims = vecs.dot(X.T)
    idx = np.argsort(sims, axis=1)[:,::-1]
    # return the rank of each sent_ind within idx
    ranks = []
    idx_lst = idx.tolist() 
    for i in range(len(sent_inds)):
        ranks.append(idx_lst[i].index(sent_inds[i]))
    return ranks

def get_closest_neighbor_to_vector_by_euclidean_distance(vec, X, sents, k=10):
    # get the closest neighbor for a given vector
    # normalize 
    sims = np.linalg.norm(X - vec, axis=1)
    idx = np.argsort(sims)

    return [sents[i] for i in idx[:k]]


# main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="mpnet-pretrained")
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--include_pubmed", type=int, default=False)
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bitfit", type=int, default=0)
    parser.add_argument("--use_hf_model", type=int, default=1)

    args = parser.parse_args()
    bitfit= args.bitfit #True if args.bitfit==1 else False
    model_type = args.model_type
    pooling = args.pooling
    include_pubmed=args.include_pubmed
    pretrained=args.pretrained

    print("args", args)

    if model_type == "roberta":
        sentence_encoder = AutoModel.from_pretrained("roberta-base")
        query_encoder = AutoModel.from_pretrained("roberta-base")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    elif model_type == "mpnet":
        sentence_encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        query_encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    elif model_type == "multi-qa-mpnet-base-dot-v1":
        sentence_encoder = AutoModel.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")
        query_encoder = AutoModel.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    elif model_type == "all-distilroberta-v1":
        sentence_encoder = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1")
        query_encoder = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")

    elif model_type == "pubmed-bert":
        sentence_encoder = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        query_encoder = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    elif model_type == "mpnet-pretrained":
        sentence_encoder = AutoModel.from_pretrained("microsoft/mpnet-base")
        query_encoder = AutoModel.from_pretrained("microsoft/mpnet-base")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
    
    if model_type == "multi-qa-mpnet-base-dot-v1":
        pooling = "cls"

    if pretrained and not args.use_hf_model:
        # load mdoel parameters
        sentence_encoder_dict = torch.load("sentence_encoder9_{}_bitfit:{}_final_mean_negatives:0.1_late:False_v2.pt".format(model_type, bitfit))
        sentence_encoder.load_state_dict(sentence_encoder_dict)

    if args.use_hf_model:
            tokenizer, query_encoder, sentence_encoder = load_finetuned_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence_encoder= torch.nn.DataParallel(sentence_encoder)
    query_encoder= torch.nn.DataParallel(query_encoder)
    sentence_encoder.to(device)
    query_encoder.to(device)
    # encode all sentences in batches.
    import tqdm
    batch_size = 256
    X = []
    sents = []
    sentence_encoder.to(device)


    with open("wiki_sents_10m_v2.txt", "r") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

    with open("pubmed.txt", "r") as f:
        pubmed_data = f.readlines()
    pubmed_data = [s.strip() for s in pubmed_data]
    if include_pubmed:
        lines =  pubmed_data[:10000000]

    for i in tqdm.tqdm(range(0, len(lines), batch_size)):
        batch = [d for d in lines[i:i+batch_size]]
        sents += batch
        with torch.no_grad():
            h = encode_batch(sentence_encoder, tokenizer, batch, device, pooling=pooling).detach().cpu().numpy()
            X.append(h)


    X = np.concatenate(X, axis=0)
    if include_pubmed:
        if not pretrained:
            np.save("X_{}_bitfit:{}_pubmed_original_mean_v9.npy".format(model_type, bitfit), X)
        else:
            np.save("X_{}_bitfit:{}_pubmed_mean_v9.npy".format(model_type, bitfit), X)
    else:
        if args.use_hf_model:
            np.save("X_{}_bitfit:{}_wiki_hf_mean_v9.npy".format(model_type, bitfit), X)
        elif pretrained:
            np.save("X_{}_bitfit:{}_wiki-trained-wiki-only_mean_v9.npy".format(model_type, bitfit), X)
        else:
            np.save("X_{}_bitfit:{}_wiki_original_mean_v9.npy".format(model_type, bitfit), X)
