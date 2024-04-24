

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
from pytorch_metric_learning import miners, losses
from datasets import load_dataset, concatenate_datasets


# import the mpnet model from hugginface
from transformers import AutoModel, AutoTokenizer
import torch
from pytorch_metric_learning import losses, miners, distances, reducers, testers
import pickle

# args
import argparse



def get_info_nce_loss(sentence_vecs, positive_vecs, temp=0.5):
    # implementaiton of the info-nce loss from the paper "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments"
    # https://arxiv.org/abs/1807.05520
    # the loss is the negative log probability of the positive example given the sentence vector
    # the positive example is the closest vector in the positive_vecs to the sentence vector
    
    # takae the dot products for the denom between corresponding vectors
    dot_products = torch.einsum("ij,ij->i", sentence_vecs, positive_vecs) # this is a vector
    denom =  dot_products/temp
    num = torch.log(torch.sum(torch.exp(dot_products/temp)))
    losses = num - denom
    return losses.mean()


def fix_module_prefix_in_state_dict(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

def fix_module_prefix_in_state_dict_electra(state_dict):
    return {k.replace('electra.', ''): v for k, v in state_dict.items()}


def calculate_hard_batch_triplet_loss(anchors, positives, margin = 0.75, cosine_distnace = True):
    # a torch implementation of the triplet loss. Finds in-batch hard negatives.

    # first step: find negatives
    # calculate the distance between each anchor and each positive
    if cosine_distnace:
        dist = distances.CosineSimilarity()
    else:
        dist = distances.LpDistance()
    dist_mat = dist(anchors, positives)
    if cosine_distnace:
        # the cosine similarity is between -1 and 1, so we need to convert it to a distance
        dist_mat = 1 - dist_mat
    # find the hardest negative for each anchor. The hardest negative is the second closest vector (the closest is the positive)
    hard_negatives = torch.topk(dist_mat, k=2, dim=1, largest=False)[1][:,1]
    # get the hardest negative for each anchor
    hard_negatives = positives[hard_negatives]
    # calculate the triplet loss
    loss = torch.nn.functional.relu(dist(anchors, positives) - dist(anchors, hard_negatives) + margin)
    return loss.mean()

def encode_batch(model, tokenizer, sentences, device, pooling = "mean"):
    input_ids = tokenizer(sentences, padding=True, max_length=128, truncation=True, return_tensors="pt",
                          add_special_tokens=True).to(device)
    features = model(**input_ids)[0]

    if pooling == "mean":
        features =  torch.sum(features[:,1:,:] * input_ids["attention_mask"][:,1:].unsqueeze(-1), dim=1) / torch.clamp(torch.sum(input_ids["attention_mask"][:,1:], dim=1, keepdims=True), min=1e-9)
    elif pooling == "cls":
        features = features[:,0,:]
    elif pooling == "mean+cls":
        # concatenate the mean and cls features. when calcualting the mean, ignore the cls token.
        mean_features = torch.sum(features[:,1:,:] * input_ids["attention_mask"][:,1:].unsqueeze(-1), dim=1) / torch.clamp(torch.sum(input_ids["attention_mask"][:,1:], dim=1, keepdims=True), min=1e-9)
        cls_features = features[:,0,:]
        features = torch.cat([mean_features, cls_features], dim=1)
        
    return features


# args

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="all-distilroberta-v1")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--pooling", type=str, default="mean")
parser.add_argument("--load_existing", type=bool, default=False)
parser.add_argument("--linear_is_unit", type=bool, default=True)
parser.add_argument("--late_interaction", type=bool, default=False)
parser.add_argument("--hard_negatives_weight", type=float, default=0.1)
parser.add_argument("--hard_negatives_margin", type=float, default=1.0)
parser.add_argument("--gating", type=bool, default=False)
parser.add_argument("--one_encoder", type=bool, default=False)
parser.add_argument("--only_wiki", type=int, default=1)
parser.add_argument("--only_pubmed", type=int, default=0)
parser.add_argument("--bitfit", type=int, default=0)
parser.add_argument("--update_last_layer", type=int, default=0)


args = parser.parse_args()

train_wiki = load_dataset("XXXX-2-XXXX-1/XXXX-9", split="train")
train_pubmed = load_dataset("XXXX-2-XXXX-1/XXXX-9-pubmed", split="train")

if args.only_wiki:
    train = train_wiki
elif args.only_pubmed:
    train = train_pubmed
else:
    train = concatenate_datasets([train_wiki, train_pubmed])

print("len data: {}".format(len(train)))

if args.model_type == "mpnet":
    sentence_encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    query_encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

elif args.model_type == "mpnet-pretrained":

    sentence_encoder = AutoModel.from_pretrained("microsoft/mpnet-base")
    query_encoder = AutoModel.from_pretrained("microsoft/mpnet-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")

elif args.model_type == "pubmed-bert":

    sentence_encoder = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    query_encoder = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

if args.model_type == "roberta":
    sentence_encoder = AutoModel.from_pretrained("roberta-base")
    query_encoder = AutoModel.from_pretrained("roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

if args.model_type == "albert":

    sentence_encoder = AutoModel.from_pretrained("albert-xxlarge-v2")
    query_encoder = AutoModel.from_pretrained("albert-xxlarge-v2")
    tokenizer = AutoTokenizer.from_pretrained("albert-xxlarge-v2")

elif args.model_type == "electra":
    sentence_encoder = AutoModel.from_pretrained("google/electra-base-discriminator")
    query_encoder = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
    tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")

elif args.model_type == "all-distilroberta-v1":

    sentence_encoder = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1")
    query_encoder = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")


embedding_shape = sentence_encoder.config.hidden_size
linear_query = torch.nn.Linear(embedding_shape, embedding_shape)
linear_sentence = torch.nn.Linear(embedding_shape, embedding_shape)

optimizer = torch.optim.Adam(list(sentence_encoder.parameters()) + list(query_encoder.parameters()) + list(linear_query.parameters())
+list(linear_sentence.parameters()), lr=0.25*1e-4, weight_decay = 1e-6)

if args.linear_is_unit:
    linear_query.weight.data = torch.eye(embedding_shape)
    linear_sentence.weight.data = torch.eye(embedding_shape)
    linear_query.bias.data = torch.zeros(embedding_shape)
    linear_sentence.bias.data = torch.zeros(embedding_shape)
    linear_query.requires_grad = False
    linear_sentence.requires_grad = False
else:
    linear_query.requires_grad = True
    linear_sentence.requires_grad = True 


update_embeddings_in_bitfit = False
freeze_sentence_encoder = False
update_only_last_layers = False
freeze_all = False
freeze_embeddings = False
one_encoder=  args.one_encoder


if update_only_last_layers:
    for param in sentence_encoder.parameters():
        param.requires_grad = False
    for param in query_encoder.parameters():
        param.requires_grad = False
    for param in sentence_encoder.encoder.layer[-1].parameters():
        param.requires_grad = True
    for param in query_encoder.encoder.layer[-1].parameters():
        param.requires_grad = True
if args.bitfit: # update only bias terms
    params_and_names = [(name, param) for name, param in sentence_encoder.named_parameters()]
    for name,p in params_and_names:

        # freeze parameters which are not bias terms and not embeddings
        # check if embedding
        vocab_size = tokenizer.vocab_size
        if len(p.shape) > 1 and ((not update_embeddings_in_bitfit) or (update_embeddings_in_bitfit and (not p.shape[0] == vocab_size))):
            p.requires_grad = False
        else:
            p.requires_grad = True

        # if it's layer 11, then freeze all parameters
        if args.update_last_layer and "layer.11" in name:
            print("Updating last layer")
            p.requires_grad = True
    

    for p in query_encoder.parameters():
        if len(p.shape) > 1:
            p.requires_grad = False
        else:
            p.requires_grad = True
        # if it's layer 11, then freeze all parameters
        if args.update_last_layer and "layer.11" in name:
            p.requires_grad = True

for p in sentence_encoder.parameters():
    print(p.requires_grad)

if freeze_sentence_encoder:
    for p in sentence_encoder.parameters():
        p.requires_grad = False

if freeze_all:
    for p in sentence_encoder.parameters():
        p.requires_grad = False
    for p in query_encoder.parameters():
        p.requires_grad = False



print("trainin on {} examples".format(len(train)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentence_encoder= torch.nn.DataParallel(sentence_encoder)
query_encoder= torch.nn.DataParallel(query_encoder)
sentence_encoder.to(device)
query_encoder.to(device)
sentence_encoder.train()
query_encoder.train()
linear_query.to(device)
linear_sentence.to(device)

sentence_encoder.train()
query_encoder.train()

def record_negatives(anchors, positives, negative_idx):
    pos2neg = []
    for i,neg_ind in enumerate(negative_idx):
        pos2neg.append((anchors[i], positives[i], positives[neg_ind]))
    random.shuffle(pos2neg)
    for i in range(10):
        print("sentence: {}".format(pos2neg[i][0]))
        print("positive: {}".format(pos2neg[i][1]))
        print("negative: {}".format(pos2neg[i][2]))
        print("---------------------")


if freeze_embeddings: # feeze the word embeddings
    for p in sentence_encoder.parameters():
        # freeze parameters which are not bias terms and not embeddings
        # check if embedding
        vocab_size = tokenizer.vocab_size
        if len(p.shape) > 1 and (p.shape[0] == vocab_size):
            p.requires_grad = False
            print("Freezed word embedddings")
        else:
            p.requires_grad = True
else:
    for p in sentence_encoder.parameters():
        p.requires_grad = True

    for p in query_encoder.parameters():
        p.requires_grad = True

if args.one_encoder:
    query_encoder = sentence_encoder
    linear_query = linear_sentence

# info-nce loss

miner = miners.BatchHardMiner()
loss_func = losses.NTXentLoss(temperature=1.0)
#loss_fn = losses.NPairsLoss()
mean_loss = []
mean_loss_batch_hard = []
mean_loss_gpt_hard = []
dist = torch.nn.PairwiseDistance(p=2)
best_loss = 1000
triplet_loss = torch.nn.TripletMarginLoss(margin=args.hard_negatives_margin, p=2)

for epoch in range(args.epochs):
    for i in range(0, len(train), args.batch_size):

        # random batch 
        batch_idx = random.sample(range(len(train)), args.batch_size)
        batch = [train[i] for i in batch_idx]
        optimizer.zero_grad()
        loss = 0
        neg_idx = random.sample(range(len(train)), args.batch_size)

        anchors = [element["sentence"] for element in batch]
        positives = [random.choice(element["good"]) for element in batch]
        gold_negatives = [random.choice(element["bad"]) for element in batch]


        if freeze_all: # no need to take gradients
            with torch.no_grad():
                sentence_features = encode_batch(sentence_encoder, tokenizer, anchors, device, pooling=args.pooling)
                positive_features = encode_batch(query_encoder, tokenizer, positives, device, pooling=args.pooling)
                negative_features = encode_batch(query_encoder, tokenizer, gold_negatives, device, pooling=args.pooling)
        else:
            positive_features = encode_batch(query_encoder, tokenizer, positives, device, pooling=args.pooling)
            sentence_features = encode_batch(sentence_encoder, tokenizer, anchors, device, pooling=args.pooling)
            negative_features = encode_batch(query_encoder, tokenizer, gold_negatives, device, pooling=args.pooling)

        if args.late_interaction:
            cls_part = sentence_features[:,768:]
            cls_part = cls_part * positive_features
            
            if not args.gating:
                sentence_features = torch.cat([sentence_features[:,:768], cls_part], dim=1)
                sentence_features = sentence_features.reshape(sentence_features.shape[0], 2, 768)
                sentence_features,_ = torch.max(sentence_features, dim=1)
            else:
                cls_part = torch.sigmoid(cls_part)
                mean_part = sentence_features[:,:768]
                mean_part = mean_part * cls_part 
                sentence_features = mean_part


        embeddings = torch.cat([sentence_features, positive_features], dim=0)

        with torch.no_grad():
           labels = torch.arange(args.batch_size).to(device)
           batch_labels = torch.cat([labels, labels]).to(device)
           hard_pairs = miner(embeddings, batch_labels)
           hard_pairs_idx = hard_pairs[-1][len(sentence_features):] - len(sentence_features)
        

        loss = info_nce_loss(positive_features, sentence_features)
        loss += args.hard_negatives_weight * triplet_loss(sentence_features, positive_features,  negative_features)
        loss.backward()
        optimizer.step()

        mean_loss.append(loss.detach().cpu().numpy().item())

        if i % (args.batch_size * 10) == 0 and i > 0:
            print("epoch: {}, batch: {}, loss: {}".format(epoch, i, np.mean(mean_loss)))

            mean_loss_val = np.mean(mean_loss)
            mean_loss_gpt_hard_val = np.mean(mean_loss_gpt_hard)
            mean_loss_batch_hard_val = np.mean(mean_loss_batch_hard)

            mean_loss = []
            mean_loss_batch_hard = []
            mean_loss_gpt_hard = []
            record_negatives(positives, anchors, hard_pairs_idx.detach().cpu().numpy())
            #print("\n\n=====================")

            # update if not none and better
            if (mean_loss_val < best_loss) and (mean_loss_gpt_hard_val is not None) and (mean_loss_batch_hard_val is not None):
                best_loss = mean_loss_val
                print("saving model")
                torch.save(sentence_encoder.cpu().state_dict(), "sentence_encoder4_mpnet_bitfit:False_final_mean+cls_negatives:0.1_late:True_v2.pt")
                torch.save(query_encoder.cpu().state_dict(), "query_encoder4_mpnet_bitfit:False_final_mean+cls_negatives:0.1_late:True_v2.pt")
                torch.save(linear_sentence.cpu().state_dict(), "linear_query4_mpnet_bitfit:False_final_mean+cls_negatives:0.1_late:True_v2.pt")
                torch.save(linear_query.cpu().state_dict(), "linear_sentence4_mpnet_bitfit:False_final_mean+cls_negatives:0.1_late:True_v2.pt")

                sentence_encoder.to(device)
                query_encoder.to(device)
                linear_sentence.to(device)
                linear_query.to(device)
                

torch.save(sentence_encoder.cpu().state_dict(), "sentence_encoder9_{}_bitfit:{}_final_{}_negatives:{}_late:{}_v2.pt".format(args.model_type, args.bitfit, args.pooling, args.hard_negatives_weight, args.late_interaction))
torch.save(query_encoder.cpu().state_dict(), "query_encoder9_{}_bitfit:{}_final_{}_negatives:{}_late:{}_v2.pt".format(args.model_type, args.bitfit, args.pooling, args.hard_negatives_weight, args.late_interaction))
torch.save(linear_sentence.cpu().state_dict(), "linear_sentence9_{}_bitfit:{}_final_{}_negatives:{}_late:{}_v2.pt".format(args.model_type, args.bitfit, args.pooling, args.hard_negatives_weight, args.late_interaction))
torch.save(linear_query.cpu().state_dict(), "linear_query9_{}_bitfit:{}_final_{}_negatives:{}_late:{}_v2.pt".format(args.model_type, args.bitfit, args.pooling, args.hard_negatives_weight, args.late_interaction))
