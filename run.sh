wget XXXX-1 .

python3 run_gpt3.py --api_key XXX --wiki
python3 run_gpt3_abstract.py -api_key XXX --wiki

# uplaod the resulting datasets to HF, then train:

python3 train.py --model_type mpnet-pretrained --only_wiki 1 --epochs 30 --batch_size 128

# upload the resulting models to HF, then encode a corpus of sentences using the model:

python3 encode.py --model_type mpnet-pretrained --use_hf_model --batch_size 256
