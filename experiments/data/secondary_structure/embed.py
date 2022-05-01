import torch
from tqdm import tqdm
import numpy as np

from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import AlbertModel, AlbertTokenizer


def build_pretrained_model(model_name):
    if "t5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(model_name)
    elif "albert" in model_name:
        tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = AlbertModel.from_pretrained(model_name)
    elif "bert" in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = BertModel.from_pretrained(model_name)
    elif "xlnet" in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False )
        model = XLNetModel.from_pretrained(model_name)
    else:
        raise ValueError(f"Unkown model name: {model_name}")
    return tokenizer, model


ppm = "Rostlab/prot_t5_xl_uniref50"
device = 0

tokenizer, embeder = build_pretrained_model(ppm)
embeder = embeder.eval().to(device)


train_lines = open("train.seq", "r").readlines()
test_lines = open("test.seq", "r").readlines()

for idx, line in tqdm(enumerate(train_lines+test_lines)):
    _id, seq, _ = line.strip().split()
    seqs = [" ".join(seq.strip())]
    inputs = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding=True)
    inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
    seq_len = (inputs['attention_mask'][0] == 1).sum()
    with torch.no_grad():
        embedding = embeder(**inputs)
    embedding = embedding.last_hidden_state.cpu().numpy()
    assert embedding.shape[0] == 1
    embedding = embedding[0, :seq_len-1]
    assert embedding.shape[0] == len(seq)
    np.save(f"embs/{_id}.npy", embedding)