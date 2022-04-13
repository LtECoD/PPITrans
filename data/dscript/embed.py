import argparse
import re
import torch
import torch.nn as nn
import numpy as np
import os


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--device", type=int, nargs="+")
    parser.add_argument("--processed_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    # load model
    print(f">>>>> load pretrained language model {args.pretrained_model}")
    tokenizer, embeder = build_pretrained_model(args.pretrained_model)
    embeder = embeder.eval()
    if len(args.device) == 1:
        embeder = embeder.to(args.device)
    elif len(args.device) > 1:
        embeder = embeder.to(args.device[0])
        embeder = nn.DataParallel(module=embeder, device_ids=args.device)
    else:
        raise NotImplementedError

    # 读取seqs中的文件
    seq_dir = os.path.join(args.processed_dir, "seqs")
    emb_dir = os.path.join(args.processed_dir, "embs")
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)

    for seqfn in os.listdir(seq_dir):
        
        sub_emb_dir = os.path.join(emb_dir, os.path.splitext(seqfn)[0])
        if not os.path.exists(sub_emb_dir):
            os.mkdir(sub_emb_dir)
        # 记录下已经编码的蛋白质
        existing_pros = [os.path.splitext(pro_fn)[0] for pro_fn in os.listdir(sub_emb_dir)]

        samples = []
        with open(os.path.join(seq_dir, seqfn), "r") as f:
            for line in f.readlines():
                pro, seq = line.strip().split("\t")
                seq = re.sub(r"[UZOB]", "X", seq)
                samples.append((pro, seq))
        print(f">>>>> Processing {os.path.splitext(seqfn)[0]}: {len(samples)} proteins")

        def process_buffer():
            pros = [s[0].strip() for s in buffer]
            seqs = [" ".join(s[1]) for s in buffer]

            inputs = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding=True)

            inputs = {k: torch.tensor(v).to(args.device[0]) for k, v in inputs.items()}
            with torch.no_grad():
                embedding = embeder(**inputs)
            embedding = embedding.last_hidden_state.cpu().numpy()
            assert len(seqs) == len(pros) == len(embedding)

            for idx in range(len(embedding)):
                seq_len = (inputs['attention_mask'][idx] == 1).sum()
                seq_emb = embedding[idx][:seq_len-1]
                assert seq_len - 1 == len(seqs[idx].strip().split())
                np.save(os.path.join(sub_emb_dir, pros[idx]+".npy"), seq_emb)
                existing_pros.append(pros[idx])

        buffer = []
        processed_num = 0
        for idx, (pro, seq) in enumerate(samples):
            if pro in existing_pros:
                continue
            buffer.append((pro, seq))

            if len(buffer) >= args.batch_size:
                process_buffer()
                processed_num += len(buffer)
                buffer = []


        if len(buffer) > 0:
            process_buffer()
            processed_num += len(buffer)
            buffer = []

        print(f">>>>> Processed {processed_num} proteins. \
            Total {len(os.listdir(sub_emb_dir))} proteins in {sub_emb_dir}.")
        
        
    