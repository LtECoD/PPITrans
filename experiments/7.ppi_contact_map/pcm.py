import os
import argparse
import random
import joblib
import torch
import numpy as np
from statistics import stdev, mean
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

import sys
sys.path.append(".")
from experiments.utils import Protein, load_model
from experiments.utils import forward_kth_translayer
from experiments.utils import lookup_embed


def load_samples(split, _dir):
    seqs = open(os.path.join(_dir, split+".seq"), "r").readlines()
    cms = np.load(os.path.join(_dir, split+".npy"), allow_pickle=True).tolist()
    samples = []
    for seq in seqs:
        fid, fseq, sid, sseq = seq.strip().split()
        fpro = Protein(fid, fseq)
        spro = Protein(sid, sseq)
        samples.append([fpro, spro, cms[f"{fid}-{sid}"]])

    return samples


def build_data(samples, threshold):
    data = []
    label = []
    for fpro, spro, pcm in samples:
        cm = pcm < threshold
        pemb = np.expand_dims(fpro.emb, axis=1) * np.expand_dims(spro.emb, axis=0)    # L x L x D

        data.append(np.reshape(pemb, (-1, pemb.shape[-1])))
        label.append(np.reshape(cm.astype(np.int32), (-1,)))

    data = np.vstack(data)
    label = np.hstack(label)

    return data, label


def evaluate(clf, data, label):
    f1s = []
    num_per_split = int(len(data) / 5)
    for idx in range(0, len(data), num_per_split):
        data_split = data[idx: idx+num_per_split]
        label_split = label[idx: idx+num_per_split]
        pred = clf.predict(data_split)
        assert len(pred) == len(label_split)
        f1s.append(f1_score(y_true=label_split, y_pred=pred, zero_division=0))

    return mean(f1s), stdev(f1s), min(f1s), max(f1s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--self_dir", type=str, default="./experiments/7.ppi_contact_map")
    parser.add_argument("--model_dir", type=str, help="saved ppi model")
    parser.add_argument("--threshold", type=float, default=15)
    args = parser.parse_args()
    random.seed(args.seed) 

    protein_dir = os.path.join(args.self_dir, "data")
    pretrained_emb_dir = os.path.join(protein_dir, "embs")

    model_name = os.path.basename(args.model_dir)
    save_dir = os.path.join(args.self_dir, 'save', model_name)
    results_dir = os.path.join(args.self_dir, 'results', model_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_samples = load_samples('train', protein_dir)
    test_samples = load_samples('test', protein_dir)

    # 加载模型
    model = load_model(args.model_dir)

    ##### 测试pretrained-embedding    
    emb_results = {}
    # load embedding
    if not hasattr(model.encoder, "embeder"):
        for (fpro, spro, _) in train_samples + test_samples:
            fpro.set_emb(np.load(os.path.join(pretrained_emb_dir, fpro.name+".npy")))
            spro.set_emb(np.load(os.path.join(pretrained_emb_dir, spro.name+".npy")))
    else:
        for (fpro, spro, _) in train_samples + test_samples:
            fpro.set_emb(lookup_embed(fpro, model.encoder.embeder))
            spro.set_emb(lookup_embed(spro, model.encoder.embeder))

    train_data, train_label = build_data(train_samples, args.threshold)
    test_data, test_label = build_data(test_samples, args.threshold)

    print(f">>>>{model_name}: train pcm classifier for pretrained embedding")
    model_ckpt_fp = os.path.join(save_dir, f"emb.ckpt")
    if os.path.exists(model_ckpt_fp):
        clf = joblib.load(model_ckpt_fp)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
        clf.fit(train_data, train_label)
        joblib.dump(clf, model_ckpt_fp)

    emb_results = evaluate(clf, test_data, test_label) 
    emb_result_fp = os.path.join(results_dir, f'emb.eval')
    with open(emb_result_fp, "w") as f:
        f.write("\t".join(list(map(str, emb_results))))
    
    # forward projecter
    for (fpro, spro, _) in train_samples + test_samples:
        femb = model.encoder.forward_projecter(torch.Tensor(fpro.emb).unsqueeze(0))
        fpro.set_emb(femb.detach().squeeze(0).numpy())
        semb = model.encoder.forward_projecter(torch.Tensor(spro.emb).unsqueeze(0))
        spro.set_emb(semb.detach().squeeze(0).numpy())
    

    for k in range(model.encoder.transformer.num_layers + 1):
        # build dataset
        train_data, train_label = build_data(train_samples, args.threshold)
        test_data, test_label = build_data(test_samples, args.threshold)

        print(f">>>>{model_name} train pcm classifier for {k}th layer")
        
        model_ckpt_fp = os.path.join(save_dir, f"{k}.ckpt")
        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
            clf.fit(train_data, train_label)
            joblib.dump(clf, model_ckpt_fp)
        enc_kth_results = evaluate(clf, test_data, test_label)

        if k < model.encoder.transformer.num_layers:
            for (fpro, spro, _) in train_samples + test_samples:
                fpro.set_emb(forward_kth_translayer(model, fpro.emb, k))
                spro.set_emb(forward_kth_translayer(model, spro.emb, k))

        enc_kth_result_fp = os.path.join(results_dir, f'{k}.eval')
        with open(enc_kth_result_fp, "w") as f:
            f.write("\t".join(list(map(str, enc_kth_results))))