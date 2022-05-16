import os
import argparse
import random
import joblib
import torch
import pickle
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
    samples = pickle.load(open(os.path.join(_dir, split+".pkl"), "rb"))
    data = []
    for seq in seqs:
        fid, fseq, sid, sseq = seq.strip().split()
        fpro = Protein(fid, fseq)
        spro = Protein(sid, sseq)
        data.append([fpro, spro, samples[f"{fid}-{sid}"]])

    return data


def build_data(samples):
    data = []
    label = []
    for fpro, spro, inst in samples:
        findices = inst[:, 0]
        sindices = inst[:, 1]

        _data = fpro.emb[findices] * spro.emb[sindices]       # l x D
        _label = inst[:, 2]

        assert _data.shape[0] == _label.shape[0]
        data.append(_data)
        label.append(_label)

    zipped = list(zip(data, label))
    random.shuffle(zipped)
    data, label = zip(*zipped)

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
    print(f"train set size: {sum([inst.shape[0] for _, _, inst in train_samples])}")
    print(f"test set size size: {sum([inst.shape[0] for _, _, inst in test_samples])}")

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

    train_data, train_label = build_data(train_samples)
    test_data, test_label = build_data(test_samples)

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
    

    for k in range(model.encoder.num_layers + 1):
        # build dataset
        train_data, train_label = build_data(train_samples)
        test_data, test_label = build_data(test_samples)

        print(f">>>>{model_name} train pcm classifier for {k}th layer")
        
        model_ckpt_fp = os.path.join(save_dir, f"{k}.ckpt")
        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
            clf.fit(train_data, train_label)
            joblib.dump(clf, model_ckpt_fp)
        enc_kth_results = evaluate(clf, test_data, test_label)

        if k < model.encoder.num_layers:
            for (fpro, spro, _) in train_samples + test_samples:
                fpro.set_emb(forward_kth_translayer(model, fpro.emb, k))
                spro.set_emb(forward_kth_translayer(model, spro.emb, k))

        enc_kth_result_fp = os.path.join(results_dir, f'{k}.eval')
        with open(enc_kth_result_fp, "w") as f:
            f.write("\t".join(list(map(str, enc_kth_results))))