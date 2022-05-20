import os
import argparse
import random
import joblib
import torch
import pickle
import numpy as np
from math import ceil
from statistics import stdev, mean
from sklearn.metrics import f1_score
# from sklearn.neural_network import MLPClassifier as CLF
from sklearn.neighbors import KNeighborsClassifier as CLF

import sys
sys.path.append(".")
from experiments.utils import Protein, load_model
from experiments.utils import forward_kth_translayer
from experiments.utils import lookup_embed


def load_samples(split, _dir):
    seqs = open(os.path.join(_dir, split+".seq"), "r").readlines()
    samples = pickle.load(open(os.path.join(_dir, split+".pkl"), "rb"))
    proteins = []
    for seq in seqs:
        _id, seq = seq.strip().split()
        pro = Protein(_id, seq)
        proteins.append([pro, samples[_id]])
    return proteins


def build_data(proteins):
    data = []
    label = []
    for pro, inst in proteins:
        # pemb = np.expand_dims(pro.emb, axis=1) * np.expand_dims(pro.emb, axis=0)    # L x L x D
        findices = inst[:, 0]
        sindices = inst[:, 1]

        _data = pro.emb[findices] * pro.emb[sindices]       # l x D
        _label = inst[:, 2]

        assert _data.shape[0] == _label.shape[0]
        data.append(_data)
        label.append(_label)

    data = np.vstack(data)
    label = np.hstack(label)

    order = np.arange(len(data))
    np.random.shuffle(order)
    return data[order], label[order]


def evaluate(clf, data, label):
    f1s = []
    num_per_split = ceil(len(data) / 5)
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
    parser.add_argument("--self_dir", type=str, default="./experiments/6.self_contact_map")
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
    print(f"train set size: {sum([inst.shape[0] for _, inst in train_samples])}")
    print(f"test set size size: {sum([inst.shape[0] for _, inst in test_samples])}")

    # 加载模型
    model = load_model(args.model_dir)

    ##### 测试pretrained-embedding    
    emb_results = {}
    # load embedding
    if not hasattr(model.encoder, "embeder"):
        for pro, _ in train_samples + test_samples:
            pro.set_emb(np.load(os.path.join(pretrained_emb_dir, pro.name+".npy")))
    else:
        for pro, _ in train_samples + test_samples:
            pro.set_emb(lookup_embed(pro, model.encoder.embeder))
    
    train_data, train_label = build_data(train_samples)
    test_data, test_label = build_data(test_samples)

    print(f">>>>{model_name}: train cm classifier for pretrained embedding")
    model_ckpt_fp = os.path.join(save_dir, f"emb.ckpt")
    if os.path.exists(model_ckpt_fp):
        clf = joblib.load(model_ckpt_fp)
    else:
        # clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
        clf = CLF()
        clf.fit(train_data, train_label)
        joblib.dump(clf, model_ckpt_fp)

    emb_results = evaluate(clf, test_data, test_label) 
    emb_result_fp = os.path.join(results_dir, f'emb.eval')
    with open(emb_result_fp, "w") as f:
        f.write("\t".join(list(map(str, emb_results))))

    # forward projecter
    for pro, _ in train_samples + test_samples:
        emb = model.encoder.forward_projecter(torch.Tensor(pro.emb).unsqueeze(0))
        pro.set_emb(emb.detach().squeeze(0).numpy())

    for k in range(model.encoder.num_layers + 1):
        # build dataset
        train_data, train_label = build_data(train_samples)
        test_data, test_label = build_data(test_samples)
        print(f">>>>{model_name} train cm classifier for {k}th layer")
        
        model_ckpt_fp = os.path.join(save_dir, f"{k}.ckpt")
        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            # clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
            clf = CLF()
            clf.fit(train_data, train_label)
            joblib.dump(clf, model_ckpt_fp)
        enc_kth_results = evaluate(clf, test_data, test_label)

        if k < model.encoder.num_layers:
            for pro, _ in train_samples + test_samples:
                pro.set_emb(forward_kth_translayer(model, pro.emb, k))

        enc_kth_result_fp = os.path.join(results_dir, f'{k}.eval')
        with open(enc_kth_result_fp, "w") as f:
            f.write("\t".join(list(map(str, enc_kth_results))))
            
