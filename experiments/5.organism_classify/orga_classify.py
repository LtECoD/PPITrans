import os
import argparse
import random
import joblib
import torch
import numpy as np
from math import ceil
from statistics import stdev, mean
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier as CLF

import sys
sys.path.append(".")
from experiments.utils import load_model
from experiments.utils import forward_kth_translayer
from experiments.utils import organisms, Protein
from experiments.utils import lookup_embed


def load_proteins(_dir, split):
    lines = open(os.path.join(_dir, split+".seq")).readlines()
    proteins = []
    for l in lines:
        name, seq = l.strip().split()
        proteins.append(Protein(name=name, seq=seq))
    return proteins


def evaluate(clf, data, label):
    """将测试集划分成5份，分别计算准确率"""

    results = defaultdict(list)
    num_per_split = ceil(len(data) / 5) 
    for idx in range(0, len(data), num_per_split):
        data_split = data[idx: idx+num_per_split]
        label_split = label[idx: idx+num_per_split]
        pred = clf.predict(data_split)
        assert len(pred) == len(label_split)
        split_f1s = f1_score(y_true=label_split, y_pred=pred, zero_division=0, average=None)    
        for idx, f1 in enumerate(split_f1s):
            results[organisms[idx]].append(f1)
    for orga in results:
        f1s = results[orga]
        results[orga] = (mean(f1s), stdev(f1s), min(f1s), max(f1s))

    return results


def build_data(proteins):
    datas = []
    labels = []
    for orga in proteins:
        for pro in proteins[orga]:
            datas.append(np.mean(pro.emb, axis=0))
            labels.append(organisms.index(orga))
    datas = np.vstack(datas)
    labels = np.array(labels)

    order = np.arange(len(datas))
    np.random.shuffle(order)
    return datas[order], labels[order]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--pretrained_emb_dir", type=str, default='./data/dscript/processed/embs')
    parser.add_argument("--self_dir", type=str, default="./experiments/5.organism_classify")
    parser.add_argument("--model_dir", type=str, help="saved ppi model")
    args = parser.parse_args()
    random.seed(args.seed)   
    
    protein_dir = os.path.join(args.self_dir, "data")

    train_proteins = {}
    test_proteins = {}
    for orga in organisms:
        train_proteins[orga] = load_proteins(os.path.join(protein_dir, "train"), orga)
        test_proteins[orga] = load_proteins(os.path.join(protein_dir, "test"), orga)
        print(f"{orga} train set size: {len(train_proteins[orga])}")
        print(f"{orga} test set size size: {len(test_proteins[orga])}")

    model_name = os.path.basename(args.model_dir)
    save_dir = os.path.join(args.self_dir, 'save', model_name)
    os.makedirs(save_dir, exist_ok=True)
    result_dir = os.path.join(args.self_dir, "results", model_name)
    os.makedirs(result_dir, exist_ok=True)

    model = load_model(args.model_dir)
    for orga in organisms:
        # load embedding
        if not hasattr(model.encoder, "embeder"):
            for pro in train_proteins[orga] + test_proteins[orga]:
                pro.set_emb(np.load(os.path.join(args.pretrained_emb_dir, orga+"_test", pro.name+".npy")))
        else:
            for pro in train_proteins[orga] + test_proteins[orga]:
                pro.set_emb(lookup_embed(pro, model.encoder.embeder))

    ##### 测试pretrained-embedding
    train_data, train_label = build_data(train_proteins)
    test_data, test_label = build_data(test_proteins)

    print(f">>>>{model_name} train orga classifier for pretrained embedding")
    model_ckpt_fp = os.path.join(save_dir, "emb.ckpt")
    if os.path.exists(model_ckpt_fp):
        clf = joblib.load(model_ckpt_fp)
    else:
        # clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
        clf = CLF()
        clf.fit(train_data, train_label)
        joblib.dump(clf, model_ckpt_fp)

    emb_results = evaluate(clf, test_data, test_label) 
    emb_result_fp = os.path.join(result_dir, 'emb.eval')
    with open(emb_result_fp, "w") as f:
        f.writelines([f"{orga}\t" + '\t'.join(list(map(str, value))) + "\n" for orga, value in emb_results.items()])

    # forward projecter
    for orga in organisms:
        for pro in train_proteins[orga] + test_proteins[orga]:
            emb = model.encoder.forward_projecter(torch.Tensor(pro.emb).unsqueeze(0))
            pro.set_emb(emb.detach().squeeze(0).numpy())

    for k in range(model.encoder.num_layers + 1):
        train_data, train_label = build_data(train_proteins)
        test_data, test_label = build_data(test_proteins)

        print(f">>>>{model_name} train orga classifier for {k}th layer")
        model_ckpt_fp = os.path.join(save_dir, f"{k}.ckpt")
        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            # clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
            clf = CLF()
            clf.fit(train_data, train_label)
            joblib.dump(clf, model_ckpt_fp)

        kth_results = evaluate(clf, test_data, test_label) 
        kth_result_fp = os.path.join(result_dir, f'{k}.eval')
        with open(kth_result_fp, "w") as f:
            f.writelines([f"{orga}\t" + '\t'.join(list(map(str, value))) + "\n" for orga, value in kth_results.items()])

        if k < model.encoder.num_layers:
            for orga in organisms:
                for pro in train_proteins[orga] + test_proteins[orga]:
                    pro.set_emb(forward_kth_translayer(model, pro.emb, k))

      
