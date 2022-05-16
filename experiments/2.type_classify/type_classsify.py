import os
import argparse
import random
import joblib
import torch
import numpy as np
from statistics import stdev, mean
from sklearn.neural_network import MLPClassifier

import sys
sys.path.append(".")
from experiments.utils import organisms, acids, types, Protein
from experiments.utils import load_model, forward_kth_translayer
from experiments.utils import lookup_embed


def load_proteins(_dir, split):
    lines = open(os.path.join(_dir, split+".seq")).readlines()
    proteins = []
    for l in lines:
        name, seq = l.strip().split()
        proteins.append(Protein(name=name, seq=seq))
    return proteins


def evaluate(clf, data, label):
    accs = []
    num_per_split = int(len(data) / 10)
    for idx in range(0, len(data), num_per_split):
        data_split = data[idx: idx+num_per_split]
        label_split = label[idx: idx+num_per_split]
        pred = clf.predict(data_split)
        assert len(pred) == len(label_split)
        accs.append(sum(pred==label_split) / len(pred))

    return mean(accs), stdev(accs), min(accs), max(accs)


def build_data(proteins):
    """构建氨基酸分类的数据集"""
    data = np.vstack([pro.emb for pro in proteins])
    label = list("".join([pro.seq for pro in proteins]))
    seq_id = [acids.index(l) for l in label]
    label = [types[_id] for _id in seq_id]
    return data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--pretrained_emb_dir", type=str, default='./data/dscript/processed/embs')
    parser.add_argument("--self_dir", type=str, default="./experiments/2.type_classify")
    parser.add_argument("--protein_dir", type=str, default="./experiments/1.acid_classify/data")
    parser.add_argument("--model_dir", type=str, help="saved ppi model")
    args = parser.parse_args()
    random.seed(args.seed)    
    
    train_proteins = load_proteins(os.path.join(args.protein_dir, "train"), "human_train")
    print(f"train set size: {sum([p.length for p in train_proteins])}")
    test_proteins = {}
    for orga in organisms:
        test_proteins[orga] = load_proteins(os.path.join(args.protein_dir, "test"), orga+"_test")
        print(f"{orga} test set size size: {sum([p.length for p in test_proteins[orga]])}")

    model_name = os.path.basename(args.model_dir)
    save_dir = os.path.join(args.self_dir, 'save', model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    results_dir = os.path.join(args.self_dir, "results", model_name)
    os.makedirs(results_dir, exist_ok=True)

    # 加载模型
    model = load_model(args.model_dir)

    # load embedding
    for pro in train_proteins:
        if not hasattr(model.encoder, "embeder"):
            pro.set_emb(np.load(os.path.join(args.pretrained_emb_dir, "human_train", pro.name+".npy")))
        else:
            pro.set_emb(lookup_embed(pro, model.encoder.embeder))
    for orga in organisms:
        if not hasattr(model.encoder, "embeder"):
            for pro in test_proteins[orga]:
                pro.set_emb(np.load(os.path.join(args.pretrained_emb_dir, orga+"_test", pro.name+".npy")))
        else:
            for pro in test_proteins[orga]:
                pro.set_emb(lookup_embed(pro, model.encoder.embeder))

    # test embedding
    emb_results = {}
    train_data, train_label = build_data(train_proteins)
    print(f">>>>{model_name}: train type classifier for pretrained embedding")
    model_ckpt_fp = os.path.join(save_dir, "emb.ckpt")
    if os.path.exists(model_ckpt_fp):
        clf = joblib.load(model_ckpt_fp)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
        clf.fit(train_data, train_label)
        joblib.dump(clf, model_ckpt_fp)

    for orga in organisms:
        test_data, test_label = build_data(test_proteins[orga])
        emb_results[orga] = evaluate(clf, test_data, test_label)
    emb_result_fp = os.path.join(results_dir, 'emb.eval')
    with open(emb_result_fp, "w") as f:
        f.writelines([f"{orga}\t" + '\t'.join(list(map(str, value))) + "\n" for orga, value in emb_results.items()])

    # forward projecter
    for pro in train_proteins:
        emb = model.encoder.forward_projecter(torch.Tensor(pro.emb).unsqueeze(0))
        pro.set_emb(emb.detach().squeeze(0).numpy())
    for orga in organisms:
        for pro in test_proteins[orga]:
            emb = model.encoder.forward_projecter(torch.Tensor(pro.emb).unsqueeze(0))
            pro.set_emb(emb.detach().squeeze(0).numpy())

    # test kth layer
    for k in range(model.encoder.num_layers + 1):
        enc_kth_results = {}
        train_data, train_label = build_data(train_proteins)
        
        print(f">>>>{model_name}: train type classifier for {k}th layer")
        model_ckpt_fp = os.path.join(save_dir, f"{k}.ckpt")
        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
            clf.fit(train_data, train_label)
            joblib.dump(clf, model_ckpt_fp)

        for orga in organisms:
            # build dataset
            test_data, test_label = build_data(test_proteins[orga])
            enc_kth_results[orga] = evaluate(clf, test_data, test_label)

        if k < model.encoder.num_layers:
            for pro in train_proteins:
                pro.set_emb(forward_kth_translayer(model, pro.emb, k))
            for pro in test_proteins[orga]:
                pro.set_emb(forward_kth_translayer(model, pro.emb, k))

        enc_kth_result_fp = os.path.join(results_dir, f'{k}.eval')
        with open(enc_kth_result_fp, "w") as f:
            f.writelines([f"{orga}\t" + '\t'.join(list(map(str, value))) + "\n" for orga, value in enc_kth_results.items()])
