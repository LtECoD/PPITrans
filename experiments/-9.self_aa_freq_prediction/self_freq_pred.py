import os
import argparse
import random
import joblib
import torch
import numpy as np
from statistics import stdev, mean
# from sklearn.kernel_ridge import KernelRidge as REG
from sklearn.linear_model import Lasso as REG
from sklearn.metrics import mean_squared_error

import sys
sys.path.append(".")
from experiments.utils import organisms, Protein
from experiments.utils import load_model
from experiments.utils import forward_kth_translayer
from experiments.utils import lookup_embed
from experiments.freq_pred_model import MyTrainer


def load_proteins(_dir, split):
    lines = open(os.path.join(_dir, split+".seq")).readlines()
    proteins = []
    for l in lines:
        name, seq = l.strip().split()
        pro = Protein(name=name, seq=seq)
        if "X" in pro.seq:
            continue
        pro.count_aa()
        proteins.append(pro)
    return proteins
    

def evaluate(clf, data, score):
    ces = []
    num_per_split = int(len(data) / 5)
    for idx in range(0, len(data), num_per_split):
        data_split = data[idx: idx+num_per_split]
        score_split = score[idx: idx+num_per_split]
        pred = clf.predict(data_split)
        pred = pred / np.sum(pred, axis=-1, keepdims=True)
        assert len(pred) == len(score_split)
        ces.append(mean_squared_error(score_split, pred))
        # f1s.append(f1_score(y_true=label_split, y_pred=pred, zero_division=0))
    return mean(ces), stdev(ces), min(ces), max(ces)


def build_data(proteins):
    """构建氨基酸分类的数据集"""
    datas = []
    freqs = []
    for pro in proteins:
        datas.append(np.mean(pro.emb, axis=0))
        freqs.append(np.array(pro.aa_freq[:20]))
    datas = np.vstack(datas)
    freqs = np.vstack(freqs)

    order = np.arange(len(datas))
    np.random.shuffle(order)
    return datas[order], freqs[order]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--pretrained_emb_dir", type=str, default='./data/dscript/processed/embs')
    parser.add_argument("--self_dir", type=str, default="./experiments/9.self_aa_freq_prediction")
    parser.add_argument("--model_dir", type=str, help="saved ppi model")
    args = parser.parse_args()
    random.seed(args.seed)    
    
    protein_dir = os.path.join(args.self_dir, "data")

    model_name = os.path.basename(args.model_dir)
    save_dir = os.path.join(args.self_dir, 'save', model_name)
    os.makedirs(save_dir, exist_ok=True)
    for orga in organisms:
        os.makedirs(os.path.join(save_dir, orga), exist_ok=True)
    results_dir = os.path.join(args.self_dir, "results", model_name)
    os.makedirs(results_dir, exist_ok=True)
    # 加载模型
    model = load_model(args.model_dir)

    train_proteins = {}
    test_proteins = {}
    for orga in organisms:
        train_proteins[orga] = load_proteins(os.path.join(protein_dir, "train"), orga)
        print(f"{orga} train set size: {len(train_proteins[orga])}")
        test_proteins[orga] = load_proteins(os.path.join(protein_dir, "test"), orga)
        print(f"{orga} test set size size: {len(test_proteins[orga])}")

    # load embedding
    for orga in organisms:
        if not hasattr(model.encoder, "embeder"):
            for pro in train_proteins[orga] + test_proteins[orga]:
                pro.set_emb(np.load(os.path.join(args.pretrained_emb_dir, orga+"_test", pro.name+".npy")))
        else:
            for pro in train_proteins[orga] + test_proteins[orga]:
                pro.set_emb(lookup_embed(pro, model.encoder.embeder))

    # test embedding
    emb_results = {}
    for orga in organisms:
        train_data, train_label = build_data(train_proteins[orga])
        test_data, test_label = build_data(test_proteins[orga])
        print(f">>>>{model_name}-{orga}: train self aa_freq predictor for pretrained embedding")

        clf = REG()
        model_ckpt_fp = os.path.join(save_dir, orga, f"emb.ckpt")
        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            clf.fit(train_data, train_label, test_data, test_label)
            joblib.dump(clf, model_ckpt_fp)

        emb_results[orga] = evaluate(clf, test_data, test_label)
    emb_result_fp = os.path.join(results_dir, 'emb.eval')
    with open(emb_result_fp, "w") as f:
        f.writelines([f"{orga}\t" + '\t'.join(list(map(str, value))) + "\n" for orga, value in emb_results.items()])

    # forward projecter
    for orga in organisms:
        for pro in train_proteins[orga] + test_proteins[orga]:
            emb = model.encoder.forward_projecter(torch.Tensor(pro.emb).unsqueeze(0))
            pro.set_emb(emb.detach().squeeze(0).numpy())

    for k in range(model.encoder.num_layers + 1):
        enc_kth_results = {}
        for orga in organisms:
            train_data, train_label = build_data(train_proteins[orga])
            test_data, test_label = build_data(test_proteins[orga])
            print(f">>>>{model_name}-{orga}: train self aa_freq predictor for {k}th layer")

            clf = REG()
            model_ckpt_fp = os.path.join(save_dir, orga, f"{k}.ckpt")
            if os.path.exists(model_ckpt_fp):
                clf = joblib.load(model_ckpt_fp)
            else:
                clf.fit(train_data, train_label, test_data, test_label)
                joblib.dump(clf, model_ckpt_fp)

            enc_kth_results[orga] = evaluate(clf, test_data, test_label)
        enc_kth_result_fp = os.path.join(results_dir, f'{k}.eval')

        with open(enc_kth_result_fp, "w") as f:
            f.writelines([f"{orga}\t" + '\t'.join(list(map(str, value))) + "\n" for orga, value in enc_kth_results.items()])

        if k < model.encoder.num_layers:
            for pro in train_proteins[orga] + test_proteins[orga]:
                pro.set_emb(forward_kth_translayer(model, pro.emb, k))
