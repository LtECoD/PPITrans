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
from experiments.utils import Protein, organisms
from experiments.utils import load_model
from experiments.utils import forward_kth_translayer
from experiments.utils import lookup_embed


def evaluate(clf, data, label):
    """将测试集划分成5份，分别计算准确率"""
    # 划分数据集成10份
    f1s = []
    num_per_split = int(len(data) / 5)
    for idx in range(0, len(data), num_per_split):
        data_split = data[idx: idx+num_per_split]
        label_split = label[idx: idx+num_per_split]
        pred = clf.predict(data_split)
        assert len(pred) == len(label_split)
        f1s.append(f1_score(y_true=label_split, y_pred=pred, zero_division=0))
        
    return mean(f1s), stdev(f1s), min(f1s), max(f1s)


def load_samples(data_dir, num, split):
    pair_fp = os.path.join(data_dir, "pairs", split+".tsv")
    seq_fp = os.path.join(data_dir, "seqs", split+".fasta")
    with open(seq_fp, "r") as f:
        seqs_dict = dict([line.strip().split("\t") for line in f.readlines()])   

    with open(pair_fp, "r") as f:
        lines = random.sample(f.readlines(), k=num)
    samples = [l.strip().split() for l in lines]

    data = []
    for idx, (fst, sec, label) in enumerate(samples):
        fpro = Protein(name=fst, seq=seqs_dict[fst])
        spro = Protein(name=sec, seq=seqs_dict[sec])
        data.append((fpro, spro, label))
    return data


def build_data(proteins):
    """构建氨基酸分类的数据集"""
    datas = []
    labels = []
    for (fpro, spro, l) in proteins:
        datas.append(np.mean(fpro.emb, axis=0)*np.mean(spro.emb, axis=0))
        labels.append(int(l))
    datas = np.vstack(datas)
    return datas, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--processed_dir", type=str, default='./data/dscript/processed')
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--test_samples", type=int, default=2000)
    parser.add_argument("--self_dir", type=str, default="./experiments/4.ppi_classify")
    parser.add_argument("--model_dir", type=str, help="saved ppi model")
    args = parser.parse_args()
    random.seed(args.seed)    
    
    train_proteins = load_samples(args.processed_dir, args.train_samples, "human_train")
    test_proteins = {}
    for orga in organisms:
        test_proteins[orga] = \
            load_samples(args.processed_dir, args.test_samples, orga+"_test")

    model_name = os.path.basename(args.model_dir)
    save_dir = os.path.join(args.self_dir, 'save', model_name)
    os.makedirs(save_dir, exist_ok=True)
    results_dir = os.path.join(args.self_dir, "results", model_name)
    os.makedirs(results_dir, exist_ok=True)

    # 加载模型
    model = load_model(args.model_dir)

    # load embedding
    if not hasattr(model.encoder, "embeder"):
        emb_dir = os.path.join(args.processed_dir, "embs")
        for (fpro, spro, _) in train_proteins:
            fpro.set_emb(np.load(os.path.join(emb_dir, "human_train", fpro.name+".npy")))
            spro.set_emb(np.load(os.path.join(emb_dir, "human_train", spro.name+".npy")))

        for orga in organisms:
            for (fpro, spro, _) in test_proteins[orga]:
                fpro.set_emb(np.load(os.path.join(emb_dir, f"{orga}_test", fpro.name+".npy")))
                spro.set_emb(np.load(os.path.join(emb_dir, f"{orga}_test", spro.name+".npy")))

    else:
        for (fpro, spro, _) in train_proteins:
            fpro.set_emb(lookup_embed(fpro, model.encoder.embeder))
            spro.set_emb(lookup_embed(spro, model.encoder.embeder))

        for orga in organisms:
            for (fpro, spro, _) in test_proteins[orga]:
                fpro.set_emb(lookup_embed(fpro, model.encoder.embeder))
                spro.set_emb(lookup_embed(spro, model.encoder.embeder))

    ##### 测试pretrained-embedding
    emb_results = {}
    # build train dataset
    train_data, train_label = build_data(train_proteins)

    print(f">>>>{model_name}: train ppi classifier for pretrained embedding")
    model_ckpt_fp = os.path.join(save_dir, "emb.ckpt")
    if os.path.exists(model_ckpt_fp):
        clf = joblib.load(model_ckpt_fp)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
        clf.fit(train_data, train_label)
        joblib.dump(clf, model_ckpt_fp)

    for orga in organisms:
        # build test dataset
        test_data, test_label = build_data(test_proteins[orga])       
        emb_results[orga] = evaluate(clf, test_data, test_label)
    emb_result_fp = os.path.join(results_dir, 'emb.eval')
    with open(emb_result_fp, "w") as f:
        f.writelines([f"{orga}\t{value[0]}\t{value[1]}\t{value[2]}\t{value[3]}\n" for orga, value in emb_results.items()])
    
    # forward projecter
    for (fpro, spro, _) in train_proteins:
        femb = model.encoder.forward_projecter(torch.Tensor(fpro.emb).unsqueeze(0))
        fpro.set_emb(femb.detach().squeeze(0).numpy())
        semb = model.encoder.forward_projecter(torch.Tensor(spro.emb).unsqueeze(0))
        spro.set_emb(semb.detach().squeeze(0).numpy())

    for orga in organisms:
        for (fpro, spro, _) in test_proteins[orga]:
            femb = model.encoder.forward_projecter(torch.Tensor(fpro.emb).unsqueeze(0))
            fpro.set_emb(femb.detach().squeeze(0).numpy())
            semb = model.encoder.forward_projecter(torch.Tensor(spro.emb).unsqueeze(0))
            spro.set_emb(semb.detach().squeeze(0).numpy())

    for k in range(model.encoder.transformer.num_layers + 1):
        enc_kth_results = {}

        # build dataset
        train_data, train_label = build_data(train_proteins)
        print(f">>>>{model_name}: train ppi classifier for {k}th layer")
        model_ckpt_fp = os.path.join(save_dir, f"{k}.ckpt")
        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
            clf.fit(train_data, train_label)
            joblib.dump(clf, model_ckpt_fp)

        for orga in organisms:
            test_data, test_label = build_data(test_proteins[orga])
            enc_kth_results[orga] = evaluate(clf, test_data, test_label) 

        if k < model.encoder.transformer.num_layers:
            for (fpro, spro, _) in train_proteins:
                fpro.set_emb(forward_kth_translayer(model, fpro.emb, k))
                spro.set_emb(forward_kth_translayer(model, spro.emb, k))
            for orga in organisms:
                for (fpro, spro, _) in test_proteins[orga]:
                    fpro.set_emb(forward_kth_translayer(model, fpro.emb, k))
                    spro.set_emb(forward_kth_translayer(model, spro.emb, k))

        enc_kth_result_fp = os.path.join(results_dir, f'{k}.eval')
        with open(enc_kth_result_fp, "w") as f:
            f.writelines([f"{orga}\t{value[0]}\t{value[1]}\t{value[2]}\t{value[3]}\n" for orga, value in enc_kth_results.items()])

    gold_results = {}
    for orga in organisms:
        test_data, test_label = build_data(test_proteins[orga])
        test_data = torch.Tensor(test_data)
        logits = model.decoder.projector(test_data)
        pred = torch.argmax(logits, dim=-1).detach().numpy()
        f1 = f1_score(y_true=test_label, y_pred=pred)
        gold_results[orga] = f1

    gold_result_fp = os.path.join(results_dir, f'gold.eval')
    with open(gold_result_fp, "w") as f:
        f.writelines([f"{orga}\t{value}\n" for orga, value in gold_results.items()])