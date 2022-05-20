import os
import argparse
import random
import joblib
import torch
import numpy as np
from statistics import stdev, mean

import sys
sys.path.append(".")
from experiments.utils import organisms, Protein
from experiments.utils import load_model
from experiments.utils import forward_kth_translayer
from experiments.utils import lookup_embed
from experiments.freq_pred_model import MyTrainer


def load_samples(data_dir, split):
    pair_fp = os.path.join(data_dir, split+".tsv")
    seq_fp = os.path.join(data_dir, split+".seq")

    with open(seq_fp, "r") as f:
        seqs_dict = dict([line.strip().split("\t") for line in f.readlines()])
    proteins = []
    for name, seq in seqs_dict.items():
        pro = Protein(name=name, seq=seq) 
        pro.count_aa()
        proteins.append(pro)
    names = [pro.name for pro in proteins]

    with open(pair_fp, "r") as f:
        pairs = [line.strip().split() for line in f.readlines()]
        pairs = [[names.index(fp), names.index(sp), int(l)]  for fp, sp, l in pairs]

    return np.array(pairs), proteins


def evaluate(clf, data, label):
    """将测试集划分成10份，分别计算准确率"""
    # 划分数据集成10份
    losses = []
    num_per_split = int(len(data) / 10)
    for idx in range(0, len(data), num_per_split):
        data_split = data[idx: idx+num_per_split]
        label_split = label[idx: idx+num_per_split]
        loss = clf.predict(data_split, label_split)
        losses.append(mean(loss))

    return mean(losses), stdev(losses), min(losses), max(losses)


def build_data(proteins, pairs):
    """构建氨基酸分类的数据集"""
    embs = np.vstack([np.mean(pro.emb, axis=0) for pro in proteins])
    fpro_index = pairs[:, 0]
    spro_index = pairs[:, 1]

    datas = [embs[fpro_index], embs[spro_index]]
    freqs = [proteins[idx].aa_freq for idx in spro_index]
    freqs = freqs + [proteins[idx].aa_freq for idx in fpro_index]
    datas = np.vstack(datas)
    freqs = np.vstack(freqs)

    return datas, freqs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--pretrained_emb_dir", type=str, default='./data/dscript/processed/embs')
    parser.add_argument("--self_dir", type=str, default="./experiments/10.ppi_aa_freq_prediction")
    parser.add_argument("--model_dir", type=str, help="saved ppi model")
    args = parser.parse_args()
    random.seed(args.seed)    
    
    data_dir = os.path.join(args.self_dir, "data")

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
    train_pairs = {}
    test_proteins = {}
    test_pairs = {}
    for orga in organisms:
        train_pairs[orga], train_proteins[orga] = load_samples(os.path.join(data_dir, "train"), orga) 
        test_pairs[orga], test_proteins[orga] = load_samples(os.path.join(data_dir, "test"), orga)
        print(f"{orga} train set size: {len(train_pairs[orga])} pairs, {len(train_proteins[orga])} proteins")
        print(f"{orga} test set size: {len(test_pairs[orga])} pairs, {len(test_proteins[orga])} proteins")

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
        train_data, train_label = build_data(train_proteins[orga], train_pairs[orga])
        test_data, test_label = build_data(test_proteins[orga], test_pairs[orga])
        print(f">>>>{model_name}-{orga}: train ppi aa_freq predictor for pretrained embedding")

        clf = MyTrainer(input_dim=train_data.shape[-1], out_dim=21)
        model_ckpt_fp = os.path.join(save_dir, orga, f"emb.ckpt")
        if os.path.exists(model_ckpt_fp):
            clf.load(model_ckpt_fp)
        else:
            clf.fit(train_data, train_label, test_data, test_label)
            clf.save(model_ckpt_fp)

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
            train_data, train_label = build_data(train_proteins[orga], train_pairs[orga])
            test_data, test_label = build_data(test_proteins[orga], test_pairs[orga])
            print(f">>>>{model_name}-{orga}: train ppi aa_freq predictor for {k}th layer")

            clf = MyTrainer(train_data.shape[-1], out_dim=21)
            model_ckpt_fp = os.path.join(save_dir, orga, f"{k}.ckpt")
            if os.path.exists(model_ckpt_fp):
                clf = joblib.load(model_ckpt_fp)
            else:
                clf.fit(train_data, train_label, test_data, test_label)
                clf.save(model_ckpt_fp)

            enc_kth_results[orga] = evaluate(clf, test_data, test_label)
        enc_kth_result_fp = os.path.join(results_dir, f'{k}.eval')

        with open(enc_kth_result_fp, "w") as f:
            f.writelines([f"{orga}\t" + '\t'.join(list(map(str, value))) + "\n" for orga, value in enc_kth_results.items()])

        if k < model.encoder.num_layers:
            for pro in train_proteins[orga] + test_proteins[orga]:
                pro.set_emb(forward_kth_translayer(model, pro.emb, k))
