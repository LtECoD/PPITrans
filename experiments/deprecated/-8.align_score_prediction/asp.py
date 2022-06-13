import os
import argparse
import random
import shutil
import joblib
import torch
import numpy as np
from math import ceil
from statistics import stdev, mean
from sklearn.metrics import mean_squared_error
# from sklearn.kernel_ridge import KernelRidge as REG
# from sklearn.linear_model import Lasso as REG
from sklearn.svm import SVR as REG

import sys
sys.path.append(".")
from experiments.utils import Protein, organisms
from experiments.utils import load_model
from experiments.utils import forward_kth_translayer
from experiments.utils import lookup_embed


def evaluate(clf, data, score):
    """将测试集划分成5份，分别计算准确率"""
    mses = []
    num_per_split = ceil(len(data) / 5)
    for idx in range(0, len(data), num_per_split):
        data_split = data[idx: idx+num_per_split]
        score_split = score[idx: idx+num_per_split]
        pred = clf.predict(data_split)
        assert len(pred) == len(score_split)
        mses.append(mean_squared_error(score_split, pred))

    return mean(mses), stdev(mses), min(mses), max(mses)


def load_samples(data_dir, split):
    pair_fp = os.path.join(data_dir, split+".tsv")
    seq_fp = os.path.join(data_dir, split+".seq")

    with open(seq_fp, "r") as f:
        seqs_dict = dict([line.strip().split("\t") for line in f.readlines()])   
    proteins = [Protein(name=name, seq=seq) for name, seq in seqs_dict.items()]
    names = [pro.name for pro in proteins]

    with open(pair_fp, "r") as f:
        pairs = [line.strip().split() for line in f.readlines()]
        pairs = [[names.index(fp), names.index(sp), float(_score)]  for fp, sp, _score in pairs]

    return np.array(pairs), proteins


def build_data(pairs, proteins):
    """构建氨基酸分类的数据集"""
    # pairs: B x 3
    embs = np.vstack([np.mean(pro.emb, axis=0) for pro in proteins])
    fpro_index = pairs[:, 0].astype(np.int16)
    spro_index = pairs[:, 1].astype(np.int16)
    datas = embs[fpro_index] * embs[spro_index]

    order = np.arange(len(datas))
    np.random.shuffle(order)

    return datas[order], pairs[:, 2][order]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--pretrained_emb_dir", type=str, default='./data/dscript/processed/embs')
    parser.add_argument("--self_dir", type=str, default="./experiments/8.align_score_prediction")
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
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
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
        train_data, train_score = build_data(train_pairs[orga], train_proteins[orga])
        print(f">>>>{model_name}-{orga}: train ppi classifier for pretrained embedding")
        model_ckpt_fp = os.path.join(save_dir, orga, "emb.ckpt")

        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            clf = REG()
            clf.fit(train_data, train_score)
            joblib.dump(clf, model_ckpt_fp)
        test_data, test_score = build_data(test_pairs[orga], test_proteins[orga])
        emb_results[orga] = evaluate(clf, test_data, test_score)
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
            train_data, train_score = build_data(train_pairs[orga], train_proteins[orga])
            print(f">>>>{model_name}-{orga}: train ppi classifier for {k}th layer")

            model_ckpt_fp = os.path.join(save_dir, orga, f"{k}.ckpt")
            if os.path.exists(model_ckpt_fp):
                clf = joblib.load(model_ckpt_fp)
            else:
                clf = REG()
                clf.fit(train_data, train_score)
                joblib.dump(clf, model_ckpt_fp)

            # build dataset
            test_data, test_score = build_data(test_pairs[orga], test_proteins[orga])
            enc_kth_results[orga] = evaluate(clf, test_data, test_score)

        enc_kth_result_fp = os.path.join(results_dir, f'{k}.eval')
        with open(enc_kth_result_fp, "w") as f:
            f.writelines([f"{orga}\t" + '\t'.join(list(map(str, value))) + "\n" for orga, value in enc_kth_results.items()])

        if k < model.encoder.num_layers:
            for orga in organisms:
                for pro in train_proteins[orga] + test_proteins[orga]:
                    pro.set_emb(forward_kth_translayer(model, pro.emb, k))
