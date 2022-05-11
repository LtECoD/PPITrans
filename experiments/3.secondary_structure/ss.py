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
from module.model import PPIModel
from experiments.utils import Protein, load_model
from experiments.utils import forward_kth_translayer
from experiments.utils import lookup_embed


SS=['L','B','E','G','I','H','S','T','X']
SS8_Dict=dict(zip(SS,range(len(SS))))
SS3_Dict={'L': 0, 'B': 1, 'E': 1, 'G': 2, 'I': 2, 'H': 2, 'S': 0, 'T': 0, 'X':3}

def evaluate(clf, data, label):
    """将测试集划分成10份，分别计算准确率"""
    # 划分数据集成10份
    accs = []
    num_per_split = int(len(data) / 10)
    for idx in range(0, len(data), num_per_split):
        data_split = data[idx: idx+num_per_split]
        label_split = label[idx: idx+num_per_split]
        pred = clf.predict(data_split)
        assert len(pred) == len(label_split)
        accs.append(sum(pred==label_split) / len(pred))

    return mean(accs), stdev(accs), min(accs), max(accs)


def load_proteins(split, _dir):
    seqs = open(os.path.join(_dir, split+".seq"), "r").readlines()
    proteins = []
    for seq in seqs:
        _id, seq, ss = seq.strip().split()
        pro = Protein(_id, seq)
        pro.set_ss(ss)
        proteins.append(pro)
    return proteins
    

def build_data(proteins, is_eight_class=False):
    data = np.vstack([pro.emb for pro in proteins])
    label = list("".join([pro.ss for pro in proteins]))
    if is_eight_class:
        label = [SS8_Dict[l] for l in label]
    else:
        label = [SS3_Dict[l] for l in label]
    return data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--self_dir", type=str, default="./experiments/3.secondary_structure")
    parser.add_argument("--model_dir", type=str, help="saved ppi model")
    parser.add_argument("--is_eight_class", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)    
    
    protein_dir = os.path.join(args.self_dir, "data")
    pretrained_emb_dir = os.path.join(protein_dir, "embs")

    class_num = 8 if args.is_eight_class else 3
    train_proteins = load_proteins('train', protein_dir)
    test_proteins = load_proteins('test', protein_dir)

    model_name = os.path.basename(args.model_dir)
    save_dir = os.path.join(args.self_dir, 'save', model_name)
    os.makedirs(save_dir, exist_ok=True)
    results_dir = os.path.join(args.self_dir, "results", model_name)
    os.makedirs(results_dir, exist_ok=True)

    # 加载模型
    model = load_model(args.model_dir)

    ##### 测试pretrained-embedding
    emb_results = {}
    # load embedding
    if not hasattr(model.encoder, "embeder"):
        for pro in train_proteins + test_proteins:
            pro.set_emb(np.load(os.path.join(pretrained_emb_dir, pro.name+".npy")))
    else:
        for pro in train_proteins + test_proteins:
            pro.set_emb(lookup_embed(pro, model.encoder.embeder))


    # build dataset
    train_data, train_label = build_data(train_proteins, args.is_eight_class)
    test_data, test_label = build_data(test_proteins, args.is_eight_class)
    print(f">>>>{model_name}: train ss classifier for pretrained embedding")
    model_ckpt_fp = os.path.join(save_dir, f"emb-{class_num}.ckpt")
    if os.path.exists(model_ckpt_fp):
        clf = joblib.load(model_ckpt_fp)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
        clf.fit(train_data, train_label)
        joblib.dump(clf, model_ckpt_fp)

    emb_results = evaluate(clf, test_data, test_label) 
    emb_result_fp = os.path.join(results_dir, f'emb-{class_num}.eval')
    with open(emb_result_fp, "w") as f:
        f.write("\t".join(list(map(str, emb_results))))
       
    #### 测试 model
    # forward projecter
    for pro in train_proteins + test_proteins:
        emb = model.encoder.forward_projecter(torch.Tensor(pro.emb).unsqueeze(0))
        pro.set_emb(emb.detach().squeeze(0).numpy())

    for k in range(model.encoder.transformer.num_layers + 1):
        # build dataset
        train_data, train_label = build_data(train_proteins, args.is_eight_class)
        test_data, test_label = build_data(test_proteins, args.is_eight_class)
        print(f">>>>{model_name}: train ss classifier for {k}th layer")
        
        model_ckpt_fp = os.path.join(save_dir, f"{k}-{class_num}.ckpt")
        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
            clf.fit(train_data, train_label)
            joblib.dump(clf, model_ckpt_fp)
        enc_kth_results = evaluate(clf, test_data, test_label)

        if k < model.encoder.transformer.num_layers:
            for pro in train_proteins + test_proteins:
                pro.set_emb(forward_kth_translayer(model, pro.emb, k))

        enc_kth_result_fp = os.path.join(results_dir, f'{k}-{class_num}.eval')
        with open(enc_kth_result_fp, "w") as f:
            f.write("\t".join(list(map(str, enc_kth_results))))
            


