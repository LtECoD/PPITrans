import os
import argparse
import random
import joblib
import torch
import numpy as np
from tqdm import tqdm
from statistics import stdev, mean
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

import sys
sys.path.append(".")
from module.model import PPIModel
from experiments.utils import Protein, load_model, forward_kth_translayer


def load_proteins(split, _dir):
    seqs = open(os.path.join(_dir, split+".seq"), "r").readlines()
    cms = np.load(os.path.join(_dir, split+".npy"), allow_pickle=True).tolist()
    proteins = []
    for seq in seqs:
        _id, seq = seq.strip().split()
        pro = Protein(_id, seq)
        pro.set_cm(cms[_id])
        proteins.append(pro)
    return proteins


def build_data(proteins, threshold):
    data = []
    label = []
    for pro in proteins:
        cm = pro.cm < threshold
        pemb = np.expand_dims(pro.emb, axis=1) * np.expand_dims(pro.emb, axis=0)    # L x L x D
        xindices, yindices = np.triu_indices(pro.length, k=1)
        pemb = pemb[xindices, yindices]
        cm = cm[xindices, yindices]

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
    parser.add_argument("--self_dir", type=str, default="./experiments/6.self_contact_map")
    parser.add_argument("--model_dir", type=str, default="./save/dscript/ppi", help="saved ppi model")
    parser.add_argument("--threshold", type=float, default=8)
    args = parser.parse_args()
    random.seed(args.seed)   

    protein_dir = os.path.join(args.self_dir, "data")
    pretrained_emb_dir = os.path.join(protein_dir, "embs")

    save_dir = os.path.join(args.self_dir, 'save')
    results_dir = os.path.join(args.self_dir, 'results')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_proteins = load_proteins('train', protein_dir)
    test_proteins = load_proteins('test', protein_dir)

    ##### 测试pretrained-embedding    
    emb_results = {}
    for pro in train_proteins + test_proteins:
        pro.set_emb(np.load(os.path.join(pretrained_emb_dir, pro.name+".npy")))
    
    train_data, train_label = build_data(train_proteins, args.threshold)
    test_data, test_label = build_data(test_proteins, args.threshold)

    print(f">>>> train cm classifier for pretrained embedding")
    model_ckpt_fp = os.path.join(save_dir, f"emb.ckpt")
    if os.path.exists(model_ckpt_fp):
        clf = joblib.load(model_ckpt_fp)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
        clf.fit(train_data, train_label)
        joblib.dump(clf, model_ckpt_fp)

    avg_acc, acc_stdev, acc_min, acc_max = evaluate(clf, test_data, test_label) 
    emb_results = (avg_acc, acc_stdev, acc_min, acc_max)

    emb_result_fp = os.path.join(args.self_dir, 'results', f'emb.eval')
    with open(emb_result_fp, "w") as f:
        f.write("\t".join(list(map(str, emb_results))))

    #### 测试ppi model
    # 加载模型
    model = load_model(PPIModel, os.path.join(args.model_dir, 'checkpoint_best.pt'))
    # forward projecter
    for pro in train_proteins + test_proteins:
        emb = model.encoder.forward_projecter(torch.Tensor(pro.emb).unsqueeze(0))
        pro.set_emb(emb.detach().squeeze(0).numpy())

    for k in range(model.encoder.transformer.num_layers + 1):
        # build dataset
        train_data, train_label = build_data(train_proteins, args.threshold)
        test_data, test_label = build_data(test_proteins, args.threshold)
        print(f">>>> train cm classifier for {k}th layer")
        
        model_ckpt_fp = os.path.join(save_dir, f"{k}.ckpt")
        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
            clf.fit(train_data, train_label)
            joblib.dump(clf, model_ckpt_fp)
        avg_acc, acc_stdev, acc_min, acc_max = evaluate(clf, test_data, test_label) 
        enc_kth_results = (avg_acc, acc_stdev, acc_min, acc_max)

        if k < model.encoder.transformer.num_layers:
            for pro in train_proteins + test_proteins:
                pro.set_emb(forward_kth_translayer(model, pro.emb, k))

        enc_kth_result_fp = os.path.join(args.self_dir, 'results', f'{k}.eval')
        with open(enc_kth_result_fp, "w") as f:
            f.write("\t".join(list(map(str, enc_kth_results))))
            
