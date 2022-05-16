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
from experiments.utils import load_proteins
from experiments.utils import organisms
from experiments.utils import load_model
from experiments.utils import forward_kth_translayer
from experiments.utils import lookup_embed


def evaluate(clf, data, label):
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


def build_data(proteins):
    data = np.vstack([pro.emb for pro in proteins])
    
    # label = list("".join([pro.seq for pro in proteins]))
    # seq_id = [acids.index(l) for l in label]
    # label = [types[_id] for _id in seq_id]
    pass
    return data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--pretrained_emb_dir", type=str, default='./data/dscript/processed/embs')
    parser.add_argument("--self_dir", type=str, default="./experiments/8.context_prediction")
    parser.add_argument("--protein_dir", type=str, default="./experiments/1.acid_classify/data")
    parser.add_argument("--model_dir", type=str, help="saved ppi model")
    args = parser.parse_args()
    random.seed(args.seed)  

    train_proteins = {}
    test_proteins = {}
    for orga in organisms:
        train_proteins[orga] = load_proteins(orga, os.path.join(args.protein_dir, "train"))
        test_proteins[orga] = load_proteins(orga, os.path.join(args.protein_dir, "test"))
    
    ##### 测试pretrained-embedding
    model_name = os.path.basename(args.model_dir)
    save_dir = os.path.join(args.self_dir, 'save', model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    results_dir = os.path.join(args.self_dir, "results", model_name)
    os.makedirs(results_dir, exist_ok=True)

    # 加载模型
    model = load_model(args.model_dir)

    emb_mlp_save_dir = os.path.join(save_dir, 'embding')
    os.makedirs(emb_mlp_save_dir, exist_ok=True)
    emb_results = {}
    for orga in organisms:
        # load embedding
        if not hasattr(model.encoder, "embeder"):
            for pro in train_proteins[orga] + test_proteins[orga]:
                pro.set_emb(np.load(os.path.join(args.pretrained_emb_dir, orga+"_test", pro.name+".npy")))
        else:
            for pro in train_proteins[orga] + test_proteins[orga]:
                pro.set_emb(lookup_embed(pro, model.encoder.embeder))

        # build dataset
        train_data, train_label = build_data(train_proteins[orga])
        test_data, test_label = build_data(test_proteins[orga])
        print(f">>>>{model_name}-{orga}: train context predictor for pretrained embedding")
        model_ckpt_fp = os.path.join(emb_mlp_save_dir, orga+".ckpt")
        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
            clf.fit(train_data, train_label)
            joblib.dump(clf, model_ckpt_fp)

        emb_results[orga] = evaluate(clf, test_data, test_label) 
    emb_result_fp = os.path.join(results_dir, 'emb.eval')
    with open(emb_result_fp, "w") as f:
        f.writelines(
            [f"{orga}\t" + '\t'.join(list(map(str, value))) + "\n" for orga, value in emb_results.items()])
    
    # forward projecter
    for orga in organisms:
        for pro in train_proteins[orga] + test_proteins[orga]:
            emb = model.encoder.forward_projecter(torch.Tensor(pro.emb).unsqueeze(0))
            pro.set_emb(emb.detach().squeeze(0).numpy())

    for k in range(model.encoder.num_layers + 1):
        enc_mlp_save_dir = os.path.join(save_dir, str(k))
        os.makedirs(enc_mlp_save_dir, exist_ok=True)
        enc_kth_results = {}

        for orga in organisms:
            # build dataset
            train_data, train_label = build_data(train_proteins[orga])
            test_data, test_label = build_data(test_proteins[orga])

            print(f">>>>{model_name}-{orga}: train context predictor for {k}th layer")
            
            model_ckpt_fp = os.path.join(enc_mlp_save_dir, orga+".ckpt")
            if os.path.exists(model_ckpt_fp):
                clf = joblib.load(model_ckpt_fp)
            else:
                clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
                clf.fit(train_data, train_label)
                joblib.dump(clf, model_ckpt_fp)
            enc_kth_results[orga] = evaluate(clf, test_data, test_label)

            if k < model.encoder.num_layers:
                for pro in train_proteins[orga] + test_proteins[orga]:
                    pro.set_emb(forward_kth_translayer(model, pro.emb, k))

        enc_kth_result_fp = os.path.join(results_dir, f'{k}.eval')
        with open(enc_kth_result_fp, "w") as f:
            f.writelines([f"{orga}\t" + '\t'.join(list(map(str, value))) + "\n" for orga, value in enc_kth_results.items()])


