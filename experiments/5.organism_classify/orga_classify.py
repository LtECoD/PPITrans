import os
import argparse
import random
import joblib
import torch
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

import sys
sys.path.append(".")
from module.model import PPIModel
from experiments.utils import load_model, forward_kth_translayer
from experiments.utils import organisms, Protein


def evaluate(clf, data, label):
    """将测试集划分成5份，分别计算准确率"""
    # 划分数据集成10份
    pred = clf.predict(data)
    assert len(pred) == len(data)
    f1s = f1_score(y_true=label, y_pred=pred, zero_division=0, average=None)    
    results = {}
    for idx, f1 in enumerate(f1s):
        results[organisms[idx]] = f1
    return results


def load_proteins(emb_dir, orga, k):
    emb_dir = os.path.join(emb_dir, orga+"_test")
    pros = os.listdir(emb_dir)
    selected_pros = random.sample(pros, k=k)

    pros = []
    for p in selected_pros:
        pp = Protein(name=p[:-4], seq=None)
        pp.set_emb(np.load(os.path.join(emb_dir, p)))
        pros.append(pp)
    return pros


def build_data(proteins):
    datas = []
    labels = []
    for orga in proteins:
        for pro in proteins[orga]:
            datas.append(np.mean(pro.emb, axis=0))
            labels.append(organisms.index(orga))
    datas = np.vstack(datas)
    return datas, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--pretrained_emb_dir", type=str, default='./data/dscript/processed/embs')
    parser.add_argument("--self_dir", type=str, default="./experiments/5.organism_classify")
    parser.add_argument("--num_per_orga", type=int, default=500)
    parser.add_argument("--model_dir", type=str, default="./save/dscript/ppi", help="saved ppi model")
    args = parser.parse_args()
    random.seed(args.seed)   

    train_proteins = {}
    test_proteins = {}
    for orga in organisms:
        train_proteins[orga] = load_proteins(args.pretrained_emb_dir, orga, args.num_per_orga)
        test_proteins[orga] = load_proteins(args.pretrained_emb_dir, orga, args.num_per_orga)

    save_dir = os.path.join(args.self_dir, 'save')
    os.makedirs(save_dir, exist_ok=True)
    result_dir = os.path.join(args.self_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    ##### 测试pretrained-embedding
    # build dataset
    train_data, train_label = build_data(train_proteins)
    test_data, test_label = build_data(test_proteins)

    print(f">>>> train acid classifier for pretrained embedding")
    model_ckpt_fp = os.path.join(save_dir, "emb.ckpt")
    if os.path.exists(model_ckpt_fp):
        clf = joblib.load(model_ckpt_fp)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
        clf.fit(train_data, train_label)
        joblib.dump(clf, model_ckpt_fp)

    emb_results = evaluate(clf, test_data, test_label) 
    emb_result_fp = os.path.join(result_dir, 'emb.eval')
    with open(emb_result_fp, "w") as f:
        f.writelines([f"{orga}\t{value}\n" for orga, value in emb_results.items()])

    #### 测试ppi model
    # 加载模型
    model = load_model(PPIModel, os.path.join(args.model_dir, 'checkpoint_best.pt'))
    # forward projecter
    for orga in organisms:
        for pro in train_proteins[orga] + test_proteins[orga]:
            emb = model.encoder.forward_projecter(torch.Tensor(pro.emb).unsqueeze(0))
            pro.set_emb(emb.detach().squeeze(0).numpy())

    for k in range(model.encoder.transformer.num_layers + 1):
        train_data, train_label = build_data(train_proteins)
        test_data, test_label = build_data(test_proteins)

        print(f">>>> train acid classifier for {k}th layer")
        model_ckpt_fp = os.path.join(save_dir, f"{k}.ckpt")
        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
            clf.fit(train_data, train_label)
            joblib.dump(clf, model_ckpt_fp)

        kth_results = evaluate(clf, test_data, test_label) 
        kth_result_fp = os.path.join(result_dir, f'{k}th.eval')
        with open(kth_result_fp, "w") as f:
            f.writelines([f"{orga}\t{value}\n" for orga, value in kth_results.items()])

        if k < model.encoder.transformer.num_layers:
            for orga in organisms:
                for pro in train_proteins[orga] + test_proteins[orga]:
                    pro.set_emb(forward_kth_translayer(model, pro.emb, k))

      
