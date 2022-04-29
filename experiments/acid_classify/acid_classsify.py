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
from module.model import PPIModel
from experiments.utils import load_proteins
from experiments.utils import organisms, acids
from experiments.utils import load_model, forward_kth_translayer


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

    return mean(accs), stdev(accs)


def build_data(proteins):
    """构建氨基酸分类的数据集"""
    data = np.vstack([pro.emb for pro in proteins])
    label = list("".join([pro.seq[:len(pro.emb)] for pro in proteins]))
    label = [acids.index(l) for l in label]
    return data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--pretrained_emb_dir", type=str, default='./data/dscript/processed/embs')
    parser.add_argument("--self_dir", type=str, default="./experiments/acid_classify")
    parser.add_argument("--protein_dir", type=str, default="./experiments/proteins")
    parser.add_argument("--model_dir", type=str, default="./save/dscript/ppi", help="saved ppi model")
    args = parser.parse_args()
    random.seed(args.seed)    
    
    train_proteins = {}
    test_proteins = {}
    for orga in organisms:
        train_proteins[orga] = load_proteins(orga, os.path.join(args.protein_dir, "train"))
        test_proteins[orga] = load_proteins(orga, os.path.join(args.protein_dir, "test"))

    ##### 测试pretrained-embedding
    emb_mlp_save_dir = os.path.join(args.self_dir, 'save', 'embding')
    os.makedirs(emb_mlp_save_dir, exist_ok=True)
    emb_results = {}
    for orga in organisms:
        # load embedding
        for pro in train_proteins[orga] + test_proteins[orga]:
            pro.set_emb(np.load(os.path.join(args.pretrained_emb_dir, orga+"_test", pro.name+".npy")))

        # build dataset
        train_data, train_label = build_data(train_proteins[orga])
        test_data, test_label = build_data(test_proteins[orga])
        print(f">>>>{orga}: train acid classifier for pretrained embedding")
        model_ckpt_fp = os.path.join(emb_mlp_save_dir, orga+".ckpt")
        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
            clf.fit(train_data, train_label)
            joblib.dump(clf, model_ckpt_fp)

        avg_acc, acc_stdev = evaluate(clf, test_data, test_label) 
        emb_results[orga] = (avg_acc, acc_stdev)
    emb_result_fp = os.path.join(args.self_dir, 'results', 'emb.eval')
    with open(emb_result_fp, "w") as f:
        f.writelines([f"{orga}\t{value[0]}\t{value[1]}\n" for orga, value in emb_results.items()])

    #### 测试ppi model
    # 加载模型
    model = load_model(PPIModel, os.path.join(args.model_dir, 'checkpoint_best.pt'))
    # forward projecter
    for orga in organisms:
        for pro in train_proteins[orga] + test_proteins[orga]:
            emb = model.encoder.forward_projecter(torch.Tensor(pro.emb).unsqueeze(0))
            pro.set_emb(emb.detach().squeeze(0).numpy())

    for k in range(model.encoder.transformer.num_layers + 1):
        enc_mlp_save_dir = os.path.join(args.self_dir, 'save', str(k))
        os.makedirs(enc_mlp_save_dir, exist_ok=True)
        enc_kth_results = {}

        for orga in organisms:
            # build dataset
            train_data, train_label = build_data(train_proteins[orga])
            test_data, test_label = build_data(test_proteins[orga])

            print(f">>>> {orga}: train acid classifier for {k}th layer")
            
            model_ckpt_fp = os.path.join(enc_mlp_save_dir, orga+".ckpt")
            if os.path.exists(model_ckpt_fp):
                clf = joblib.load(model_ckpt_fp)
            else:
                clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
                clf.fit(train_data, train_label)
                joblib.dump(clf, model_ckpt_fp)
            avg_acc, acc_stdev = evaluate(clf, test_data, test_label) 
            enc_kth_results[orga] = (avg_acc, acc_stdev)

            if k < model.encoder.transformer.num_layers:
                for pro in train_proteins[orga] + test_proteins[orga]:
                    pro.set_emb(forward_kth_translayer(model, pro.emb, k))

        enc_kth_result_fp = os.path.join(args.self_dir, 'results', f'{k}.eval')
        with open(enc_kth_result_fp, "w") as f:
            f.writelines([f"{orga}\t{value[0]}\t{value[1]}\n" for orga, value in enc_kth_results.items()])


