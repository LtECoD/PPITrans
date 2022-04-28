import os
import argparse
import random
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

import sys
sys.path.append(".")
from module.model import PPIModel
from experiments.utils import load_proteins
from experiments.utils import organisms, acids
from experiments.utils import load_model, forward_module


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

    return np.average(accs), max(accs), min(accs)


def build_data(proteins):
    """构建氨基酸分类的数据集"""
    data = np.vstack([pro.emb for pro in proteins[orga]])
    label = list("".join([pro.seq[:len(pro.emb)] for pro in proteins[orga]]))
    label = [acids.index(l) for l in label]
    return data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--pretrained_emb_dir", type=str, default='./data/dscript/processed/embs')
    parser.add_argument("--self_dir", type=str, default="./experiments/acid_classify")
    parser.add_argument("--protein_dir", type=str, default="./experiments/proteins")
    parser.add_argument("--model_dir", type=str, default="./save/dscript/ppi-wopool", help="saved model without pooling layer")
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
        for pro in train_proteins[orga]:
            emb = np.load(os.path.join(args.pretrained_emb_dir, orga+"_test", pro.name+".npy"))
            pro.set_emb(emb)
        for pro in test_proteins[orga]:
            emb = np.load(os.path.join(args.pretrained_emb_dir, orga+"_test", pro.name+".npy"))
            pro.set_emb(emb)

        # build dataset
        train_data, train_label = build_data(train_proteins)
        test_data, test_label = build_data(test_proteins)
        print(orga)
        print('train', len(train_data))
        print('test', len(test_data))

        print(f">>>> train acid classifier for {orga}")
        model_ckpt_fp = os.path.join(emb_mlp_save_dir, orga+".ckpt")
        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
            clf.fit(train_data, train_label)
            joblib.dump(clf, model_ckpt_fp)

        avg_acc, max_acc, min_acc = evaluate(clf, test_data, test_label) 
        emb_results[orga] = (avg_acc, max_acc, min_acc)
    emb_result_fp = os.path.join(args.self_dir, 'results', 'embed.eval')
    with open(emb_result_fp, "w") as f:
        f.writelines([f"{orga}\t{value[0]}\t{value[1]}\t{value[2]}\n" for orga, value in emb_results.items()])
    
    #### 测试ppi model without pooling layers
    # 加载模型
    model = load_model(PPIModel, os.path.join(args.model_dir, 'checkpoint_best.pt'))
    # 测试conv层
    conv_mlp_save_dir = os.path.join(args.self_dir, 'save', 'conv')
    os.makedirs(conv_mlp_save_dir, exist_ok=True)
    conv_results = {}
    for orga in organisms:
        # forward conv layers
        for pro in train_proteins[orga]:
            conv_enc = forward_module(model.encoder.forward_cnn_blocks, pro.emb)
            pro.set_emb(conv_enc)
        for pro in test_proteins[orga]:
            conv_enc = forward_module(model.encoder.forward_cnn_blocks, pro.emb)
            pro.set_emb(conv_enc)

        # build dataset
        train_data, train_label = build_data(train_proteins)
        test_data, test_label = build_data(test_proteins)
        print(orga)
        print('train', train_data.shape)
        print('test', test_data.shape)

        print(f">>>> train acid classifier for {orga}")
        model_ckpt_fp = os.path.join(conv_mlp_save_dir, orga+".ckpt")
        if os.path.exists(model_ckpt_fp):
            clf = joblib.load(model_ckpt_fp)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=1)
            clf.fit(train_data, train_label)
            joblib.dump(clf, model_ckpt_fp)

        avg_acc, max_acc, min_acc = evaluate(clf, test_data, test_label) 
        conv_results[orga] = (avg_acc, max_acc, min_acc)
    emb_result_fp = os.path.join(args.self_dir, 'results', 'conv.eval')
    with open(emb_result_fp, "w") as f:
        f.writelines([f"{orga}\t{value[0]}\t{value[1]}\t{value[2]}\n" for orga, value in conv_results.items()])


