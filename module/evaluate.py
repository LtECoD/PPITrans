import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

def plot_eval_predictions(labels, predictions, _path):
    """
    Plot histogram of positive and negative predictions, precision-recall curve, and receiver operating characteristic curve.

    :param y: Labels
    :type y: np.ndarray
    :param phat: Predicted probabilities
    :type phat: np.ndarray
    :param path: File prefix for plots to be saved to [default: figure]
    :type path: str
    """
    metrics = {}

    pos_phat = predictions[labels == 1]
    neg_phat = predictions[labels == 0]

    acc = accuracy_score(labels, predictions>0.5)
    precision = precision_score(labels, predictions>0.5)
    recall = recall_score(labels, predictions>0.5)
    metrics["ACC"] = acc
    metrics["Precision"] =  precision
    metrics["Recall"] =  recall
    metrics["F1"] =  2*precision*recall/(precision+recall)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Distribution of Predictions")
    ax1.hist(pos_phat)
    ax1.set_xlim(0, 1)
    ax1.set_title("Positive")
    ax1.set_xlabel("p-hat")
    ax2.hist(neg_phat)
    ax2.set_xlim(0, 1)
    ax2.set_title("Negative")
    ax2.set_xlabel("p-hat")
    plt.savefig(_path + ".phat_dist.png")
    plt.close()

    precision, recall, pr_thresh = precision_recall_curve(labels, predictions)
    aupr = average_precision_score(labels, predictions)
    metrics["AUPR"] =  aupr

    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall (AUPR: {:.3})".format(aupr))
    plt.savefig(_path + ".aupr.png")
    plt.close()

    fpr, tpr, roc_thresh = roc_curve(labels, predictions)
    auroc = roc_auc_score(labels, predictions)
    metrics["AUROC"] =  auroc

    plt.step(fpr, tpr, color="b", alpha=0.2, where="post")
    plt.fill_between(fpr, tpr, step="post", alpha=0.2, color="b")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Receiver Operating Characteristic (AUROC: {:.3})".format(auroc))
    plt.savefig(_path + ".auroc.png")
    plt.close()

    lines = [f"{k}:\t{v}\n" for k, v in metrics.items()]
    for line in lines:
        print(line, end="")
    with open(_path+".eval", "w") as f:
        f.writelines(lines)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--metric_dir", type=str)
    parser.add_argument("--split", type=str)
    args = parser.parse_args()

    labels = []
    predictions = []
    with open(os.path.join(args.result_dir, args.split+".txt"), "r") as f:
        for idx, line in enumerate(f.readlines()):
            pro1, pro2, label, p0, p1 = line.strip().split()
            labels.append(int(label))
            predictions.append(float(p1))
    
    labels = np.array(labels)
    predictions = np.array(predictions)
    if not os.path.exists(args.metric_dir):
        os.mkdir(args.metric_dir)
    print(f">>>>>>>>>>{args.split}<<<<<<<<<")
    plot_eval_predictions(labels=labels, predictions=predictions, \
        _path=os.path.join(args.metric_dir, args.split))