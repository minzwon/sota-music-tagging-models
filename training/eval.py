# coding: utf-8
import argparse
from pathlib import Path

import model as Model
import numpy as np
import torch
import torch.nn as nn
import tqdm
from datasets import DATASETS, SplitType, get_dataset
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.autograd import Variable


class Predict(object):
    def __init__(self, config):
        self.model_type = config.model_type
        self.model_load_path = config.model_load_path
        self.data_path = config.data_path
        self.dataset_name = config.dataset
        self.batch_size = config.batch_size
        self.is_cuda = torch.cuda.is_available()
        self.build_model()

        self.dataset = get_dataset(
            config.dataset,
            config.data_path,
            self.input_length,
            config.batch_size,
            SplitType.TEST,
        )

    def build_model(self):
        self.model, self.input_length = Model.get_model(
            self.model_type, self.dataset_name
        )

        # load model
        self.load(self.model_load_path)

        # cuda
        if self.is_cuda:
            self.model.cuda()

    def load(self, filename):
        S = torch.load(filename)
        if "spec.mel_scale.fb" in S.keys():
            self.model.spec.mel_scale.fb = S["spec.mel_scale.fb"]
        self.model.load_state_dict(S)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def test(self):
        est_array, gt_array, loss = self.get_test_score()
        roc_auc_real = metrics.roc_auc_score(gt_array, est_array, average="macro")
        pr_auc_real = metrics.average_precision_score(
            gt_array, est_array, average="macro"
        )

        # calculate pr-auc and roc-auc, and graph both curves

        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        precision, recall, fpr, tpr, roc_auc, pr_auc = {}, {}, {}, {}, {}, {}
        n_classes = gt_array.shape[-1]
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(gt_array[..., i], est_array[..., i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            precision[i], recall[i], _ = metrics.precision_recall_curve(
                gt_array[..., i], est_array[..., i]
            )
            pr_auc[i] = metrics.auc(recall[i], precision[i])

        fpr_grid = np.linspace(0.0, 1.0, 1000)
        recall_grid = np.linspace(0.0, 1.0, 1000)

        # Interpolate all curves at these points
        mean_tpr = np.zeros_like(fpr_grid)
        mean_precision = np.zeros_like(recall_grid)

        for i in range(n_classes):
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
            mean_precision += np.interp(
                recall_grid, recall[i][::-1], precision[i][::-1]
            )  # reverse so recall array is increasing

        # Average it and compute AUC
        mean_tpr /= n_classes
        mean_precision /= n_classes

        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        recall["macro"] = recall_grid
        precision["macro"] = mean_precision
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
        pr_auc["macro"] = metrics.auc(recall["macro"], precision["macro"])

        print("loss: %.4f" % loss)
        print("roc_auc: %.4f" % roc_auc_real)
        print("pr_auc: %.4f" % pr_auc_real)
        plt.figure(1)
        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label=f"macro-average ROC curve (AUC = {roc_auc_real:.2f})",
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()

        plt.figure(2)
        plt.plot(
            recall["macro"],
            precision["macro"],
            label=f"macro-average PR curve (AUC = {pr_auc_real:.2f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.show()

    def get_test_score(self):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = nn.BCELoss()
        for data in tqdm.tqdm(self.dataset.data_list):
            # load and split
            x = self.dataset.get_tensor(data)

            # ground truth
            ground_truth = self.dataset.get_ground_truth(data)

            # forward
            x = self.to_var(x)
            y = torch.tensor(
                np.tile(ground_truth.astype("float32"), (self.batch_size, 1))
            ).cuda()
            out = self.model(x)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()

            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)
            gt_array.append(ground_truth)

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)

        return est_array, gt_array, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        default="mtat",
        choices=list(DATASETS),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="fcn",
        choices=[
            "fcn",
            "musicnn",
            "crnn",
            "sample",
            "se",
            "short",
            "short_res",
            "attention",
            "hcnn",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_load_path", type=str, default="./models")
    parser.add_argument("--data_path", type=str, default="./data")

    config = parser.parse_args()

    model_load_dir: Path = (
        Path(config.model_load_path) / config.dataset / config.model_type
    )
    model_load_dir.mkdir(parents=True, exist_ok=True)

    config.model_load_path = model_load_dir / "best_model.pth"

    p = Predict(config)
    p.test()
