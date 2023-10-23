# coding: utf-8
import argparse

import model as Model
import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn import metrics
from torch.autograd import Variable

from training.datasets import SplitType, get_dataset


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

    def get_model(self):
        if self.model_type == "fcn":
            self.input_length = 29 * 16000
            return Model.FCN()
        elif self.model_type == "musicnn":
            self.input_length = 3 * 16000
            return Model.Musicnn(dataset=self.dataset_name)
        elif self.model_type == "crnn":
            self.input_length = 29 * 16000
            return Model.CRNN()
        elif self.model_type == "sample":
            self.input_length = 59049
            return Model.SampleCNN()
        elif self.model_type == "se":
            self.input_length = 59049
            return Model.SampleCNNSE()
        elif self.model_type == "attention":
            self.input_length = 15 * 16000
            return Model.CNNSA()
        elif self.model_type == "hcnn":
            self.input_length = 5 * 16000
            return Model.HarmonicCNN()
        elif self.model_type == "short":
            self.input_length = 59049
            return Model.ShortChunkCNN()
        elif self.model_type == "short_res":
            self.input_length = 59049
            return Model.ShortChunkCNN_Res()
        else:
            print(
                "model_type has to be one of [fcn, musicnn, crnn, sample, se, short, short_res, attention]"
            )

    def build_model(self):
        self.model = self.get_model()

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

    def get_auc(self, est_array, gt_array):
        roc_aucs = metrics.roc_auc_score(gt_array, est_array, average="macro")
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average="macro")
        return roc_aucs, pr_aucs

    def test(self):
        roc_auc, pr_auc, loss = self.get_test_score()
        print("loss: %.4f" % loss)
        print("roc_auc: %.4f" % roc_auc)
        print("pr_auc: %.4f" % pr_auc)

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
                [ground_truth.astype("float32") for i in range(self.batch_size)]
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

        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        return roc_auc, pr_auc, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--dataset", type=str, default="mtat", choices=["mtat", "msd", "jamendo"]
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
    parser.add_argument("--model_load_path", type=str, default=".")
    parser.add_argument("--data_path", type=str, default="./data")

    config = parser.parse_args()

    p = Predict(config)
    p.test()
