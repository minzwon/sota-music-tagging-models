# coding: utf-8
"""
Deformation codes are borrowed from MUDA
McFee et al., A software framework for musical data augmentation, 2015
https://github.com/bmcfee/muda
"""
import argparse
import csv
import os
import pickle
import subprocess
import tempfile

import librosa
import model as Model
import numpy as np
import soundfile as psf
import torch
import torch.nn as nn
import tqdm
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from torch.autograd import Variable

from training.datasets import SplitType, get_dataset


class Predict(object):
    def __init__(self, config):
        self.model_type = config.model_type
        self.model_load_path = config.model_load_path
        self.dataset_name = config.dataset
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.is_cuda = torch.cuda.is_available()
        self.build_model()
        self.mod = config.mod
        self.rate = config.rate
        self.PRESETS = {
            "radio": ["0.01,1", "-90,-90,-70,-70,-60,-20,0,0", "-5"],
            "film standard": [
                "0.1,0.3",
                "-90,-90,-70,-64,-43,-37,-31,-31,-21,-21,0,-20",
                "0",
                "0",
                "0.1",
            ],
            "film light": [
                "0.1,0.3",
                "-90,-90,-70,-64,-53,-47,-41,-41,-21,-21,0,-20",
                "0",
                "0",
                "0.1",
            ],
            "music standard": [
                "0.1,0.3",
                "-90,-90,-70,-58,-55,-43,-31,-31,-21,-21,0,-20",
                "0",
                "0",
                "0.1",
            ],
            "music light": [
                "0.1,0.3",
                "-90,-90,-70,-58,-65,-53,-41,-41,-21,-21,0,-11",
                "0",
                "0",
                "0.1",
            ],
            "speech": [
                "0.1,0.3",
                "-90,-90,-70,-55,-50,-35,-31,-31,-21,-21,0,-20",
                "0",
                "0",
                "0.1",
            ],
        }
        self.preset_dict = {
            1: "radio",
            2: "film standard",
            3: "film light",
            4: "music standard",
            5: "music light",
            6: "speech",
        }

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
        elif self.model_type == "short":
            self.input_length = 59049
            return Model.ShortChunkCNN()
        elif self.model_type == "short_res":
            self.input_length = 59049
            return Model.ShortChunkCNN_Res()
        elif self.model_type == "attention":
            self.input_length = 15 * 16000
            return Model.CNNSA()
        elif self.model_type == "hcnn":
            self.input_length = 5 * 16000
            return Model.HarmonicCNN()
        else:
            print(
                "model_type has to be one of [fcn, musicnn, crnn, sample, se, short, short_res, attention]"
            )

    def build_model(self):
        self.model = self.get_model()

        # cuda
        if self.is_cuda:
            self.model.cuda()

        # load model
        self.load(self.model_load_path)

    def load(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def modify(self, x, mod_rate, mod_type):
        if mod_type == "time_stretch":
            return self.time_stretch(x, mod_rate)
        elif mod_type == "pitch_shift":
            return self.pitch_shift(x, mod_rate)
        elif mod_type == "dynamic_range":
            return self.dynamic_range_compression(x, mod_rate)
        elif mod_type == "white_noise":
            return self.white_noise(x, mod_rate)
        else:
            print("choose from [time_stretch, pitch_shift, dynamic_range, white_noise]")

    def time_stretch(self, x, rate):
        """
        [2 ** (-.5), 2 ** (.5)]
        """
        return librosa.effects.time_stretch(x, rate)

    def pitch_shift(self, x, rate):
        """
        [-1, 1]
        """
        return librosa.effects.pitch_shift(x, 16000, rate)

    def dynamic_range_compression(self, x, rate):
        """
        [4, 6]
        Music standard & Speech
        """
        return self.sox(x, 16000, "compand", *self.PRESETS[self.preset_dict[rate]])

    @staticmethod
    def sox(x, fs, *args):
        assert fs > 0

        fdesc, infile = tempfile.mkstemp(suffix=".wav")
        os.close(fdesc)
        fdesc, outfile = tempfile.mkstemp(suffix=".wav")
        os.close(fdesc)

        psf.write(infile, x, fs)

        try:
            arguments = ["sox", infile, outfile, "-q"]
            arguments.extend(args)

            subprocess.check_call(arguments)

            x_out, fs = psf.read(outfile)
            x_out = x_out.T
            if x.ndim == 1:
                x_out = librosa.to_mono(x_out)

        finally:
            os.unlink(infile)
            os.unlink(outfile)

        return x_out

    def white_noise(self, x, rate):
        """
        [0.1, 0.4]
        """
        n_frames = len(x)
        noise_white = np.random.RandomState().randn(n_frames)
        noise_fft = np.fft.rfft(noise_white)
        values = np.linspace(1, n_frames * 0.5 + 1, n_frames // 2 + 1)
        colored_filter = np.linspace(1, n_frames / 2 + 1, n_frames // 2 + 1) ** 0
        noise_filtered = noise_fft * colored_filter
        noise = librosa.util.normalize(np.fft.irfft(noise_filtered)) * (x.max())
        if len(noise) < len(x):
            x = x[: len(noise)]
        return (1 - rate) * x + (noise * rate)

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
    parser.add_argument("--mod", type=str, default="time_stretch")
    parser.add_argument("--rate", type=float, default=0)

    config = parser.parse_args()

    p = Predict(config)
    p.test()
