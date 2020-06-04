# coding: utf-8
'''
Deformation codes are borrowed from MUDA
McFee et al., A software framework for musical data augmentation, 2015
https://github.com/bmcfee/muda
'''
import os
import time
import subprocess
import tempfile
import numpy as np
import pandas as pd
import datetime
import tqdm
import csv
import fire
import argparse
import pickle
from sklearn import metrics
import pandas as pd
import librosa
import soundfile as psf

import torch
import torch.nn as nn
from torch.autograd import Variable
from solver import skip_files
from sklearn.preprocessing import LabelBinarizer

import model as Model


TAGS = ['genre---downtempo', 'genre---ambient', 'genre---rock', 'instrument---synthesizer', 'genre---atmospheric', 'genre---indie', 'instrument---electricpiano', 'genre---newage', 'instrument---strings', 'instrument---drums', 'instrument---drummachine', 'genre---techno', 'instrument---guitar', 'genre---alternative', 'genre---easylistening', 'genre---instrumentalpop', 'genre---chillout', 'genre---metal', 'mood/theme---happy', 'genre---lounge', 'genre---reggae', 'genre---popfolk', 'genre---orchestral', 'instrument---acousticguitar', 'genre---poprock', 'instrument---piano', 'genre---trance', 'genre---dance', 'instrument---electricguitar', 'genre---soundtrack', 'genre---house', 'genre---hiphop', 'genre---classical', 'mood/theme---energetic', 'genre---electronic', 'genre---world', 'genre---experimental', 'instrument---violin', 'genre---folk', 'mood/theme---emotional', 'instrument---voice', 'instrument---keyboard', 'genre---pop', 'instrument---bass', 'instrument---computer', 'mood/theme---film', 'genre---triphop', 'genre---jazz', 'genre---funk', 'mood/theme---relaxing']

def read_file(tsv_file):
    tracks = {}
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            track_id = row[0]
            tracks[track_id] = {
                'path': row[3].replace('.mp3', '.npy'),
                'tags': row[5:],
            }
    return tracks


class Predict(object):
    def __init__(self, config):
        self.model_type = config.model_type
        self.model_load_path = config.model_load_path
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.is_cuda = torch.cuda.is_available()
        self.build_model()
        self.get_dataset()
        self.mod = config.mod
        self.rate = config.rate
        self.PRESETS = {
                        "radio":            ["0.01,1", "-90,-90,-70,-70,-60,-20,0,0", "-5"],
                        "film standard":    ["0.1,0.3", "-90,-90,-70,-64,-43,-37,-31,-31,-21,-21,0,-20", "0", "0", "0.1"],
                        "film light":       ["0.1,0.3", "-90,-90,-70,-64,-53,-47,-41,-41,-21,-21,0,-20", "0", "0", "0.1"],
                        "music standard":   ["0.1,0.3", "-90,-90,-70,-58,-55,-43,-31,-31,-21,-21,0,-20", "0", "0", "0.1"],
                        "music light":      ["0.1,0.3", "-90,-90,-70,-58,-65,-53,-41,-41,-21,-21,0,-11", "0", "0", "0.1"],
                        "speech":           ["0.1,0.3", "-90,-90,-70,-55,-50,-35,-31,-31,-21,-21,0,-20", "0", "0", "0.1"]
                        }
        self.preset_dict = {1: "radio",
                            2: "film standard",
                            3: "film light",
                            4: "music standard",
                            5: "music light",
                            6: "speech"}
    def get_model(self):
        if self.model_type == 'fcn':
            self.input_length = 29 * 16000
            return Model.FCN()
        elif self.model_type == 'musicnn':
            self.input_length = 3 * 16000
            return Model.Musicnn(dataset=self.dataset)
        elif self.model_type == 'crnn':
            self.input_length = 29 * 16000
            return Model.CRNN()
        elif self.model_type == 'sample':
            self.input_length = 59049
            return Model.SampleCNN()
        elif self.model_type == 'se':
            self.input_length = 59049
            return Model.SampleCNNSE()
        elif self.model_type == 'short':
            self.input_length = 59049
            return Model.ShortChunkCNN()
        elif self.model_type == 'short_res':
            self.input_length = 59049
            return Model.ShortChunkCNN_Res()
        elif self.model_type == 'attention':
            self.input_length = 15 * 16000
            return Model.CNNSA()
        elif self.model_type == 'hcnn':
            self.input_length = 5 * 16000
            return Model.HarmonicCNN()
        else:
            print('model_type has to be one of [fcn, musicnn, crnn, sample, se, short, short_res, attention]')

    def build_model(self):
        self.model = self.get_model()

        # cuda
        if self.is_cuda:
            self.model.cuda()

        # load model
        self.load(self.model_load_path)

    def get_dataset(self):
        if self.dataset == 'mtat':
            self.test_list = np.load('./../split/mtat/test.npy')
            self.binary = np.load('./../split/mtat/binary.npy')
        if self.dataset == 'msd':
            test_file = os.path.join('./../split/msd','filtered_list_test.cP')
            test_list = pickle.load(open(test_file,'rb'), encoding='bytes')
            self.test_list = [value for value in test_list if value.decode() not in skip_files]
            id2tag_file = os.path.join('./../split/msd', 'msd_id_to_tag_vector.cP')
            self.id2tag = pickle.load(open(id2tag_file,'rb'), encoding='bytes')
        if self.dataset == 'jamendo':
            test_file = os.path.join('./../split/mtg-jamendo', 'autotagging_top50tags-test.tsv')
            self.file_dict= read_file(test_file)
            self.test_list= list(self.file_dict.keys())
            self.mlb = LabelBinarizer().fit(TAGS)

    def load(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def get_tensor(self, fn):
        # load audio
        if self.dataset == 'mtat':
            npy_path = os.path.join(self.data_path, 'mtat', 'npy', fn.split('/')[1][:-3]) + 'npy'
        elif self.dataset == 'msd':
            msid = fn.decode()
            filename = '{}/{}/{}/{}.npy'.format(msid[2], msid[3], msid[4], msid)
            npy_path = os.path.join(self.data_path, filename)
        elif self.dataset == 'jamendo':
            filename = self.file_dict[fn]['path']
            npy_path = os.path.join(self.data_path, filename)
        raw = np.load(npy_path, mmap_mode='r')
        raw = self.modify(raw, self.rate, self.mod)

        # split chunk
        length = len(raw)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(raw[i*hop:i*hop+self.input_length]).unsqueeze(0)
        return x

    def modify(self, x, mod_rate, mod_type):
        if mod_type == 'time_stretch':
            return self.time_stretch(x, mod_rate)
        elif mod_type == 'pitch_shift':
            return self.pitch_shift(x, mod_rate)
        elif mod_type == 'dynamic_range':
            return self.dynamic_range_compression(x, mod_rate)
        elif mod_type == 'white_noise':
            return self.white_noise(x, mod_rate)
        else:
            print('choose from [time_stretch, pitch_shift, dynamic_range, white_noise]')

    def time_stretch(self, x, rate):
        '''
        [2 ** (-.5), 2 ** (.5)]
        '''
        return librosa.effects.time_stretch(x, rate)

    def pitch_shift(self, x, rate):
        '''
        [-1, 1]
        '''
        return librosa.effects.pitch_shift(x, 16000, rate)

    def dynamic_range_compression(self, x, rate):
        '''
        [4, 6]
        Music standard & Speech
        '''
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
        '''
        [0.1, 0.4]
        '''
        n_frames = len(x)
        noise_white = np.random.RandomState().randn(n_frames)
        noise_fft = np.fft.rfft(noise_white)
        values = np.linspace(1, n_frames * 0.5 + 1, n_frames // 2 + 1)
        colored_filter = np.linspace(1, n_frames / 2 + 1, n_frames // 2 + 1) ** 0
        noise_filtered = noise_fft * colored_filter
        noise = librosa.util.normalize(np.fft.irfft(noise_filtered)) * (x.max())
        if len(noise) < len(x):
            x = x[:len(noise)]
        return (1 - rate) * x + (noise * rate)

    def get_auc(self, est_array, gt_array):
        roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
        return roc_aucs, pr_aucs

    def test(self):
        roc_auc, pr_auc, loss = self.get_test_score()
        print('loss: %.4f' % loss)
        print('roc_auc: %.4f' % roc_auc)
        print('pr_auc: %.4f' % pr_auc)

    def get_test_score(self):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = nn.BCELoss()
        for line in tqdm.tqdm(self.test_list):
            if self.dataset == 'mtat':
                ix, fn = line.split('\t')
            elif self.dataset == 'msd':
                fn = line
                if fn.decode() in skip_files:
                    continue
            elif self.dataset == 'jamendo':
                fn = line

            # load and split
            x = self.get_tensor(fn)

            # ground truth
            if self.dataset == 'mtat':
                ground_truth = self.binary[int(ix)]
            elif self.dataset == 'msd':
                ground_truth = self.id2tag[fn].flatten()
            elif self.dataset == 'jamendo':
                ground_truth = np.sum(self.mlb.transform(self.file_dict[fn]['tags']), axis=0)

            # forward
            x = self.to_var(x)
            y = torch.tensor([ground_truth.astype('float32') for i in range(self.batch_size)]).cuda()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='mtat', choices=['mtat', 'msd', 'jamendo'])
    parser.add_argument('--model_type', type=str, default='fcn',
                        choices=['fcn', 'musicnn', 'crnn', 'sample', 'se', 'short', 'short_res', 'attention', 'hcnn'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_load_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--mod', type=str, default='time_stretch')
    parser.add_argument('--rate', type=float, default=0)

    config = parser.parse_args()

    p = Predict(config)
    p.test()






