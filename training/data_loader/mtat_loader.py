# coding: utf-8
import os
import numpy as np
from torch.utils import data

class AudioFolder(data.Dataset):
    def __init__(self, root, dataset=None, input_length=None):
        self.root = root
        self.dataset = dataset
        self.input_length = input_length
        self.get_songlist()
        self.binary = np.load(os.path.join(self.root, 'data/binary.npy'))

    def __getitem__(self, index):
        raw, tag_binary = self.get_raw(index)
        return raw.astype('float32'), tag_binary.astype('float32')

    def get_songlist(self):
        if self.trval == 'TRAIN':
            self.fl = np.load(os.path.join(self.root, 'data/train_new.npy'))
        elif self.trval == 'VALID':
            self.fl = np.load(os.path.join(self.root, 'data/valid_new.npy'))
        elif self.trval == 'TEST':
            self.fl = np.load(os.path.join(self.root, 'data/test_new.npy'))

    def get_raw(self, index):
        ix, fn = self.fl[index].split('\t')
        raw_path = os.path.join(self.root, 'raw', fn.split('/')[1][:-3]) + 'npy'
        raw = np.load(raw_path, mmap_mode='r')
        if self.trval == 'TRAIN' or self.trval == 'VALID':
            random_idx = int(np.floor(np.random.random(1) * ((29*16000)-self.input_length)))
            raw = np.array(raw[random_idx:random_idx+self.input_length])
        tag_binary = self.binary[int(ix)]
        return raw, tag_binary

    def __len__(self):
        return len(self.fl)


def get_audio_loader(root, batch_size, num_workers=0, dataset=None, input_length=None):
    data_loader = data.DataLoader(dataset=AudioFolder(root, dataset=dataset, input_length=input_length),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader

