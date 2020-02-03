# coding: utf-8
import pickle
import os
import numpy as np
from torch.utils import data

skip_files = set(['TRAIISZ128F42684BB', 'TRAONEQ128F42A8AB7', 'TRADRNH128E0784511', 'TRBGHEU128F92D778F',
                 'TRCHYIF128F1464CE7', 'TRCVDKQ128E0790C86', 'TREWVFM128F146816E', 'TREQRIV128F1468B08',
                 'TREUVBN128F1468AC9', 'TRDKNBI128F14682B0', 'TRFWOAG128F14B12CB', 'TRFIYAF128F14688A6',
                 'TRGYAEZ128F14A473F', 'TRIXPRK128F1468472', 'TRAQKCW128F9352A52', 'TRLAWQU128F1468AC8',
                 'TRMSPLW128F14A544A', 'TRLNGQT128F1468261', 'TROTUWC128F1468AB4', 'TRNDAXE128F934C50E',
                 'TRNHIBI128EF35F57D', 'TRMOREL128F1468AC4',  'TRPNFAG128F146825F', 'TRIXPOY128F14A46C7',
                 'TROCQVE128F1468AC6', 'TRPCXJI128F14688A8', 'TRQKRKL128F1468AAE', 'TRPKNDC128F145998B',
                 'TRRUHEH128F1468AAD', 'TRLUSKX128F14A4E50', 'TRMIRQA128F92F11F1', 'TRSRUXF128F1468784',
                 'TRTNQKQ128F931C74D',  'TRTTUYE128F4244068', 'TRUQZKD128F1468243', 'TRUINWL128F1468258',
                 'TRVRHOY128F14680BC', 'TRWVEYR128F1458A6F', 'TRVLISA128F1468960', 'TRYDUYU128F92F6BE0',
                 'TRYOLFS128F9308346', 'TRMVCVS128F1468256', 'TRZSPHR128F1468AAC', 'TRXBJBW128F92EBD96',
                 'TRYPGJX128F1468479', 'TRYNNNZ128F1468994', 'TRVDOVF128F92DC7F3', 'TRWUHZQ128F1451979',
                 'TRXMAVV128F146825C', 'TRYNMEX128F14A401D', 'TREGWSL128F92C9D42', 'TRJKZDA12903CFBA43',
                  'TRBGJIZ128F92E42BC', 'TRVWNOH128E0788B78'])

META_PATH = './../split/msd/'

class AudioFolder(data.Dataset):
    def __init__(self, root, split, input_length=None):
        self.root = root
        self.split = split
        self.input_length = input_length
        self.get_songlist()

        id2tag_file = os.path.join(META_PATH,'msd_id_to_tag_vector.cP')
        self.id2tag = pickle.load(open(id2tag_file,'rb'), encoding='bytes')

    def __getitem__(self, index):
        npy, tag_binary = self.get_npy(index)
        return npy.astype('float32'), tag_binary.astype('float32')

    def get_songlist(self):
        if self.split == 'TRAIN':
            train_file = os.path.join(META_PATH,'filtered_list_train.cP')
            train_list = pickle.load(open(train_file,'rb'), encoding='bytes')
            train_set = train_list[0:201680]
            self.fl = [value for value in train_set if value.decode() not in skip_files]
        elif self.split == 'VALID':
            train_file = os.path.join(META_PATH,'filtered_list_train.cP')
            train_list = pickle.load(open(train_file,'rb'), encoding='bytes')
            val_set = train_list[201680:]
            self.fl = [value for value in val_set if value.decode() not in skip_files]
        elif self.split == 'TEST':
            test_file = os.path.join(META_PATH,'filtered_list_test.cP')
            test_set = pickle.load(open(test_file,'rb'), encoding='bytes')
            self.fl = [value for value in test_set if value.decode() not in skip_files]
        else:
            print('Split should be one of [TRAIN, VALID, TEST]')


    def get_npy(self, index):
        msdid = self.fl[index].decode()
        filename = '{}/{}/{}/{}.npy'.format(msdid[2], msdid[3], msdid[4], msdid)
        npy_path = os.path.join(self.root, filename)
        npy = np.load(npy_path, mmap_mode='r')
        random_idx = int(np.floor(np.random.random(1) * (len(npy)-self.input_length)))
        npy = np.array(npy[random_idx:random_idx+self.input_length])
        tag_binary = self.id2tag[msdid.encode()].flatten()
        return npy, tag_binary

    def __len__(self):
        return len(self.fl) - len(skip_files)


def get_audio_loader(root, batch_size, split='TRAIN', num_workers=0, input_length=None):
    data_loader = data.DataLoader(dataset=AudioFolder(root, split=split, input_length=input_length),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader

