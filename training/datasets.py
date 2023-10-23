import csv
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Type

import numpy as np
import torch
from sklearn.naive_bayes import LabelBinarizer
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


class SplitType(str, Enum):
    TRAIN = "TRAIN"
    VALIDATE = "VALIDATE"
    TEST = "TEST"


class Dataset(ABC, TorchDataset):
    data_list: list

    def __init__(
        self, data_path: str, input_length: int, batch_size: int, split: SplitType
    ):
        self.data_path = Path(data_path)
        self.input_length = input_length
        self.batch_size = batch_size
        self.load_dataset(split)

    @abstractmethod
    def load_dataset(self, split: SplitType) -> None:
        pass

    @abstractmethod
    def get_npy_path(self, data) -> Path:
        pass

    @abstractmethod
    def get_ground_truth(self, data) -> np.ndarray:
        pass

    def get_tensor(self, data) -> Tensor:
        npy_path = self.get_npy_path(data)
        raw = np.load(npy_path, mmap_mode="r")
        # split chunk
        length = len(raw)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(raw[i * hop : i * hop + self.input_length]).unsqueeze(0)
        return x

    def get_npy(self, index) -> tuple[np.ndarray, np.ndarray]:
        npy_path = self.get_npy_path(self.data_list[index])
        npy = np.load(npy_path, mmap_mode="r")
        random_idx = int(np.floor(np.random.random(1) * (len(npy) - self.input_length)))
        npy = np.array(npy[random_idx : random_idx + self.input_length])
        tag_binary = self.get_ground_truth(self.data_list[index])
        return npy, tag_binary

    def __getitem__(self, index):
        npy, tag_binary = self.get_npy(index)
        return npy.astype("float32"), tag_binary.astype("float32")

    def __len__(self):
        return len(self.data_list)

    def get_audio_loader(
        self, root, batch_size, split="TRAIN", num_workers=0, input_length=None
    ) -> DataLoader:
        return DataLoader(
            dataset=self.AudioFolder(root, split=split, input_length=input_length),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
        )


SOURCE_DIR = Path(__file__).parent.parent


class MTGJamendoDataset(Dataset):
    TAGS = [
        "genre---downtempo",
        "genre---ambient",
        "genre---rock",
        "instrument---synthesizer",
        "genre---atmospheric",
        "genre---indie",
        "instrument---electricpiano",
        "genre---newage",
        "instrument---strings",
        "instrument---drums",
        "instrument---drummachine",
        "genre---techno",
        "instrument---guitar",
        "genre---alternative",
        "genre---easylistening",
        "genre---instrumentalpop",
        "genre---chillout",
        "genre---metal",
        "mood/theme---happy",
        "genre---lounge",
        "genre---reggae",
        "genre---popfolk",
        "genre---orchestral",
        "instrument---acousticguitar",
        "genre---poprock",
        "instrument---piano",
        "genre---trance",
        "genre---dance",
        "instrument---electricguitar",
        "genre---soundtrack",
        "genre---house",
        "genre---hiphop",
        "genre---classical",
        "mood/theme---energetic",
        "genre---electronic",
        "genre---world",
        "genre---experimental",
        "instrument---violin",
        "genre---folk",
        "mood/theme---emotional",
        "instrument---voice",
        "instrument---keyboard",
        "genre---pop",
        "instrument---bass",
        "instrument---computer",
        "mood/theme---film",
        "genre---triphop",
        "genre---jazz",
        "genre---funk",
        "mood/theme---relaxing",
    ]

    def read_file(tsv_file):
        tracks = {}
        with open(tsv_file) as fp:
            reader = csv.reader(fp, delimiter="\t")
            next(reader, None)  # skip header
            for row in reader:
                track_id = row[0]
                tracks[track_id] = {
                    "path": row[3].replace(".mp3", ".npy"),
                    "tags": row[5:],
                }
        return tracks

    def load_dataset(self, split: SplitType) -> None:
        match split:
            case SplitType.TEST:
                file = SOURCE_DIR / "split/mtg-jamendo/autotagging_top50tags-test.tsv"
            case SplitType.TRAIN:
                file = SOURCE_DIR / "split/mtg-jamendo/autotagging_top50tags-train.tsv"
            case SplitType.VALIDATE:
                file = (
                    SOURCE_DIR
                    / "split/mtg-jamendo/autotagging_top50tags-validation.tsv"
                )
        self.file_dict = MTGJamendoDataset.read_file(file)
        self.data_list = list(self.file_dict.keys())
        self.mlb = LabelBinarizer().fit(MTGJamendoDataset.TAGS)

    def get_npy_path(self, data) -> Path:
        filename = self.file_dict[data]["path"]
        return Path(self.data_path) / "npy" / filename

    def get_ground_truth(self, data) -> np.ndarray:
        return np.sum(self.mlb.transform(self.file_dict[data]["tags"]), axis=0)


class MTaTDataset(Dataset):
    def load_dataset(self, split: SplitType) -> None:
        match split:
            case SplitType.TEST:
                file = SOURCE_DIR / "split/mtat/test.npy"
            case SplitType.TRAIN:
                file = SOURCE_DIR / "split/mtat/train.npy"
            case SplitType.VALIDATE:
                file = SOURCE_DIR / "split/mtat/valid.npy"
        self.data_list = np.load(file)
        self.binary = np.load(SOURCE_DIR / "split/mtat/binary.npy")

    def get_npy_path(self, data) -> Path:
        _, name = data.split("\t")
        return Path(self.data_path) / "npy" / Path(name).with_suffix("npy").name

    def get_ground_truth(self, data) -> np.ndarray:
        ix, _ = data.split("\t")
        return self.binary[int(ix)]


class MSDDataset(Dataset):
    SKIP_FILES = {
        "TRAIISZ128F42684BB",
        "TRAONEQ128F42A8AB7",
        "TRADRNH128E0784511",
        "TRBGHEU128F92D778F",
        "TRCHYIF128F1464CE7",
        "TRCVDKQ128E0790C86",
        "TREWVFM128F146816E",
        "TREQRIV128F1468B08",
        "TREUVBN128F1468AC9",
        "TRDKNBI128F14682B0",
        "TRFWOAG128F14B12CB",
        "TRFIYAF128F14688A6",
        "TRGYAEZ128F14A473F",
        "TRIXPRK128F1468472",
        "TRAQKCW128F9352A52",
        "TRLAWQU128F1468AC8",
        "TRMSPLW128F14A544A",
        "TRLNGQT128F1468261",
        "TROTUWC128F1468AB4",
        "TRNDAXE128F934C50E",
        "TRNHIBI128EF35F57D",
        "TRMOREL128F1468AC4",
        "TRPNFAG128F146825F",
        "TRIXPOY128F14A46C7",
        "TROCQVE128F1468AC6",
        "TRPCXJI128F14688A8",
        "TRQKRKL128F1468AAE",
        "TRPKNDC128F145998B",
        "TRRUHEH128F1468AAD",
        "TRLUSKX128F14A4E50",
        "TRMIRQA128F92F11F1",
        "TRSRUXF128F1468784",
        "TRTNQKQ128F931C74D",
        "TRTTUYE128F4244068",
        "TRUQZKD128F1468243",
        "TRUINWL128F1468258",
        "TRVRHOY128F14680BC",
        "TRWVEYR128F1458A6F",
        "TRVLISA128F1468960",
        "TRYDUYU128F92F6BE0",
        "TRYOLFS128F9308346",
        "TRMVCVS128F1468256",
        "TRZSPHR128F1468AAC",
        "TRXBJBW128F92EBD96",
        "TRYPGJX128F1468479",
        "TRYNNNZ128F1468994",
        "TRVDOVF128F92DC7F3",
        "TRWUHZQ128F1451979",
        "TRXMAVV128F146825C",
        "TRYNMEX128F14A401D",
        "TREGWSL128F92C9D42",
        "TRJKZDA12903CFBA43",
        "TRBGJIZ128F92E42BC",
        "TRVWNOH128E0788B78",
        "TRCGBRK128F146A901",
    }

    def load_dataset(self, split: SplitType) -> None:
        match split:
            case SplitType.TEST:
                file = SOURCE_DIR / "split/msd/filtered_list_test.cP"
            case SplitType.TRAIN:
                file = SOURCE_DIR / "split/msd/filtered_list_train.cP"
            case SplitType.VALIDATE:
                file = SOURCE_DIR / "split/msd/filtered_list_train.cP"
        with file.open("rb") as f:
            data_list = pickle.load(f, encoding="bytes")
        if split == SplitType.VALIDATE:
            data_list = data_list[201680:]
        elif split == SplitType.TRAIN:
            data_list = data_list[:201680]
        self.data_list = [
            value.decode()
            for value in data_list
            if value.decode() not in MSDDataset.SKIP_FILES
        ]
        id2tag_file = SOURCE_DIR / "split/msd/msd_id_to_tag_vector.cP"
        with id2tag_file.open("rb") as f:
            self.id2tag = pickle.load(f, encoding="bytes")

    def load_validation_dataset(self) -> None:
        train_file = SOURCE_DIR / "split/msd/filtered_list_train.cP"
        with train_file.open("rb") as f:
            train_list = pickle.load(f, encoding="bytes")
        val_set = train_list[201680:]
        self.data_list = [
            value.decode()
            for value in val_set
            if value.decode() not in MSDDataset.SKIP_FILES
        ]
        id2tag_file = SOURCE_DIR / "split/msd/msd_id_to_tag_vector.cP"
        with id2tag_file.open("rb") as f:
            self.id2tag = pickle.load(f, encoding="bytes")

    def get_npy_path(self, data) -> Path:
        filename = f"{data[2]}/{data[3]}/{data[4]}/{data}.npy"
        return Path(self.data_path) / "npy" / filename

    def get_ground_truth(self, data) -> np.ndarray:
        return self.id2tag[data].flatten()


DATASETS = {"mtat": MTaTDataset, "msd": MSDDataset, "jamendo": MTGJamendoDataset}


def get_dataset(
    name: str,
    data_path: str,
    input_length: int,
    batch_size: int,
    split: SplitType,
) -> Dataset:
    return DATASETS[name](data_path, input_length, batch_size, split)
