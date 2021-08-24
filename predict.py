import sys
import tempfile
from pathlib import Path
import os
import torch
import librosa
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cog

sys.path.insert(0, "training")

import model

SAMPLE_RATE = 16000
DATASET = "mtat"
MODEL_NAMES = {
    "Self-attention": "attention",
    "CRNN": "crnn",
    "FCN": "fcn",
    "Harmonic CNN": "hcnn",
    "MusicNN": "musicnn",
    "Sample-level CNN": "sample",
    "Sample-level CNN + Squeeze-and-excitation": "se",
}


class Predictor(cog.Predictor):
    def setup(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.models = {
            "fcn": model.FCN().to(self.device),
            "musicnn": model.Musicnn(dataset=DATASET).to(self.device),
            "crnn": model.CRNN().to(self.device),
            "sample": model.SampleCNN().to(self.device),
            "se": model.SampleCNNSE().to(self.device),
            "attention": model.CNNSA().to(self.device),
            "hcnn": model.HarmonicCNN().to(self.device),
        }
        self.input_lengths = {
            "fcn": 29 * 16000,
            "musicnn": 3 * 16000,
            "crnn": 29 * 16000,
            "sample": 59049,
            "se": 59049,
            "attention": 15 * 16000,
            "hcnn": 5 * 16000,
        }

        for key, mod in self.models.items():
            filename = os.path.join("models", DATASET, key, "best_model.pth")
            state_dict = torch.load(filename, map_location=self.device)
            if "spec.mel_scale.fb" in state_dict.keys():
                mod.spec.mel_scale.fb = state_dict["spec.mel_scale.fb"]
            mod.load_state_dict(state_dict)

        self.tags = np.load("split/mtat/tags.npy")

    @cog.input("input", type=Path, help="Input audio file")
    @cog.input(
        "variant",
        type=str,
        default="Harmonic CNN",
        options=MODEL_NAMES.keys(),
        help="Model variant",
    )
    @cog.input(
        "output_format",
        type=str,
        default="Visualization",
        options=["Visualization", "JSON"],
        help="Output either a bar chart visualization or a JSON blob",
    )
    def predict(self, input, variant, output_format):
        key = MODEL_NAMES[variant]
        model = self.models[key].eval()
        input_length = self.input_lengths[key]
        signal, _ = librosa.core.load(str(input), sr=SAMPLE_RATE)
        length = len(signal)
        hop = length // 2 - input_length // 2
        print("length, input_length", length, input_length)
        x = torch.zeros(1, input_length)
        x[0] = torch.Tensor(signal[hop : hop + input_length]).unsqueeze(0)
        x = Variable(x.to(self.device))
        print("x.max(), x.min(), x.mean()", x.max(), x.min(), x.mean())
        # asdf()
        out = model(x)
        result = dict(zip(self.tags, out[0].detach().numpy().tolist()))

        if output_format == "JSON":
            return result

        result_list = list(sorted(result.items(), key=lambda x: x[1]))
        plt.figure(figsize=[5, 10])
        plt.barh(
            np.arange(len(result_list)), [r[1] for r in result_list], align="center"
        )
        plt.yticks(np.arange(len(result_list)), [r[0] for r in result_list])
        plt.tight_layout()

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        plt.savefig(out_path)
        return out_path
