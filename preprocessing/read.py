from pathlib import Path
from typing import Iterable

import fire
import librosa
import numpy as np
import tqdm


class Processor:
    def _get_paths(self, data_dir: Path) -> Iterable[Path]:
        paths = []
        data_dir: Path = Path(data_dir)
        for ext in ("wav", "mp3", "m4a", "flac", "ogg"):
            paths += list(data_dir.rglob(f"*.{ext}"))
        return paths

    def run(self, data_dir: str, sample_rate: float = 16000):
        """Convert audo files to .npy format

        Args:
            data_dir (str): Directory of audio files to convert
            sample_rate (float, optional): Sample rate of .npy files. Defaults to 16000.
        """
        # create npy dir
        data_dir: Path = Path(data_dir)
        npy_dir = data_dir / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        # convert files to npy
        for audio_file in tqdm.tqdm(self._get_paths(data_dir)):
            npy_file = npy_dir / (audio_file.relative_to(data_dir).with_suffix(".npy"))
            if not npy_file.exists():
                try:
                    npy_file.parent.mkdir(parents=True, exist_ok=True)
                    data, _ = librosa.core.load(audio_file, sr=sample_rate)
                    with npy_file.open("wb") as f:
                        np.save(f, data)
                except Exception:
                    # some audio files are broken
                    print(f"Could not convert '{audio_file}'")
                    continue


if __name__ == "__main__":
    p = Processor()
    fire.Fire({"run": p.run})
