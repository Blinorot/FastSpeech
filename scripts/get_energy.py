import os
import shutil
from pathlib import Path

import numpy as np


def get_energy():
    data_dir = Path(__file__).absolute().resolve().parent.parent
    save_dir = data_dir / "data" / "energy"
    mel_dir = data_dir / "data" / "mels"
    save_dir.mkdir(exist_ok=True, parents=True)

    assert mel_dir.exists(), "Mel dir not found, download data first"

    min_energy = 1e10
    max_energy = -1e10

    for fpath in mel_dir.iterdir():
        mel = np.load(fpath)
        energy = np.linalg.norm(mel, axis=-1)
        new_name = fpath.name.replace('mel', 'energy')
        np.save(save_dir / new_name, energy)     

        min_energy = min(min_energy, energy.min())
        max_energy = max(max_energy, energy.max())

    print('min:', min_energy, 'max:', max_energy) # pass this values to config


if __name__ == '__main__':
    get_energy()
