import os
import shutil
from pathlib import Path

import numpy as np
import pyworld as pw
import torch
import torchaudio
from scipy.interpolate import interp1d
from tqdm.auto import tqdm


def get_pitch():
    data_dir = Path(__file__).absolute().resolve().parent.parent
    wav_dir = data_dir / "data" / "LJSpeech-1.1" / "wavs"
    mel_dir = data_dir / "data" / "mels"
    save_dir = data_dir / "data" / "pitch"
    save_dir.mkdir(exist_ok=True, parents=True)

    assert wav_dir.exists(), "Wav dir not found, download data first"
    
    names = []
    for fpath in wav_dir.iterdir():
        names.append(fpath.name)

    names_dict = {name: i for i, name in enumerate(sorted(names))}

    min_pitch = 1e10
    max_pitch = 1e-10

    for fpath in tqdm(wav_dir.iterdir(), total=len(names)):
        real_i = names_dict[fpath.name]
        new_name = "ljspeech-pitch-%05d.npy" % (real_i+1)
        mel_name = "ljspeech-mel-%05d.npy" % (real_i+1)

        mel = np.load(mel_dir / mel_name)

        audio, sr = torchaudio.load(fpath)
        audio = audio.to(torch.float64).numpy().sum(axis=0)

        # with frame_period=k  k * f0.shape == len(audio) in seconds * 1000
        # we want mel.shape[0] frames, we want f0.shape = mel.shape[0]
        # frame_period = (len(audio) / sr) * 1000  / mel.shape[0]

        frame_period = (audio.shape[0] / sr * 1000) / mel.shape[0]

        _f0, t = pw.dio(audio, sr, frame_period=frame_period)
        f0 = pw.stonemask(audio, _f0, t, sr)[:mel.shape[0]]

        nonzeros = np.nonzero(f0)

        x = np.arange(f0.shape[0])[nonzeros]

        values = (f0[nonzeros][0], f0[nonzeros][-1])

        f = interp1d(x, f0[nonzeros], bounds_error=False, fill_value=values)

        new_f0 = f(np.arange(f0.shape[0]))

        np.save(save_dir / new_name, new_f0)

        min_pitch = min(min_pitch, new_f0.min())
        max_pitch = max(max_pitch, new_f0.max())

    print('min:', min_pitch, 'max:', max_pitch)

if __name__ == '__main__':
    get_pitch()
