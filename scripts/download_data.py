import os
import shutil
from pathlib import Path

import gdown
from speechbrain.utils.data_utils import download_file

URL_LINKS = {
    "LJ": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
    "train.txt": "https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx",
    "waveglow": "https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx",
    "mel": "https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j",
    "alignments": "https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip",
}

def download():
    data_dir = Path(__file__).absolute().resolve().parent.parent
    data_dir = data_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    #download LjSpeech
    arc_path = data_dir / 'LJSpeech-1.1.tar.bz2'
    if not arc_path.exists():
        download_file(URL_LINKS['LJ'], arc_path)
    shutil.unpack_archive(str(arc_path), str(data_dir))
    os.remove(str(arc_path))

    gdown.download(URL_LINKS['train.txt'], str(data_dir / 'train.txt'))

    #download Waveglow
    waveglow_dir = data_dir.parent / 'waveglow' / 'pretrained_model'
    waveglow_dir.mkdir(exist_ok=True, parents=True)
    gdown.download(URL_LINKS['waveglow'], str(waveglow_dir / 'waveglow_256channels.pt'))

    gdown.download(URL_LINKS['mel'], str(data_dir / 'mel.tar.gz'))
    shutil.unpack_archive(str(data_dir / 'mel.tar.gz'), str(data_dir))
    os.remove(str(data_dir / 'mel.tar.gz'))

    #download alignments
    arc_path = data_dir / 'alignments.zip'
    if not arc_path.exists():
        download_file(URL_LINKS['alignments'], arc_path)
    shutil.unpack_archive(str(arc_path), str(data_dir))
    os.remove(str(arc_path))

if __name__ == '__main__':
    download()
