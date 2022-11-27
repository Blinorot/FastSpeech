import os
import shutil
from pathlib import Path

import gdown
from speechbrain.utils.data_utils import download_file

URL_LINKS = {
    "final": "https://drive.google.com/u/0/uc?id=1zb7OEMiqafg8-Unh0_e615h2RfZ-SMAi&export=download",
}

def download():
    data_dir = Path(__file__).absolute().resolve().parent.parent
    data_dir = data_dir / "saved" / "models" / "pretrained"
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / 'final.zip'
    if not arc_path.exists():
        gdown.download(URL_LINKS['final'], str(arc_path))
    shutil.unpack_archive(str(arc_path), str(data_dir))
    os.remove(str(arc_path))

if __name__ == '__main__':
    download()
