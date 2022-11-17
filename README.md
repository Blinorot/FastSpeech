# FastSpeech

This is a repository for TTS homework of HSE DLA Course. The implemented model if FastSpeech2 trained on LJSpeech.

## Getting Started

These instructions will help you to run a project on your machine.

### Prerequisites

Install [pyenv](https://github.com/pyenv/pyenv#installation) following the instructions in the official repository.

### Installation

Clone the repository into your local folder:

```bash
cd path/to/local/folder
git clone https://github.com/Blinorot/FastSpeech.git
```

Install `python 3.9.7.` for `pyenv` and create virtual environment:

```bash
pyenv install 3.9.7
cd path/to/cloned/FastSpeech/project
~/.pyenv/versions/3.9.7/bin/python -m venv fs_env
```

Install required python packages to python environment:

```bash
source fs_env/bin/activate
pip install -r requirements.txt
```

### Data downloading

To download data run the following command:

```bash
python3 scripts/download_data.py
```

## Project structure

Repository is structured in the following way.

-   `src` includes code for all used objects and functions including:

    -   `base`: base class code for models and trainers in the correspongding `base_name.py` file.

    -   `collate_fn`: code for the corresponding function of the dataloader.

    -   `configs`: configs for model training in `config_name.json`.

    -   `datasets`: code for LJSpeech dataset.

    -   `logger`: code for different loggers (including W&B) and some utils for visualization.

    -   `loss`: definition of loss function for FastSpeech.

    -   `model`: code for FastSpeech model and its subparts in `blocks` subdir.

    -   `trainer`: code for training models.

    -   `utils`: basic utils including `parse_config.py` for parsing `.json` configs, `object_loading.py` for dataloaders structuring and `util.py` for files\device control and metrics wrapper.

-   `data` folder consists of downloaded datasets folders created by running `download_data.py` script.

-   `saved` folder consists of logs and model checkpoints \ their configs in `log` and `models` subdirs respectively.

-   `scripts` folder consists of different `.py` files for downloading data and Pre-trained models.

-   `results` folder consists of synthesized test speech.

-   `requirements.txt` includes all packages required to run the project.

-   `train.py` script for training models.

-   `sythesize.py` script for synthesizing speech.

## Training

To train model run the following script:

```
python3 train.py -c src/configs/config_name.json
```

To resume training from checkpoint (resume optimizers, schedulers etc.)

```
python3 train.py -r path\to\saved\checkpoint.pth
```

To train model with initialization from checkpoint:

```
python3 train.py -c src/configs/config_name.json \
    -p path\to\saved\checkpoint.pth
```

## Authors

-   Petr Grinberg

## License

This project is licensed under the [MIT](LICENSE) License - see the [LICENSE](LICENSE) file for details.

## Credits

Base code was taken from [DLA Seminar 7](https://github.com/markovka17/dla/blob/2022/week07/FastSpeech_sem.ipynb). Project template was taken from [ASR repository](https://github.com/Blinorot/ASR). WaveGlow implementation, some data and some utils were taken from [FastSpeech repository](https://github.com/xcmyz/FastSpeech)
