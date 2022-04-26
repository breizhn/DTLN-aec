# DTLN-aec
This Repostory contains the pretrained DTLN-aec model for real-time acoustic echo cancellation in TF-lite format. This model was handed in to the acoustic echo cancellation challenge ([AEC-Challenge](https://www.microsoft.com/en-us/research/academic-program/acoustic-echo-cancellation-challenge-icassp-2021/)) organized by Microsoft. The DTLN-aec model reached the 3rd place. The results of the AEC-Challenge can be found [here](https://www.microsoft.com/en-us/research/academic-program/acoustic-echo-cancellation-challenge-icassp-2021/results/).

The model was trained on data from the [DNS-Challenge](https://github.com/microsoft/AEC-Challenge) and the [AEC-Challenge](https://github.com/microsoft/DNS-Challenge) reposetories.

The arXiv preprint can be found [here](https://arxiv.org/pdf/2010.14337.pdf).
Please cite:
```bitbtex
@INPROCEEDINGS{9413510,
  author={Westhausen, Nils L. and Meyer, Bernd T.},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Acoustic Echo Cancellation with the Dual-Signal Transformation LSTM Network}, 
  year={2021},
  volume={},
  number={},
  pages={7138-7142},
  doi={10.1109/ICASSP39728.2021.9413510}
  }

```


Author: Nils L. Westhausen ([Communication Acoustics](https://uol.de/en/kommunikationsakustik) , Carl von Ossietzky University, Oldenburg, Germany)

This code is licensed under the terms of the MIT license.

---

## Contents:

This repository contains three prtrained models of different size: 
* `dtln_aec_128` (model with 128 LSTM units per layer, 1.8M parameters)
* `dtln_aec_256` (model with 256 LSTM units per layer, 3.9M parameters)
* `dtln_aec_512` (model with 512 LSTM units per layer, 10.4M parameters)

The `dtln_aec_512` was handed in to the challenge.

---
## Usage:

First install the depencies from `requirements.txt` 

Afterwards the model can be tested with:
```
$ python run_aec.py -i /folder/with/input/files -o /target/folder/ -m ./pretrained_models/dtln_aec_512
```

Files for testing can be found in the [AEC-Challenge](https://github.com/microsoft/DNS-Challenge) respository. The convention for file names is `*_mic.wav` for the near-end microphone signals and `*_lpb.wav` for the far-end microphone or loopback signals. The folder `audio_samples` contains one audio sample for each condition. The `*_processed.wav` files are created by the `dtln_aec_512` model.

---


