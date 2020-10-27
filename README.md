# DTLN-aec
This Repostory contains the pretrained DTLN-aec model for real-time acoustic echo cancellation in TF-lite format. This model was handed in to acoustic echo cancellation challenge ([AEC-Challenge](https://aec-challenge.azurewebsites.net/index.html)) organized by Microsoft. The DTLN-aec model is among the top-five models of the challenge. The results of the AEC-Challenge can be found [here](https://aec-challenge.azurewebsites.net/results.html).

The model was trained on data from the [DNS-Challenge](https://github.com/microsoft/AEC-Challenge) and the [AEC-Challenge](https://github.com/microsoft/DNS-Challenge) reposetories.

The ArXiv preprint with further detail will be published in the comming days.

--

## Contents:

This repository contains three prtrained models of different size: 
* `dtln_aec_128` (model with 128 LSTM units per layer, 1.8M parameters)
* `dtln_aec_256` (model with 256 LSTM units per layer, 3.9M parameters)
* `dtln_aec_512` (model with 512 LSTM units per layer, 10.4M parameters)

The `dtln_aec_512` was handed in to the challenge.
--
## Usage:

First install the depencies from `requirements.txt` 

Afterwards the model can be tested with:
```
$ python run_aec.py -i /folder/with/input/files -o /target/folder/ -m ./pretrained_models/dtln_aec_512
```

Files for testing can be found in the [AEC-Challenge](https://github.com/microsoft/DNS-Challenge) respository. The convention for file names is `*_mic.wav` for the near-end microphone signals and `*_lpb.wav` for the far-end microphone or loopback signals. Some example files will be added later.

## This repository is still under construction.
