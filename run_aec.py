# -*- coding: utf-8 -*-
"""
Script to process a folder of .wav files with a trained DTLN-aec model. 
This script supports subfolders and names the processed files the same as the 
original. The model expects 16kHz single channel audio .wav files.
The idea of this script is to use it for baseline or comparison purpose.

Example call:
    $python run_aec.py -i /name/of/input/folder  \
                              -o /name/of/output/folder \
                              -m /name/of/the/model

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 27.10.2020

This code is licensed under the terms of the MIT-license.
"""

import soundfile as sf
import numpy as np
import os
import argparse
import tensorflow.lite as tflite

# make GPUs invisible
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def process_file(interpreter_1, interpreter_2, audio_file_name, out_file_name):
    """
    Funtion to read an audio file, rocess it by the network and write the
    enhanced audio to .wav file.

    Parameters
    ----------
    interpreter_1 : TF-LITE INTERPRETER
        TF-lite interpreter of first model part
    interpreter_2 : TF-LITE INTERPRETER
        TF-lite interpreter of second model part

    audio_file_name : STRING
        Name and path of the input audio file.
    out_file_name : STRING
        Name and path of the target file.

    """

    # read audio
    audio, fs = sf.read(audio_file_name)
    lpb, fs_2 = sf.read(audio_file_name.replace("mic.wav", "lpb.wav"))
    # check fs
    if fs != 16000 or fs_2 != 16000:
        raise ValueError("Sampling rate must be 16kHz.")
    # check for single channel files
    if len(audio.shape) > 1 or len(lpb.shape) > 1:
        raise ValueError("Only single channel files are allowed.")
    # check for unequal length
    if len(lpb) > len(audio):
        lpb = lpb[: len(audio)]
    if len(lpb) < len(audio):
        audio = audio[: len(lpb)]
    # set block len and block shift
    block_len = 512
    block_shift = 128
    # save the len of the audio for later
    len_audio = len(audio)
    # pad the audio file
    padding = np.zeros((block_len - block_shift))
    audio = np.concatenate((padding, audio, padding))
    lpb = np.concatenate((padding, lpb, padding))
    # get details from interpreters
    input_details_1 = interpreter_1.get_input_details()
    output_details_1 = interpreter_1.get_output_details()
    input_details_2 = interpreter_2.get_input_details()
    output_details_2 = interpreter_2.get_output_details()
    # preallocate states for lstms
    states_1 = np.zeros(input_details_1[1]["shape"]).astype("float32")
    states_2 = np.zeros(input_details_2[1]["shape"]).astype("float32")
    # preallocate out file
    out_file = np.zeros((len(audio)))
    # create buffer
    in_buffer = np.zeros((block_len)).astype("float32")
    in_buffer_lpb = np.zeros((block_len)).astype("float32")
    out_buffer = np.zeros((block_len)).astype("float32")
    # calculate number of frames
    num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift
    # iterate over the number of frames
    for idx in range(num_blocks):
        # shift values and write to buffer of the input audio
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[
            idx * block_shift : (idx * block_shift) + block_shift
        ]
        # shift values and write to buffer of the loopback audio
        in_buffer_lpb[:-block_shift] = in_buffer_lpb[block_shift:]
        in_buffer_lpb[-block_shift:] = lpb[
            idx * block_shift : (idx * block_shift) + block_shift
        ]

        # calculate fft of input block
        in_block_fft = np.fft.rfft(np.squeeze(in_buffer)).astype("complex64")
        # create magnitude
        in_mag = np.abs(in_block_fft)
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype("float32")
        # calculate log pow of lpb
        lpb_block_fft = np.fft.rfft(np.squeeze(in_buffer_lpb)).astype("complex64")
        lpb_mag = np.abs(lpb_block_fft)
        lpb_mag = np.reshape(lpb_mag, (1, 1, -1)).astype("float32")
        # set tensors to the first model
        interpreter_1.set_tensor(input_details_1[0]["index"], in_mag)
        interpreter_1.set_tensor(input_details_1[2]["index"], lpb_mag)
        interpreter_1.set_tensor(input_details_1[1]["index"], states_1)
        # run calculation
        interpreter_1.invoke()
        # # get the output of the first block
        out_mask = interpreter_1.get_tensor(output_details_1[0]["index"])
        states_1 = interpreter_1.get_tensor(output_details_1[1]["index"])
        # apply mask and calculate the ifft
        estimated_block = np.fft.irfft(in_block_fft * out_mask)
        # reshape the time domain frames
        estimated_block = np.reshape(estimated_block, (1, 1, -1)).astype("float32")
        in_lpb = np.reshape(in_buffer_lpb, (1, 1, -1)).astype("float32")
        # set tensors to the second block
        interpreter_2.set_tensor(input_details_2[1]["index"], states_2)
        interpreter_2.set_tensor(input_details_2[0]["index"], estimated_block)
        interpreter_2.set_tensor(input_details_2[2]["index"], in_lpb)
        # run calculation
        interpreter_2.invoke()
        # get output tensors
        out_block = interpreter_2.get_tensor(output_details_2[0]["index"])
        states_2 = interpreter_2.get_tensor(output_details_2[1]["index"])

        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block)
        # write block to output file
        out_file[idx * block_shift : (idx * block_shift) + block_shift] = out_buffer[
            :block_shift
        ]
    # cut audio to otiginal length
    predicted_speech = out_file[
        (block_len - block_shift) : (block_len - block_shift) + len_audio
    ]
    # check for clipping
    if np.max(predicted_speech) > 1:
        predicted_speech = predicted_speech / np.max(predicted_speech) * 0.99
    # write output file
    sf.write(out_file_name, predicted_speech, fs)


def process_folder(model, folder_name, new_folder_name):
    """
    Function to find .wav files in the folder and subfolders of "folder_name",
    process each .wav file with an algorithm and write it back to disk in the
    folder "new_folder_name". The structure of the original directory is
    preserved. The processed files will be saved with the same name as the
    original file.

    Parameters
    ----------
    model : STRING
        Name of TF-Lite model.
    folder_name : STRING
        Input folder with .wav files.
    new_folder_name : STRING
        Target folder for the processed files.

    """

    # create interpreters
    interpreter_1 = tflite.Interpreter(model_path=model + "_1.tflite")
    interpreter_1.allocate_tensors()
    interpreter_2 = tflite.Interpreter(model_path=model + "_2.tflite")
    interpreter_2.allocate_tensors()

    # empty list for file and folder names
    file_names = []
    directories = []
    new_directories = []
    # walk through the directory
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            # look for .wav files
            if file.endswith("mic.wav"):
                # write paths and filenames to lists
                file_names.append(file)
                directories.append(root)
                # create new directory names
                new_directories.append(root.replace(folder_name, new_folder_name))
                # check if the new directory already exists, if not create it
                if not os.path.exists(root.replace(folder_name, new_folder_name)):
                    os.makedirs(root.replace(folder_name, new_folder_name))
    # iterate over all .wav files
    for idx in range(len(file_names)):

        # process each file with the mode
        process_file(
            interpreter_1,
            interpreter_2,
            os.path.join(directories[idx], file_names[idx]),
            os.path.join(new_directories[idx], file_names[idx]),
        )
        print(file_names[idx] + " processed successfully!")


if __name__ == "__main__":
    # arguement parser for running directly from the command line
    parser = argparse.ArgumentParser(description="data evaluation")
    parser.add_argument("--in_folder", "-i", help="folder with input files")
    parser.add_argument("--out_folder", "-o", help="target folder for processed files")
    parser.add_argument("--model", "-m", help="name of tf-lite model")
    args = parser.parse_args()

    # process the folder
    process_folder(args.model, args.in_folder, args.out_folder)
