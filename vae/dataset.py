import lzma
import random

import os
import csv
import re
import librosa
import lz4
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from datasets import tqdm
from librosa.feature import melspectrogram
from torch.utils.data import Dataset
import pretty_midi

import mido
import numpy as np
from MIDI_AUTOMATION.constants import voice_struct, VOICE_KEYS

HEADER_SIZE = 5   # 0x43 0x00 0x09 0x20 0x00
CHECKSUM_SIZE = 1

def unpack_voice(voice_bytes):
    """Convert packed DX7 voice bytes -> parameter array"""
    unpacked = voice_struct.unpack(bytes(voice_bytes))
    params = [unpacked[key] for key in VOICE_KEYS]
    return np.array(params)

def read_syx_file(path):
    msgs = mido.read_syx_file(path)

    msg = msgs[0]
    data = msg.data

    body = data[HEADER_SIZE:-CHECKSUM_SIZE]

    voice_size = voice_struct.calcsize()

    i = 0
    start = i * voice_size
    end = start + voice_size

    voice_bytes = body[start:end]
    params = unpack_voice(voice_bytes)

    return params

class SpectrogramMidiStreamingDataset(Dataset):
    def __init__(self, spec_directory, midi_directory, patch_directory):
        self.spec_directory = spec_directory
        self.midi_directory = midi_directory
        self.patch_directory = patch_directory

        files = os.listdir(self.spec_directory)

        self.midi_files = {}
        self.patch_files = {}
        self.spectrograms = {}
        index = 0

        for file in sorted(files):
            parts = file[6:-4].split("-")
            patch_id = parts[0]
            midi_id = parts[1]

            self.midi_files[index] = os.path.join(self.midi_directory, midi_id)
            self.patch_files[index] = os.path.join(self.patch_directory, "patch_" + patch_id)
            self.spectrograms[index] = os.path.join(self.spec_directory, file)

            index += 1

    def __len__(self):
        return len(self.spectrograms.keys())

    def __getitem__(self, idx):
        spectrogram_path = self.spectrograms[idx]
        with open(spectrogram_path, "rb") as f:
            compressed = f.read()
        try:
            decompressed = lz4.frame.decompress(compressed)
        except Exception as e:
            print(spectrogram_path)
            raise Exception(spectrogram_path)

        spectrogram = np.frombuffer(decompressed, dtype=np.float16)
        spectrogram = spectrogram.reshape(128, -1)

        midi_path = self.midi_files[idx]
        midi_file = pretty_midi.PrettyMIDI(midi_path + ".midi")
        notes = midi_file.instruments[0].notes
        notes = torch.tensor([[x.start, x.end, x.pitch, x.velocity] for x in notes])
        notes = torch.cat([notes, torch.tensor([[0, 0, 0, 0] for x in range(12 - len(notes))])], dim=0)
        midi_mask = torch.cat([torch.ones(len(notes)), torch.ones(12 - len(notes))], dim=0)

        patch_path = self.patch_files[idx]
        patch = np.load(patch_path + ".npy")

        return torch.from_numpy(spectrogram.copy()), notes, torch.from_numpy(patch.copy()), midi_mask

class SpectrogramStreamingDataset(Dataset):
    def __init__(self, spec_directory, patch_directory, num=-1):
        self.spec_directory = spec_directory
        self.patch_directory = patch_directory

        files = os.listdir(self.spec_directory)

        self.patch_files = {}
        self.spectrograms = {}
        index = 0

        for file in sorted(files):
            parts = file[6:-4].split("-")
            patch_id = parts[0]

            self.patch_files[index] = os.path.join(self.patch_directory, "patch_" + patch_id)
            self.spectrograms[index] = os.path.join(self.spec_directory, file)

            index += 1

        new_specs = {}
        index = 0
        array_of_pairs = [(k, v) for k, v in self.spectrograms.items()]
        random.shuffle(array_of_pairs)

        if num > 0:
            for key, val in array_of_pairs:
                index += 1
                new_specs[key] = val
                if index > num:
                    break

            self.spectrograms = new_specs

    def __len__(self):
        return len(self.spectrograms.keys())

    def __getitem__(self, idx):
        spectrogram_path = self.spectrograms[idx]
        with open(spectrogram_path, "rb") as f:
            compressed = f.read()
        try:
            decompressed = lz4.frame.decompress(compressed)
        except Exception as e:
            print(spectrogram_path)
            raise Exception(spectrogram_path)

        spectrogram = np.frombuffer(decompressed, dtype=np.float16)
        spectrogram = spectrogram.reshape(128, -1)

        patch_path = self.patch_files[idx]
        patch = np.load(patch_path + ".npy")

        return torch.from_numpy(spectrogram.copy()), torch.from_numpy(patch.copy()), idx
