# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 23:08:23 2020

@author: ImNotDani
"""
import torch as t
import torchaudio as ta
import matplotlib.pyplot as plt


filename = "Example/ex.wav" #File path
waveform, sample_rate = ta.load(filename)
specgram = ta.transforms.MelSpectrogram()(waveform)

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))
print("Specgram: {}".format(specgram.size()))

plt.figure()
plt.imshow(specgram.log2()[0,:,:].detach().numpy(), cmap='gray')