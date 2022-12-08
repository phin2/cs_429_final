import numpy as np
import pandas as pd
import os
from music21 import *
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as pyplot
import matplotlib.lines as mlines
from mido import MidiFile
import mido

def motion(measure):
    melody = []
    for chord in measure.recurse().getElementsByClass('Chord'):
        melody.append(chord.pitches[len(chord.pitches) - 1])

    return melody


def melodic_motion(midi):
    melody = []
    motion = []
    temp_chords = midi.chordify()
    
    for m in temp_chords.measures(0,None):
        if type(m) != stream.Measure:
            continue
        melody += melodic_motion(m)

    window_len = 4
    prev = 0
    for i in range(0,len(melody) - window_len):
        if melody[i] < melody[i + window_len]:
            if prev + 1 > 1: prev = 0
            motion.append(prev + 1)
            prev = motion[i]
        elif melody[i] > melody[i + window_len]:
            if prev - 1 < -1: prev = 0
            motion.append(prev - 1)
            prev = motion[i]
        else:
            motion.append(prev) 

    print(motion)
    return motion


def open_midi(file):
    mf = midi.MidiFile()
    mf.open(file)
    mf.read()
    mf.close()
    for i in range(len(mf.tracks)):
        mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]       

    return midi.translate.midiFileToStream(mf)


midi = open_midi('./smallSet/September-1.mid')
extract_chords(midi)