import numpy as np
import pandas as pd
import os
from music21 import *
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as pyplot
import matplotlib.lines as mlines


def open_midi(file):
    mf = midi.MidiFile()
    mf.open(file)
    mf.read()
    mf.close()
    for i in range(len(mf.tracks)):
        mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]       

    return midi.translate.midiFileToStream(mf)

def list_instruments(midi):
    partStream = midi.parts.stream()
    for p in partStream:
        print(p.partName)

#time_sig = midi_file.getTimeSignatures()[0]
#music_analysis = midi_file.analyze('key')

#print("Time Signature: {0}/{1}".format(time_sig.beatCount,time_sig.denominator))
#print("Music Key: {0}".format(music_analysis))
#print("Key Confidence: {0}".format(music_analysis.correlationCoefficient))

#temp_chords = midi_file.chordify()
#temp_midi = stream.Score()
#temp_midi.insert(0,temp_chords)

    
#counts the number of times each note appears in a measure
def note_count(measure, count_dict):

    bass_note = None
    for chord in measure.recurse().getElementsByClass('Chord'):
        note_length = chord.quarterLength
        for note in chord.pitches:
            note_name = str(note)
            if bass_note is None or bass_note.ps > note.ps:
                bass_note = note

            if note_name in count_dict:
                count_dict[note_name] += note_length
            else:
                count_dict[note_name] = note_length

    return bass_note

#simplifies chord names
def simplify_chord(roman_numeral):
    ret = roman_numeral.romanNumeral
    inversion_name = None
    inversion = roman_numeral.inversion()

    if ((roman_numeral.isTriad() and inversion < 3) or (inversion < 4 and (roman_numeral.seventh is not None or roman_numeral.isSeventh()))):
        inversion_name = roman_numeral.inversionName()

    if inversion_name is not None:
        ret = ret + str(inversion_name)
    elif roman_numeral.isDominantSeventh(): ret = ret + 'M7'
    elif roman_numeral.isDiminishedSeventh(): ret = ret + "o7"
    return ret

def extract_chords(midi):
    ret = []
    temp_midi = stream.Score()
    temp_chords = midi_file.chordify()
    temp_midi.insert(0,temp_chords)
    music_key = temp_midi.analyze('key')
    max_notes_per_chord = 4
    
    for m in temp_chords.measures(0,None):
        if type(m) != stream.Measure:
            continue

        count_dict = dict()
        bass_note = note_count(m,count_dict)
        if len(count_dict) < 1:
            ret.append("-")
            continue
        sorted_items = sorted(count_dict.items(),key = lambda x:x[1])
        sorted_notes = [item[0] for item in sorted_items[-max_notes_per_chord:]]
        measure_chord = chord.Chord(sorted_notes)

        roman_numeral = roman.romanNumeralFromChord(measure_chord,music_key)
        ret.append(simplify_chord(roman_numeral))

    return ret

#for root,dirs,files in os.walk('./smallSet'):
#    for name in files:
#        file_name = os.path.join(root,name)
#        midi_file = open_midi(file_name)
#        time_sig = midi_file.getTimeSignatures()[0]    
#        music_analysis = midi_file.analyze('key')
#
#        print(name)
#        print("Time Signature: {0}/{1}".format(time_sig.beatCount,time_sig.denominator))
#        print("Music Key: {0}".format(music_analysis))
#        print("Key Confidence: {0}".format(music_analysis.correlationCoefficient))


midi_file = open_midi('./smallSet/September-1.mid')
print(extract_chords(midi_file))


