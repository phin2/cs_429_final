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
import time
start_time =time.time()
from multiprocessing import Pool
from multiprocessing import set_start_method
import warnings
warnings.filterwarnings("ignore")

# extracts time signature from midi file
def get_time_sig(midi):
    time_sig = midi.getTimeSignatures()[0]
    return [time_sig.numerator, time_sig.denominator]

# extracts key signature from midi file
def get_key_sig(midi):
    key_sig = midi.analyze('key')
    return [key_sig.sharps, 0 if key_sig.mode == 'minor' else 1]

# extracts bpm from midi file
def get_bpm(mido):
    tempo = 500000
    for msg in mido:     # Search for tempo
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            break
    tempo = (1 / tempo) * 60 * 1000000
    return tempo

# open midi file for analysis using music21
def open_midi(file):
    mf = midi.MidiFile()
    mf.open(file)
    mf.read()
    mf.close()
    for i in range(len(mf.tracks)):
        mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]       

    return midi.translate.midiFileToStream(mf)

# (unused) extracts all instruments used in midi file
def list_instruments(midi):
    partStream = midi.parts.stream()
    inst_arr = []
    for p in partStream:
        inst_arr.append(p.partName)

    return inst_arr
    
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
    if roman_numeral.isDominantSeventh(): ret = ret + 'M7'
    elif roman_numeral.isDiminishedSeventh(): ret = ret + "o7"
    return ret

# extracts an array of chords from midi file
def extract_chords(midi):
    ret = []
    temp_midi = stream.Score()
    temp_chords = midi.chordify()
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

# (unused) attempt to extract a melody from a midi file
def extract_melody(measure):
    melody = []
    for chord in measure.recurse().getElementsByClass('Chord'):
        melody.append(chord.pitches[len(chord.pitches) - 1])

    return melody

# (unused) another attempt to extract a melody from a midi file
def melodic_motion(midi):
    melody = []
    motion = []
    temp_chords = midi.chordify()
    
    for m in temp_chords.measures(0,None):
        if type(m) != stream.Measure:
            continue
        melody += extract_melody(m)

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

    return motion

folder = './largeSet'
# given a midi file extracts the feature vector and writes to a csv
def process(file_name):
    feature_arr = []
    all_chords = []
    try:
        midi_file = open_midi(file_name)

        time_sig = midi_file.getTimeSignatures()[0]    
        key_sig = midi_file.analyze('key')
        song_name = file_name.replace(folder + "/","").replace(".mid","")
        chords = extract_chords(midi_file)
        bpm = get_bpm(MidiFile(file_name))
        feature = [bpm, key_sig, chords,song_name]
        feature_arr.append(feature)
        all_chords += chords

        feature_arr = np.array(feature_arr,dtype='object')
        df = pd.DataFrame(feature_arr,columns=['bpm','key_signature','chords','song_name'])
        print(file_name,flush=True)
        df.to_csv('./temp/' + song_name +'.csv' )

    except Exception as e:
        print('Error:', e)

# main function for the program
def main():
    pool = Pool(8) # sets up multiprocessing to run faster
    file_names = [file for root,dirs,file in os.walk(folder)][0]
    file_names = [folder + '/' + s for s in file_names]
    pool.map(process,file_names) # run preprocess on all midi files
    pool.close()
    pool.join()

# if statement protects against recursively creating threads
if __name__ == '__main__':
    main()
    print(time.time() -  start_time)