import numpy as np
import pandas as pd
import os
from music21 import *
from mido import MidiFile
from tqdm import tqdm

def get_time_sig(midi):
    time_sig = midi.getTimeSignatures()[0]
    return np.array([time_sig.numerator, time_sig.denominator])

def get_key_sig(midi):
    key_sig = midi.analyze('key')
    return np.array([key_sig.sharps, 0 if key_sig.mode == 'minor' else 1])

def get_bpm(mido):
    tempo = 500000
    for msg in mido:     # Search for tempo
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            break
    tempo = (1 / tempo) * 60 * 1000000
    return tempo

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

def extract_chords(midi_file):
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

    return np.array(ret)

def get_features(dir):

    features = np.empty(shape=(0, 6))
    song_chords = []
    all_chords = np.array([])

    print("extracting features...")

    for root,dirs,files in os.walk(dir):
        for name in tqdm(files):
            file_name = os.path.join(root,name)
            try:
                midi_file = open_midi(file_name)
                mido_file = MidiFile(file_name)

                time_sig = get_time_sig(midi_file)
                key_sig = get_key_sig(midi_file)

                chords = extract_chords(midi_file)
                bpm = get_bpm(mido_file)
                feature = np.concatenate(([bpm], time_sig, key_sig, [name]))

            except:
                continue

            features = np.append(features, [feature], axis=0)
            song_chords.append(chords)
            all_chords = np.unique(np.append(all_chords, chords))
    
    
    chord_hist = get_chord_hist(song_chords, all_chords)
    features = np.insert(features, [1], chord_hist, axis=1)

    return features

def get_chord_hist(song_chords, all_chords):
    chord_hist = []
    print("generating chord histogram...")
    for chords in tqdm(song_chords):
        song_dict = dict.fromkeys(set(all_chords),0)
        for chord in chords:
                song_dict[chord] += 1
        chord_hist.append(list(song_dict.values()))
    return np.array(chord_hist)

dir_path = input("input directory: ")
out_path = input("output file: ")

df = pd.DataFrame(get_features(dir_path))
print("writing to csv...")
df.to_csv(out_path)
print("done.")