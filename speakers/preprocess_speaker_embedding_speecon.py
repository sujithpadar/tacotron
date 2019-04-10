#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse
import pandas as pd
import os
from multiprocessing import cpu_count
# In[ ]:

# This script will just extract speaker embeddings with Tuomas Kaseva's 
# speaker identification model, if you don't need diarisation
# author: Aku Rouhe

import wavefile
import numpy as np
import librosa
import keras
import sklearn
import os
import pickle

from spherecluster import SphericalKMeans

# This can be changed based on your needs.
SEGMENT_SHIFT = 0.5 #seconds (hop size)

# These parameters need to match the model:
SEGMENT_LENGTH = 2. #seconds
MFCC_SHIFT = 0.01 #seconds
N_MFFC_COMPONENTS = 20

def load_model(modelpath):
    model = keras.models.load_model(modelpath)
    SE_extractor = keras.models.Model(inputs=model.input,
                        outputs=model.layers[-2].output)
    return SE_extractor

def feature_extraction(audiopath):
    rate, sig = wavefile.load(audiopath)
    sig = np.sum(sig, axis=0) #sum to mono
    sig = _trim_wav(sig)
    if len(sig) < 32000:
        sig = np.pad(sig, pad_width=(0, 32000 - len(sig)%32000), mode='constant')
    S = np.transpose(librosa.util.frame(sig,
        int(rate*SEGMENT_LENGTH), 
        int(rate*SEGMENT_SHIFT)))
    mfcc_feats = []
    for frame in S:
        # MFCC
        mfcc_feat = librosa.feature.mfcc(frame, 
                n_mfcc = N_MFFC_COMPONENTS, 
                sr = rate, 
                n_fft = 512, 
                hop_length = int(rate*MFCC_SHIFT))
        mfcc_feat = sklearn.preprocessing.scale(mfcc_feat, axis = 1)
        # Derivatives
        mfcc_d = librosa.feature.delta(mfcc_feat, mode = "nearest")
        mfcc_d2 = librosa.feature.delta(mfcc_feat, order = 2, mode = "nearest")
        x = np.concatenate([mfcc_feat, mfcc_d, mfcc_d2], axis = 0)
        # Energy removed
        x = np.delete(x, 0, axis = 0)
        mfcc_feats.append(x.T) #want input in [frame-dim, mfcc-sequence-dim, mfcc-coefficent-dim]
    return np.array(mfcc_feats)

_min_samples = 2000
_threshold_db = 25


def _trim_wav(wav):
    '''Trims silence from the ends of the wav'''
    splits = librosa.effects.split(wav, _threshold_db, frame_length=1024, hop_length=512)
    return wav[_find_start(splits):_find_end(splits, len(wav))]


def _find_start(splits):
    for split_start, split_end in splits:
        if split_end - split_start > _min_samples:
            return max(0, split_start - _min_samples)
    return 0


def _find_end(splits, num_samples):
    for split_start, split_end in reversed(splits):
        if split_end - split_start > _min_samples:
            return min(num_samples, split_end + _min_samples)
    return num_samples


def extract_embeddings(features, model):
    embeddings = model.predict(features)
    return embeddings 


def save_embeddings(embeddings, outpath):
    np.savetxt(outpath, embeddings)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Generate sigle speaker embedding for Each speaker in the speecon dataser")
    # Add dataset parameter in future if needed
    # parser.add_argument('--dataset', required=True, choices=['blizzard', 'ljspeech', 'vctk', 'speecon'])
    parser.add_argument('--transcript_filename', default='',
                        help='Name of the file with all the information about the indivudual wave files')
    parser.add_argument('--output_dir', default='same_dir_as_transcript_file_dir',
                       help='Path to the directory to save the embeddings')
    parser.add_argument('--base_dir', default='same_dir_as_transcript_file_dir',
                       help='Path to the base directory of the dataset')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    parser.add_argument('--emb_size', type=int, default=256)

    parser.add_argument("--model", help = "path to model to use", 
                default = "./models/EMB_model.250.256.voxceleb.1.2000.625.1")
    parser.add_argument("--segment-shift", type=float,
                        help="how often to extract an embedding, fractions of seconds",
                        default=0.5)
    args = parser.parse_args()
    
    if args.base_dir == 'same_dir_as_transcript_file_dir':
        base_dir = os.path.dirname(args.transcript_filename)
    else:
        base_dir = args.base_dir
            
    if args.output_dir == 'same_dir_as_transcript_file_dir':
        output_dir = os.path.dirname(args.transcript_filename)
    else:
        output_dir = args.output_dir

    SEGMENT_SHIFT = args.segment_shift
    model = load_model(args.model)
    
    transcripts = pd.read_csv(args.transcript_filename, names=['relative_path','speaker_id','transcript'])
    
    uniq_speakers = np.unique(transcripts.speaker_id)

    global_emb = {}
    global_combined_emb = np.empty((0, int(args.emb_size)))
    for speaker in uniq_speakers[:2]:
        transcript_speaker = transcripts[transcripts.speaker_id.apply(lambda x: speaker == x)]
        outfilename = 'spk_emb_' + str(args.emb_size) + str(args.model).split('/')[-1] + 'embedding'
        op_path = os.path.join(output_dir, outfilename)

        if not os.path.exists(op_path):
            combined_emb = np.empty((0, int(args.emb_size)))
            for relative_path in transcript_speaker.relative_path:
                wav_file = os.path.join(base_dir, relative_path)
                features = feature_extraction(wav_file)
                print('features dim:', features.shape)
                embeddings = extract_embeddings(features, model)
                if len(embeddings) == int(args.emb_size):
                    embeddings = embeddings[np.newaxis, :]
                combined_emb = np.concatenate([combined_emb, embeddings])

            print('calculated all embeddings for the speaker')
            spkmeans = SphericalKMeans(n_clusters=1,
                                       init='k-means++',
                                       max_iter=10, n_init=1, n_jobs=1).fit(combined_emb)    
            spk_embeddings = spkmeans.cluster_centers_

            global_emb[str(speaker)] = spk_embeddings
            global_combined_emb = np.concatenate([global_combined_emb, spk_embeddings])

        else:
            print('Not regenerating, Embedding already exists for ', wav_file)

    save_embeddings(global_combined_emb, op_path)
    print("Saved in:", op_path)
    pickle_name = op_path + '.pickle'
    pickle.dump(global_emb, open(pickle_name,'wb'))
