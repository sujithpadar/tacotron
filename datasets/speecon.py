from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import librosa
import re
from util import audio


_min_samples = 2000
_threshold_db = 25

def build_from_path(in_dir, out_dir, transcript_filename, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the Speecon Speech dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the speecon dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallelize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  with open(os.path.join(in_dir, transcript_filename), encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split(',')
      wav_path = os.path.join(in_dir, '%s.wav' % parts[0].split('.')[0])
      file_name = parts[0].split('/')[3].split('.')[0]
      text = parts[2]
      speaker_id = int(parts[1])
      futures.append(executor.submit(partial(_process_utterance, out_dir, file_name, wav_path, text, speaker_id)))
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, file_name, wav_path, text, speaker_id):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    file_name: The file name to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file
    speaker_id: The speaker Id of the audio file.

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''
  #Load the wav file after trimming the file of the silences.
  wav = _trim_wav(audio.load_wav(wav_path))
  # If Loading the audio without trimming silences use:
  #wav = audio.load_wav(wav_path)

  # Compute the linear-scale spectrogram from the wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # Write the spectrograms to disk:
  spectrogram_filename = file_name + '-speecon-spec.npy'
  mel_filename = file_name + '-speecon-mel.npy'
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames, text, speaker_id)


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
