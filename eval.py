import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from speakers.embeddings import gen_new_embedding_table

sentences = [
  # From July 8, 2017 New York Times:
  'Scientists at the CERN laboratory say they have discovered a new particle.',
  'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
  'President Trump met with other leaders at the Group of 20 conference.',
  'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
  # From Google's Tacotron example page:
  'Generative adversarial network or variational auto-encoder.',
  'The buses aren\'t the problem, they actually provide a solution.',
  'Does the quick brown fox jump over the lazy dog?',
  'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
]


def get_output_base_path(checkpoint_path,text_file_name):
  base_dir = os.path.dirname(checkpoint_path)
  base_dir = os.path.join(base_dir,text_file_name.split('/')[-1].split('.')[0])
  if not os.path.exists(base_dir):
      os.makedirs(base_dir)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  if args.speaker_dir is not 'None':
      gen_new_embedding_table(args.speaker_dir)
  synth = Synthesizer()
  synth.load(args.checkpoint,args.num_speakers)
  base_path = get_output_base_path(args.checkpoint,args.text_file)
  with open(args.text_file) as file:
      for line in file:
          parts = line.split('|')
          i = parts[0]
          text = parts[1]
          speaker = int(parts[2])
          path = '%s-%s-%d.wav' % (base_path, i, speaker)
          print('Synthesizing: %s' % path)
          with open(path, 'wb') as f:
              f.write(synth.synthesize(text, speaker))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--text_file', required=True, help='Path to text file to generate <id>|<sentence>|<speaker id>')
  parser.add_argument('--speaker_dir', default='None', help='Path to the folder with wav files of the new speaker')
  parser.add_argument('--num_speakers', required=True, help='number of speakers during the training')
  # parser.add_argument('--speaker', type=int, default=374, help='Speaker ID')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
