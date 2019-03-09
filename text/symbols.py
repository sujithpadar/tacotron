'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''

''' Symbols for the usual text representation of input characters
Writing the custom representation for Global phoneme text transcripts
'''

'''
from text import cmudict
#_characters = ' bdefghijklmnoprstuvyæøŋɑː'
_pad        = '_'
_eos        = '~'
#_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '
_characters = ' ːæɑbdefghijklmnŋoøprɾstuvy' 
#Removing the error cased by the previous dictionary with r
# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):

_arpabet = ['@' + s for s in cmudict.valid_symbols]
# Export all symbols:
symbols = [_pad, _eos] + list(_characters)  + _arpabet

'''
import pickle

global_phone_rep = pickle.load(open('phone_attributes.pickle', 'rb'))

id_to_phone = global_phone_rep['phones']
symbols = list(id_to_phone.values())
phone_to_id = {value: key for key, value in id_to_phone.items()}

def phone_to_sequence(text):
    sequence = [phone_to_id[s] for s in text.split(' ')]
    return sequence


glob_ph_attribute_vector = global_phone_rep['phone_attribute_vectors']
