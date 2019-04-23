import numpy as np
import os
import pickle
from speakers.preprocess_speaker_embedding_speecon import feature_extraction, extract_embeddings,load_model
from speakers.preprocess_speaker_embedding_speecon import SphericalKMeans

'''
This script Generates the Speaker Embedding table for models/tacotron.py
New speaker embedding is added at the position 230 as in speecon dataset we don't have any data for
the speaker 231. So this location is used by default for New speaker.
During evaluation if we want to use a new speaker, Using the wav files of that speaker,
a new speaker embedding is generated and passed here. The Embedding should be generated using the same 
speaker embedding modela s used for training the model.
'''

# Add New Speaker Embedding at the position 230.[corresponds to speaker 231 who does not exist in traing data]
# This is done so that, the speaker numbers need not be changed for training.
def get_spk_emb_table(is_training):
    '''
    Returns the speaker embedding table based whether it is called during training or testing.
    :param is_training:  Indicator from tacotron if it is training or evaluating
    :return: Returns the speaker embedding
    '''
    if is_training:
        spk_emb = np.loadtxt('speakers/spk_emb_200_AkuNet.256.200.2.2000.1000._embedding_training')
    else:
        spk_emb = np.loadtxt('speakers/spk_emb_200_AkuNet.256.200.2.2000.1000._embedding_evaluate')

    # Checking if the speaker embeddings are not corrupted by comparing it with the saved dictionary
    dic_spk_emb = pickle.load(open('speakers/spk_emb_200_AkuNet.256.200.2.2000.1000._embedding.pickle', 'rb'))
    spk_list = list(dic_spk_emb.keys())
    for speaker in spk_list:
        if dic_spk_emb[speaker].all() == spk_emb[int(speaker) - 1].all():
            continue
        else:
            print('Error while is_training = ', str(is_training))
            print('Corrupted Speaker Embeddings, Please check the model and the embedding file')
    return spk_emb

def gen_new_embedding_table(dir_path):
    '''
    Generates the speaker embedding table based on all the wav files in that directory.
    :param dir_path: Path to the directory with the wave files.
    :return: None, Saves the new embedding table
    '''

    spk_emb = np.loadtxt('speakers/spk_emb_200_AkuNet.256.200.2.2000.1000._embedding_training')
    new_speaker_embedding = gen_emb(dir_path)
    spk_emb[230, :] = new_speaker_embedding[np.newaxis, :]
    np.savetxt('speakers/spk_emb_200_AkuNet.256.200.2.2000.1000._embedding_evaluate', spk_emb)
    return 0

def gen_emb(dir_path):
    '''
    Generates the embedding for the new speaker
    :param dir_path: Path to the directory with the wav files
    :return: 1 embedding for all the wav files
    '''
    model = load_model('speaker_emb_models/AkuNet.256.200.2.2000.1000.')
    files = os.listdir(dir_path)
    for file in files:
        print('Processing:', file)
        combined_emb = np.empty([0, 200])
        if ".wav" in file:
            features = feature_extraction(os.path.join(dir_path,file))
            print('features dim:', features.shape)
            embeddings = extract_embeddings(features, model)
            if len(embeddings) == int(200):
                embeddings = embeddings[np.newaxis, :]
            combined_emb = np.concatenate([combined_emb, embeddings])

    print('calculated all embeddings for the speaker')
    if len(combined_emb) is not 0:
        spkmeans = SphericalKMeans(n_clusters=1,
                                   init='k-means++',
                                   max_iter=10, n_init=1, n_jobs=1).fit(combined_emb)
        spk_emb = spkmeans.cluster_centers_
        return spk_emb
    else:
        print('Pass a valid path with wav files, no wav files found in ',dir_path)
        return 1
