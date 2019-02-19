#1. reconstruct a sequence but using a different labels

import numpy as np
import os
from keras.models import model_from_yaml
from Seq_AAE_V1.models.Conditional_SAAE.conditional_saae import *

def convert_indices_2_onehot( targets, nb_labels):
    tmp_targets = targets.astype(int)
    ohm = np.zeros((tmp_targets.shape[0], nb_labels), dtype=np.float32)
    ohm[np.arange(tmp_targets.shape[0]), tmp_targets] = 1.0
    return ohm

if __name__=='__main__':
    import argparse
    paser = argparse.ArgumentParser()
    paser.add_argument('--which_epoch',default=390,type=int)
    paser.add_argument('--nb_test', default=200, type=int)
    paser.add_argument('--label_encoder',default=True,action='store_false')
    args = paser.parse_args()
    root_path = os.getenv('Seq_AAE_V1')
    #path_model =root_path +'Training/Conditional_SAAE/Expr_Emilya/expr2510/expr006/'
    path_model = '../'
    encoder_name='encoder'+str(args.which_epoch)
    with open(path_model+encoder_name+'.yaml','r') as f:
        encoder = model_from_yaml(f)
    encoder.load_weights(path_model+encoder_name+'.h5')

    ##load some walking data from dataset
    from Seq_AAE_V1.datasets.dataset import Emilya_Dataset

    dataset_obj = Emilya_Dataset(window_width=200, shift_step=20,
                                 sampling_interval=None,
                                 with_velocity=False,
                                 number=2, nb_valid=2, nb_test=args.nb_test)

    X = dataset_obj.test_X[:,:,1:]
    Y1 = dataset_obj.test_Y1
    Y2 = dataset_obj.test_Y2
    # only choose 'walking' activity
    indices = np.where(Y1 == 2)[0]
    X = X[indices]
    Y1 = Y1[indices]
    Y2 = Y2[indices]
    Y1 = convert_indices_2_onehot(Y1,nb_labels=8)
    Y2 = convert_indices_2_onehot(Y2, nb_labels=8)
    max_vector = dataset_obj.max_vector[1:]
    min_vector = dataset_obj.min_vector[1:]
    # get the latent codes using its true emotion label
    if args.label_encoder ==True:
        latent_codes = encoder.predict(x=[X, Y2], batch_size=100)
    else:
        latent_codes = encoder.predict(x=X, batch_size=100)
    #define decoder model
    Con_SAAE = Conditional_SAAE(latent_dim=50,latent_activation='tanh',
                     hidden_dim_enc_list=[100,100],hidden_dim_dec_list=[100,100],
                     activation_enc_list=['tanh','tanh'],activation_dec_list=['tanh','tanh'])
    Con_SAAE.nb_label =8
    Con_SAAE.max_len = 1000
    Con_SAAE.dof = 69
    Con_SAAE.postprocess = dataset_obj.postprocess
    Con_SAAE.sampling_interval = dataset_obj.sampling_interval
    decoder = Con_SAAE.Decoder()

    Emotion_name = ['Anger', 'Anxiety', 'Joy', 'Neutral', 'Panic Fear', 'Pride', 'Sadness', 'Shame']

    #load decoder
    decoder_name = 'de'+encoder_name[2:]
    # with open(path_model+decoder_name+'.yaml','r') as f:
    #     decoder = model_from_yaml(f)
    decoder.load_weights(path_model+decoder_name+'.h5')
    for y in range(8):
        tmp_Y = np.asarray([y]*len(latent_codes))
        tmp_Y = convert_indices_2_onehot(tmp_Y,8)
        reconstructed_seqs = decoder.predict(x=[latent_codes,tmp_Y],verbose=0)
        Con_SAAE.save_generated_seqs(reconstructed_seqs,max_vector=max_vector,min_vector=min_vector,suffix='long_reconstruted_seqs_emotion_'+Emotion_name[y])






