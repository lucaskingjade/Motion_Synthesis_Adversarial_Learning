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


def Decoder(latent_dim,max_len,nb_label,hidden_dim_dec_list,activation_dec_list,
            fully_condition=True,dof = 69):
    latent_input = Input(shape=(latent_dim,), name='latent_input')
    latent_input_seq = RepeatVector(max_len)(latent_input)


    label_input = Input(shape=(max_len,nb_label), name='label_input')
    label_seq   =label_input
    decoded = merge([latent_input_seq, label_seq], mode='concat')

    for i, (dim, activation) in enumerate(zip(hidden_dim_dec_list, activation_dec_list)):
        decoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(decoded)
        if fully_condition:
            decoded = merge([decoded, label_seq], mode='concat')

    decoded = SimpleRNN(output_dim=dof, activation='sigmoid', name='decoder_output', return_sequences=True)(
        decoded)
    return Model(input=[latent_input, label_input], output=decoded, name='Decoder')


if __name__=='__main__':
    import argparse
    paser = argparse.ArgumentParser()
    paser.add_argument('--which_epoch',default=390,type=int)
    paser.add_argument('--nb_test', default=100, type=int)
    args = paser.parse_args()
    root_path = os.getenv('Seq_AAE_V1')
    path_model =root_path +'Training/Conditional_SAAE/Expr_Emilya/expr2510/expr006/'
    #path_model = '../'
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
    latent_codes = encoder.predict(x=[X, Y2], batch_size=100)
    #define decoder model
    Con_SAAE = Conditional_SAAE(latent_dim=50,latent_activation='tanh',
                     hidden_dim_enc_list=[100,100],hidden_dim_dec_list=[100,100],
                     activation_enc_list=['tanh','tanh'],activation_dec_list=['tanh','tanh'])
    Con_SAAE.nb_label =8
    Con_SAAE.max_len = 1000
    Con_SAAE.dof = 69
    Con_SAAE.postprocess = dataset_obj.postprocess
    Con_SAAE.sampling_interval = dataset_obj.sampling_interval
    decoder = Decoder(latent_dim=50,max_len=500*8,nb_label=8,
                      hidden_dim_dec_list=[100,100],activation_dec_list=['tanh','tanh'],
                      fully_condition=True,dof=69)

    Emotion_name = ['Anger', 'Anxiety', 'Joy', 'Neutral', 'Panic Fear', 'Pride', 'Sadness', 'Shame']

    #load decoder
    decoder_name = 'de'+encoder_name[2:]
    # with open(path_model+decoder_name+'.yaml','r') as f:
    #     decoder = model_from_yaml(f)
    decoder.load_weights(path_model+decoder_name+'.h5')
    Y_concat = np.zeros(shape=(len(latent_codes),500*8,8),dtype=np.float32)
    for i in range(8):
        begin = i*500
        end = (i+1)*500
        Y_concat[:,begin:end,i] = 1.

    generated_long_seqs = decoder.predict(x=[latent_codes,Y_concat],verbose=0)
    Con_SAAE.save_generated_seqs(generated_long_seqs,max_vector=max_vector,min_vector=min_vector,suffix='long_reconstruted_seqs_emotion_concate')






