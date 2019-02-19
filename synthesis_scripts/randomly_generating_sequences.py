#This is used for randomly generating long sequences.

#1. reconstruct a sequence but using a different labels

import numpy as np
import os
from keras.models import model_from_yaml, Model
from Seq_AAE_V1.models.Seq_AAE.seq_aae_new_loss import *
from keras.layers import Input,RepeatVector,LSTM,SimpleRNN

def Decoder(latent_dim,max_len,hidden_dim_dec_list,activation_dec_list,dof):
    latent_input = Input(shape=(latent_dim,), name='latent_input')
    latent_input_seq = RepeatVector(max_len)(latent_input)
    decoded = latent_input_seq
    for i, (dim, activation) in enumerate(zip(hidden_dim_dec_list, activation_dec_list)):
        decoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(decoded)

    decoded = SimpleRNN(output_dim=dof, activation='sigmoid', name='decoder_output', return_sequences=True)(
        decoded)
    return Model(input=latent_input, output=decoded, name='Decoder')

if __name__=='__main__':
    import argparse
    paser = argparse.ArgumentParser()
    paser.add_argument('--which_epoch',default=200,type=int)
    paser.add_argument('--nb_test', default=100, type=int)
    paser.add_argument('--which_activity', default=0, type=int)
    args = paser.parse_args()
    root_path = os.getenv('Seq_AAE_V1')
    path_model =root_path +'Training/Seq_AAE/Expr_Emilya/exp2310/expr001/'

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
    indices = np.where(Y1 == args.which_activity)[0]
    X = X[indices]
    Y1 = Y1[indices]
    Y2 = Y2[indices]
    max_vector = dataset_obj.max_vector[1:]
    min_vector = dataset_obj.min_vector[1:]

    # get the latent codes using its true emotion label
    latent_codes = encoder.predict(x=X, batch_size=100)
    #define decoder model
    SAAE = Sequence_Adversrial_Autoencoder_with_New_Loss(latent_dim=50,latent_activation='tanh',
                     hidden_dim_enc_list=[100,100],hidden_dim_dec_list=[100,100],
                     activation_enc_list=['tanh','tanh'],activation_dec_list=['tanh','tanh'])
    SAAE.nb_label =8
    SAAE.max_len = 1000
    SAAE.dof = 69
    SAAE.postprocess = dataset_obj.postprocess
    SAAE.sampling_interval = dataset_obj.sampling_interval

    SAAE.save_generated_seqs(X, max_vector=max_vector, min_vector=min_vector,
                             suffix='true_seq_activity' + str(args.which_activity))

    # decoder = Decoder(latent_dim=50,max_len=500,
    #                   hidden_dim_dec_list=[100,100],activation_dec_list=['tanh','tanh'],dof=69)
    #
    # #Emotion_name = ['Anger', 'Anxiety', 'Joy', 'Neutral', 'Panic Fear', 'Pride', 'Sadness', 'Shame']
    #
    # #load decoder
    # decoder_name = 'de'+encoder_name[2:]
    # # with open(path_model+decoder_name+'.yaml','r') as f:
    # #     decoder = model_from_yaml(f)
    # decoder.load_weights(path_model+decoder_name+'.h5')
    # latent_codes = latent_codes*1.0+np.random.normal(size=latent_codes.shape,scale=1)*0.
    # generated_long_seqs = decoder.predict(x=latent_codes,verbose=0)
    # SAAE.save_generated_seqs(generated_long_seqs,max_vector=max_vector,min_vector=min_vector,
    #                          suffix='random_generated_activity'+str(args.which_activity))








