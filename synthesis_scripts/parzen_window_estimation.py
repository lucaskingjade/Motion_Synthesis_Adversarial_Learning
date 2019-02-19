#This is used for estimate the PDF of x by fitting a gaussian parzen window with the examples generated
#from the generator and reporting the probability under test set.

import numpy as np
import os
import h5py
from Seq_AAE_V1.models.Velocity_Seq_AAE.velocity_seq_aae import Delta_Sequence_Adversrial_Autoencoder
from Seq_AAE_V1.datasets.dataset import generate_positive_samples
from keras.layers import Input,LSTM,Dense,BatchNormalization,RepeatVector,SimpleRNN
from keras.models import Model
from sklearn.model_selection import GridSearchCV

#define encoder
def encoder(max_len,dof,hidden_dim_enc_list,activation_enc_list,
            latent_BN,latent_dim,latent_activation):
    input = Input(shape=(max_len, dof), name='encoder_input')
    # speed_input = Input(shape = (max_len, speed_dim), name='speed_input1')
    # encoded = merge(motion_input, mode='concat')
    for i, (dim, activation) in enumerate(zip(hidden_dim_enc_list, activation_enc_list)):
        if i == 0:
            encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(input)
        else:
            encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(encoded)
    if latent_BN == True:
        encoded = LSTM(output_dim=latent_dim, activation=latent_activation, name='encoded_layer',
                       return_sequences=False)(encoded)
        encoded = Dense(output_dim=latent_dim, activation='linear')(encoded)
        encoded = BatchNormalization(name='latent_BN')(encoded)
    else:
        encoded = LSTM(output_dim=latent_dim, activation=latent_activation, name='encoded_layer',
                       return_sequences=False)(encoded)
        encoded = Dense(output_dim=latent_dim, activation='linear')(encoded)

    return Model(input=input, output=encoded, name='Encoder')


#define decoder
def decoder(latent_dim,max_len,hidden_dim_dec_list,activation_dec_list,dof):
    latent_input = Input(shape=(latent_dim,), name='latent_input')
    decoded = RepeatVector(max_len)(latent_input)
    for i, (dim, activation) in enumerate(zip(hidden_dim_dec_list, activation_dec_list)):
        decoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(decoded)
    output = SimpleRNN(output_dim=dof, activation='sigmoid', name='decoder_output', return_sequences=True)(decoded)
    return Model(input=latent_input, output=output, name='Decoder')





if __name__ == "__main__":

    root_path =os.getenv('Seq_AAE_V1')
    data_path = root_path+'datasets/EmilyaDataset/alldataset_frame200_shift20_without_sampling_with_velocity.h5'
    h5_handle = h5py.File(data_path,mode='r')

    N_test = 1000
    N_g = 1000
    tt = 20 #subsampline step

    X = h5_handle['test_set']['X'][:N_test]
    Y1 = h5_handle['test_set']['Y1']
    Y2 = h5_handle['test_set']['Y2']
    Y3 = h5_handle['test_set']['Y3']
    train_X = h5_handle['training_set']['X'][:5000]

    #decoder
    decoder_saae = decoder(latent_dim=50,max_len=200,dof= 70, hidden_dim_dec_list=[100,100],activation_dec_list=['tanh','tanh'])
    decoder_seq2seq = decoder(latent_dim=50,max_len=200,dof= 70, hidden_dim_dec_list=[100,100],activation_dec_list=['tanh','tanh'])
    decoder_saae.summary()
    decoder_seq2seq.summary()

    #load weights from expr021 and expr022
    expr022_path = root_path+ 'Training/Delta_Seq_AAE/Expr_Emilya/expr1303/expr022/'
    expr021_path = root_path+ 'Training/Delta_Seq_AAE/Expr_Emilya/expr1303/expr021/'

    decoder_saae.load_weights(expr022_path+'decoder.h5')
    decoder_seq2seq.load_weights(expr021_path+'decoder.h5')

    #estimate the mu and covariance of the latent space of seq2seq
    encoder_seq2seq = encoder(max_len=200,dof=70,hidden_dim_enc_list=[100,100],activation_enc_list=['tanh','tanh'],
                              latent_BN=False,latent_dim=50,latent_activation='tanh')
    encoder_seq2seq.summary()
    encoder_seq2seq.load_weights(expr021_path+'encoder.h5')

    latent_codes_seq2seq = encoder_seq2seq.predict(x=train_X,batch_size=1000)
    mean_seq2seq = np.mean(latent_codes_seq2seq,axis=0)
    cov_seq2seq = np.cov(latent_codes_seq2seq,rowvar=False)
    print "shape of mean of seq2seq is {}".format(mean_seq2seq.shape)
    print "shape of covariance of seq2seq is {}".format(cov_seq2seq.shape)

    #sample z from prior distribution p(z)~N(0,1.0),and p(z)~N(mean,cov)
    mean_saae = np.zeros(50)
    cov_saae = np.eye(N=50) * 1.0

    z_saae= generate_positive_samples(N_g, mean_saae, cov_saae, 'Gaussian',
                                                                seed=np.random.randint(0, 2000))

    z_seq2seq = generate_positive_samples(N_g, mean_seq2seq, cov_seq2seq, 'Gaussian',
                                       seed=np.random.randint(0, 2000))


    #generate sequences from sampled z
    seq_saae = decoder_saae.predict(x=z_saae)
    seq_seq2seq = decoder_seq2seq.predict(x=z_seq2seq)
    print "shape of seq_saae is {}".format(seq_saae.shape)
    print "shape of seq_seq2seq is {}".format(seq_seq2seq.shape)
    #truncate sequences of 200 frames into ones of 10 frames and flatten them into a vector

    N = seq_saae.shape[0]
    samples_saae = np.zeros(shape=(N,70*200/tt),dtype=np.float32)
    samples_seq2seq = np.zeros(shape=(N, 70 * 200/tt), dtype=np.float32)
    samples_test_set = np.zeros(shape=(N_test, 70 * 200/tt), dtype=np.float32)
    count = 0
    for seq1,seq2 in zip(seq_saae,seq_seq2seq):
        for i in range(1):
            samples_saae[count,:] = seq1[i::tt,:].flatten()
            samples_seq2seq[count, :] = seq2[i::tt, :].flatten()
            count= count+1

    count =0
    for xx in X:
        for ii in range(1):
            samples_test_set[count, :] = xx[i::tt, :].flatten()
            count = count +1

    print "shape of samples_saae is {}".format(samples_saae.shape)
    print "shape of samples_seq2seq is {}".format(samples_seq2seq.shape)
    print "shape of samples_test_set is {}".format(samples_test_set.shape)
    #fit Gaussian Parzen Window using sequences
    from sklearn.neighbors import KernelDensity
    bandwidth = 0.25
    ParzenWindow_saae = KernelDensity(bandwidth=bandwidth,algorithm='auto',kernel='gaussian',metric='euclidean')
    ParzenWindow_seq2seq = KernelDensity(bandwidth=bandwidth, algorithm='auto', kernel='gaussian', metric='euclidean')
    #params = {'bandwidth': np.logspace(-1, 1, 20)}
    #grid1 = GridSearchCV(KernelDensity(), params)


    ParzenWindow_saae.fit(samples_saae)
    probability_saae_test_set = ParzenWindow_saae.score_samples(samples_test_set)
    print "shape of pb is {}".format(probability_saae_test_set.shape)
    mean_pb_saae = np.mean(probability_saae_test_set)
    std_saae = np.std(probability_saae_test_set)
    #density_saae = ParzenWindow_saae.score_samples(samples_test_set)

    ParzenWindow_seq2seq.fit(samples_seq2seq)
    probability_seq2seq_test_set = ParzenWindow_seq2seq.score_samples(samples_test_set)

    mean_pb_seq2seq = np.mean(probability_seq2seq_test_set)
    std_seq2seq = np.std(probability_seq2seq_test_set)
    #density_seq2seq = ParzenWindow_seq2seq.score_samples(samples_test_set)

    # print "probability saae is {}".format(probability_saae_test_set/N_test)
    # print "probability seq2seq is {}".format(probability_seq2seq_test_set/N_test)
    print "SAAE: mean of probability:{0},std:{1}".format(mean_pb_saae,std_saae)
    print "seq2seq: mean of probability:{0},std:{1}".format(mean_pb_seq2seq,std_seq2seq)

    #k-fold
    k = 10
    probability_saae_test_set = np.random.permutation(probability_saae_test_set)
    probability_seq2seq_test_set = np.random.permutation(probability_seq2seq_test_set)
    N_pb = probability_saae_test_set.shape[0]
    mean_saae_list = []
    mean_seq2seq_list = []
    for i in range(k):
        start = i*N_pb/10
        end =start+(N_pb/10)
        mean_saae_list.append(np.mean(probability_saae_test_set[start:end]))
        mean_seq2seq_list.append(np.mean(probability_seq2seq_test_set[start:end]))
        #std_saae_list.append(np.std(probability_saae_test_set[start:end]))
    mean_saae = np.mean(mean_saae_list)
    std_saae = np.std(mean_saae_list)
    mean_seq2seq = np.mean(mean_seq2seq_list)
    std_seq2seq = np.std(mean_seq2seq_list)
    print "SAAE: k-fold, mean of pb:{0},std:{1}".format(mean_saae,std_saae)
    print "Seq2Seq: k-fold, mean of pb:{0},std:{1}".format(mean_seq2seq, std_seq2seq)










