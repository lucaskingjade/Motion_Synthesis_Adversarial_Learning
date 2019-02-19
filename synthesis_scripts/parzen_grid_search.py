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

    N_test = 20000
    N_g = 30000
    tt = 10 #subsampline step
    expr_name = 'saae'
    X = h5_handle['test_set']['X'][:N_test]
    Y1 = h5_handle['test_set']['Y1']
    Y2 = h5_handle['test_set']['Y2']
    Y3 = h5_handle['test_set']['Y3']
    train_X = h5_handle['training_set']['X'][:10000]

    #decoder
    decoder = decoder(latent_dim=50,max_len=200,dof= 70, hidden_dim_dec_list=[100,100],activation_dec_list=['tanh','tanh'])
    decoder.summary()

    #load weights from expr021 and expr022
    #expr022_path = root_path+ 'Training/Delta_Seq_AAE/Expr_Emilya/expr1303/expr022/'
    #expr021_path = root_path+ 'Training/Delta_Seq_AAE/Expr_Emilya/expr1303/expr021/'
    expr_path = './'

    decoder.load_weights(expr_path+'decoder.h5')


    #estimate the mu and covariance of the latent space of seq2seq
    encoder = encoder(max_len=200,dof=70,hidden_dim_enc_list=[100,100],activation_enc_list=['tanh','tanh'],
                              latent_BN=False,latent_dim=50,latent_activation='tanh')
    encoder.summary()
    encoder.load_weights(expr_path+'encoder.h5')

    latent_codes = encoder.predict(x=train_X,batch_size=1000)
    if expr_name == 'saae':
        mean = np.zeros(50)
        covariance = np.eye(N=50) * 1.0
    else:
        mean = np.mean(latent_codes,axis=0)
        covariance = np.cov(latent_codes,rowvar=False)
    print "shape of mean of {0} is {1}".format(expr_name,mean.shape)
    print "shape of covariance of {0} is {1}".format(expr_name, covariance.shape)

    #sample z from prior distribution p(z)~N(0,1.0),and p(z)~N(mean,cov)

    z = generate_positive_samples(N_g, mean, covariance, 'Gaussian',
                                                                seed=np.random.randint(0, 2000))


    #generate sequences from sampled z
    sequences = decoder.predict(x=z)
    print "shape of sequences of {0} is {1}".format(expr_name, sequences.shape)

    #truncate sequences of 200 frames into ones of 10 frames and flatten them into a vector

    N = sequences.shape[0]
    samples_generated = np.zeros(shape=(N,70*200/tt),dtype=np.float32)
    samples_test_set = np.zeros(shape=(N_test, 70 * 200/tt), dtype=np.float32)
    count = 0
    for seq in sequences:
        for i in range(1):
            samples_generated[count,:] = seq[i::tt,:].flatten()
            count= count+1

    count =0
    for xx in X:
        for ii in range(1):
            samples_test_set[count, :] = xx[i::tt, :].flatten()
            count = count +1

    print "shape of samples_generated is {}".format(samples_generated.shape)

    print "shape of samples_test_set is {}".format(samples_test_set.shape)
    #fit Gaussian Parzen Window using sequences
    from sklearn.neighbors import KernelDensity
    #bandwidth = 0.25
    #ParzenWindow_saae = KernelDensity(bandwidth=bandwidth,algorithm='auto',kernel='gaussian',metric='euclidean')
    #ParzenWindow_seq2seq = KernelDensity(bandwidth=bandwidth, algorithm='auto', kernel='gaussian', metric='euclidean')
    params = {'bandwidth': np.logspace(-1, 0., 10)}

    grid = GridSearchCV(KernelDensity(), params,cv=3,verbose=1)
    grid_result = grid.fit(samples_generated)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.grid_scores_:
        print("scores.mean:%f (score.std:%f) with: %r" % (scores.mean(), scores.std(), params))

    #compute log probability under best estimator
    kde = grid.best_estimator_
    probability_test_set =kde.score_samples(samples_test_set)
    print "shape of pb is {}".format(probability_test_set.shape)
    mean_pb = np.mean(probability_test_set)
    std_pb = np.std(probability_test_set)
    #density_saae = ParzenWindow_saae.score_samples(samples_test_set)
    print "{0}: mean of probability:{1},std:{2}".format(expr_name, mean_pb,std_pb)









