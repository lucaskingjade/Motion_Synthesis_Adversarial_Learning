#this is used for compare two version of SAAE about dynamics vanishing when generating long sequences
import os
from keras.layers import Input,LSTM,Dense,BatchNormalization,RepeatVector,SimpleRNN,merge
from keras.models import Model



def encoder_no_speed(max_len,dof,hidden_dim_enc_list,activation_enc_list,
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
def decoder_no_speed(latent_dim,max_len,hidden_dim_dec_list,activation_dec_list,dof):
    latent_input = Input(shape=(latent_dim,), name='latent_input')
    decoded = RepeatVector(max_len)(latent_input)
    for i, (dim, activation) in enumerate(zip(hidden_dim_dec_list, activation_dec_list)):
        decoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(decoded)
    output = SimpleRNN(output_dim=dof, activation='sigmoid', name='decoder_output', return_sequences=True)(decoded)
    return Model(input=latent_input, output=output, name='Decoder')


#define encoder_with_speed
def encoder_with_speed(max_len,dof,hidden_dim_enc_list=[100,100],activation_enc_list=['tanh','tanh'],
            latent_dim=50,speed_dim=1):

    motion_input = Input(shape=(max_len, dof), name='encoder_input')
    speed_input = Input(shape=(max_len, speed_dim), name='speed_input1')
    encoded = merge([motion_input, speed_input], mode='concat')
    for i, (dim, activation) in enumerate(zip(hidden_dim_enc_list, activation_enc_list)):
        encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(encoded)

    encoded = LSTM(output_dim=latent_dim, activation=activation, name='encoded_layer',
                           return_sequences=False)(encoded)
    encoded = Dense(output_dim=latent_dim, activation='linear')(encoded)

    return Model(input=[motion_input, speed_input], output=encoded, name='Encoder')


#define decoder_with_speed
def decoder_with_speed(latent_dim,max_len,dof,speed_dim,hidden_dim_dec_list,activation_dec_list):
    latent_input = Input(shape=(latent_dim,), name='latent_input')
    latent_input_seq = RepeatVector(max_len)(latent_input)
    speed_input = Input(shape=(max_len, speed_dim), name='speed_input2')
    decoded = merge([latent_input_seq, speed_input], mode='concat')

    for i, (dim, activation) in enumerate(zip(hidden_dim_dec_list, activation_dec_list)):
        decoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(decoded)

    decoded = SimpleRNN(output_dim = dof, activation='sigmoid', name='decoder_output', return_sequences=True)(
        decoded)
    return Model(input=[latent_input, speed_input], output=decoded, name='Decoder')



if __name__=='__main__':

    root_path = os.getenv('Seq_AAE_V1')
    expr1_path = root_path+'Training/Delta_Seq_AAE/Expr_Emilya/expr1303/expr022/' #without speed as contextual information
    expr2_path = root_path+'Training/Seq_AAE/Expr_Emilya/expr0103/expr012/' #with speed as contextual information

#load two models
    length_reconstructed = 1000
    encoder_nospeed = encoder_no_speed(max_len=200,dof=70,hidden_dim_enc_list=[100,100],
                                       activation_enc_list=['tanh','tanh'],latent_BN=False,latent_dim=50,
                                       latent_activation='tanh')

    decoder_nospeed = decoder_no_speed(latent_dim=50,max_len=length_reconstructed,hidden_dim_dec_list=[100,100],
                                       activation_dec_list=['tanh','tanh'],dof=70)
    encoder_nospeed.load_weights(expr1_path+'encoder.h5')
    decoder_nospeed.load_weights(expr1_path+'decoder.h5')

    encoder_withspeed = encoder_with_speed(max_len=200,dof=70,hidden_dim_enc_list=[100,100],
                                           activation_enc_list=['tanh','tanh'],
                                           latent_dim=50,speed_dim=1)

    decoder_withspeed = decoder_with_speed(latent_dim=50,max_len=length_reconstructed,dof=70,speed_dim=1,
                                           hidden_dim_dec_list=[100,100],activation_dec_list=['tanh','tanh'])
    encoder_withspeed.load_weights(expr2_path+'encoder.h5')
    decoder_withspeed.load_weights(expr2_path+'decoder.h5')


    #load dataset
    import h5py
    data_path = root_path+'datasets/EmilyaDataset/alldataset_frame200_shift20_without_sampling.h5'
    h5_handle = h5py.File(data_path,'r')
    max_vertical_position = h5_handle['vertical_position']['max_vertical_position']
    min_vertical_position = h5_handle['vertical_position']['min_vertical_position']
    max_vector = h5_handle['max_min_vectors']['max_vector']
    min_vector = h5_handle['max_min_vectors']['min_vector']


    import numpy as np
    #SW :2, WH:7
    #from npz file get 1000 speed sequences with 1000 timesteps
    npz_path = root_path+'datasets/EmilyaDataset/Emilya_Dataset.npz'
    from Seq_AAE_V1.datasets.EmilyaDataset.src.EmilyData_utils import load_data_from_npz,truncate_long_sequence,compute_speed_xz
    from Seq_AAE_V1.datasets.EmilyaDataset.src.EmilyData_utils import normalization_vertical_position_per_subject,normalization_joint_angles
    X, Y1, Y2, Y3 = load_data_from_npz(npz_path)
    X, max_vertical_position_each_actor, min_vertical_position_each_actor \
        = normalization_vertical_position_per_subject(X, Y3, actor_labels=range(12))

    # normalization for joint angles
    X, max_vector, min_vector = normalization_joint_angles(X)

    indices = np.where(Y1 ==2)[0]
    print indices
    long_sequences=[]
    for idx in indices:
        long_sequences.append(X[idx])
    # truncate long sequences into 1000 timesteps
    dataset= truncate_long_sequence((long_sequences,Y1[indices],Y2[indices],Y3[indices]),window_size=1000,shift_step=20)

    long_sequences = dataset[0]
    #compute speed_xz
    print "shape of long speed is {}".format(long_sequences.shape)
    long_sequences,long_speed = compute_speed_xz(long_sequences)

    print "shape of long speed is {}".format(long_speed.shape)
    del dataset,X,Y1,Y2,Y3
    #reconstruct using two models
    latent_codes_no_speed = encoder_nospeed.predict(x=long_sequences[:,:200,:])
    reconstructed_seqs_no_speed = decoder_nospeed.predict(x=latent_codes_no_speed)
    latent_codes_withspeed = encoder_withspeed.predict(x=[long_sequences[:,:200,:], long_speed[:,:200]])
    # print "shape of speed_seqs is {}".format(speed_seqs.shape)

    #new_speed_seq = np.asarray(new_speed_seq).reshape(nb_seq, length_reconstructed, 1)
    #new_speed_seq = new_speed_seq[:,:,:]+np.random.uniform(size=(nb_seq,length_reconstructed,1))*0.0
    #print "shape of extended speed is {}".format(new_speed_seq.shape)

    reconstructed_seqs_withspeed = decoder_withspeed.predict(x=[latent_codes_withspeed,long_speed])
    print "begin compute norm of differences between "
    #compute norm of the difference of every two successive frames and average over samples.]
    norm_velocity_no_speed = np.mean(np.sqrt(np.sum(np.square((reconstructed_seqs_no_speed[:,18:length_reconstructed, 4:]
                                      - reconstructed_seqs_no_speed[:,17:(length_reconstructed-1), 4:])), axis=-1)),axis=0)

    norm_velocity_withspeed = np.mean(np.sqrt(np.sum(np.square((reconstructed_seqs_withspeed[:,18:length_reconstructed, 4:]
                                      - reconstructed_seqs_withspeed[:,17:(length_reconstructed-1), 4:])), axis=-1)),axis=0)

    #plot the averaged dynamics curves.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(norm_velocity_no_speed,label='SAAE without speed',color='b')
    plt.plot(norm_velocity_withspeed, label='SAAE with speed',color='r')
    plt.legend( ['SAAE without speeds', 'SAAE with speeds'],fontsize=12)
    #plt.axis([0, length_reconstructed, 0, 0.025])
    plt.savefig('./comparing_dynamics.png')
    #np.save('./results/norm_velocity.npz', norm_velocity)