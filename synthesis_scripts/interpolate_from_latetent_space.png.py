#This is used for comparing the performance of the interpolation of SAAE and Seq2Seq


from Seq_AAE_V1.datasets.EmilyaDataset.src.get_dataset import get_dataset
import numpy as np
from keras.layers import Input,RepeatVector,merge,LSTM,SimpleRNN,Dense
from keras.models import Model
from Seq_AAE_V1.synthesis_scripts.synthesis_utils import get_line_space_array

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def encoder(max_len,dof,hidden_dim_enc_list=[100,100],activation_enc_list=['tanh','tanh'],
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

def decoder(latent_dim,max_len,dof,speed_dim,hidden_dim_dec_list,activation_dec_list):
    latent_input = Input(shape=(max_len,latent_dim), name='latent_input')
    #latent_input_seq = RepeatVector(max_len)(latent_input)
    speed_input = Input(shape=(max_len, speed_dim), name='speed_input2')
    decoded = merge([latent_input, speed_input], mode='concat')

    for i, (dim, activation) in enumerate(zip(hidden_dim_dec_list, activation_dec_list)):
        decoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(decoded)

    decoded = SimpleRNN(output_dim = dof, activation='sigmoid', name='decoder_output', return_sequences=True)(
        decoded)
    return Model(input=[latent_input, speed_input], output=decoded, name='Decoder')

if __name__ == '__main__':

    from keras.models import load_model

    # get dataset
    Actor_name = ['Brian', 'Elie', 'Florian', 'Hu', 'Janina', 'Jessica', 'Maria', 'Muriel', 'Robert', 'Sally',
                  'Samih', 'Tatiana']
    h5_handle, filesource = get_dataset(actor_list=Actor_name, window_width=200, shift_step=20, sampling_interval=None)

    # #load test set
    # X = h5_handle['test_set']['X'][:]
    # Y1 = h5_handle['test_set']['Y1'][:]
    # Y2 = h5_handle['test_set']['Y2']
    # Y3 = h5_handle['test_set']['Y3']
    # speed_xz = h5_handle['test_set']['speed_xz'][:]
    max_vector = h5_handle['max_min_vectors']['max_vector'][:]
    min_vector = h5_handle['max_min_vectors']['min_vector'][:]
    max_vertical_position = h5_handle['vertical_position']['max_vertical_position'][:]
    min_vertical_position = h5_handle['vertical_position']['min_vertical_position'][:]
    print 'Successfully get sequences'
    import os
    root_path = os.getenv('Seq_AAE_V1')

    # from npz file get 1000 speed sequences with 1000 timesteps
    npz_path = root_path + 'datasets/EmilyaDataset/Emilya_Dataset.npz'
    from Seq_AAE_V1.datasets.EmilyaDataset.src.EmilyData_utils import load_data_from_npz, truncate_long_sequence, \
        compute_speed_xz
    from Seq_AAE_V1.datasets.EmilyaDataset.src.EmilyData_utils import normalization_vertical_position_per_subject, \
        normalization_joint_angles

    X, Y1, Y2, Y3 = load_data_from_npz(npz_path)
    X, max_vertical_position_each_actor, min_vertical_position_each_actor \
        = normalization_vertical_position_per_subject(X, Y3, actor_labels=range(12))

    # normalization for joint angles
    X, max_vector, min_vector = normalization_joint_angles(X)

    activity1 = 2
    activity2 = 6
    indices1 = np.where(Y1 == activity1)[0]
    indices2 = np.where(Y1 == activity2)[0]

    long_sequences1 = []
    long_sequences2 = []
    for idx1,idx2 in zip(indices1,indices2):
        long_sequences1.append(X[idx1])
        long_sequences2.append(X[idx2])
    # truncate long sequences into 1000 timesteps
    dataset = truncate_long_sequence((long_sequences1, Y1[indices1], Y2[indices1], Y3[indices1]), window_size=400,
                                     shift_step=20)

    long_sequences1 = dataset[0][:100]
    # compute speed_xz
    print "shape of long speed is {}".format(long_sequences1.shape)
    sequence_1, speed_1 = compute_speed_xz(long_sequences1)

    print "shape of long speed is {}".format(speed_1.shape)
    del dataset,long_sequences1

    dataset = truncate_long_sequence((long_sequences2, Y1[indices2], Y2[indices2], Y3[indices2]), window_size=300,
                                     shift_step=20)

    long_sequences2 = dataset[0][:100]
    # compute speed_xz
    print "shape of long speed is {}".format(long_sequences2.shape)
    sequence_2, speed_2 = compute_speed_xz(long_sequences2)

    print "shape of long speed is {}".format(speed_2.shape)
    del dataset,X,Y1,Y2,Y3,long_sequences2


    # activity1 = 2
    # activity2 = 6
    # indices1 = np.where(Y1==activity1)[0]
    # indices2 = np.where(Y1==activity2)[0]

    # sequence_1 = X[indices1][:1000]
    # speed_1 = speed_xz[indices1][:1000]
    # sequence_2 = X[indices2][:1000]
    # speed_2 = speed_xz[indices2][:1000]

    print "shape of sequence_1 is {}".format(sequence_1.shape)
    print "shape of sequence_2 is {}".format(sequence_2.shape)
    print "shape of speed_1 is {}".format(speed_1.shape)
    print "shape of speed_2 is {}".format(speed_2.shape)

    #load encoder and decoder
    print 'load encoder ...'
    dof = sequence_1.shape[-1]
    # max_len = sequence_1.shape[-2]
    max_len = 200
    encoder = encoder(max_len=max_len, dof=dof,
                      hidden_dim_enc_list=[100, 100], activation_enc_list=['tanh', 'tanh'],
                      latent_dim=50, speed_dim=1)
    encoder.load_weights('./encoder.h5')

    latent_code_1 = encoder.predict(x=[sequence_1[:,:200,:],speed_1[:,:200,:]])
    latent_code_2 = encoder.predict(x=[sequence_2[:,:200,:],speed_2[:,:200,:]])

    #construct a latent_seq
    nb_interpolated = max_len
    print "length of interpolated sequence is %d"%nb_interpolated
    interpolated_latent_seq = []
    for latent1, latent2 in zip(latent_code_1,latent_code_2):
        cur_interpolate_latent = get_line_space_array(latent1,latent2,nb=nb_interpolated)
        interpolated_latent_seq.append(cur_interpolate_latent)

    interpolated_latent_seq = np.asarray(interpolated_latent_seq)
    #interpolated_latent_seq = interpolated_latent_seq.reshape((len(interpolated_latent_seq),max_len,dof))
    print 'shape of interpolated_latent_seq is {}'.format(interpolated_latent_seq.shape)
    latent_seq_1 = []
    latent_seq_2 = []
    for  latent1,latent2 in zip(latent_code_1,latent_code_2):
        cur_latent_seq_1 = []
        for cc in range(200):
            cur_latent_seq_1.append(latent1)
        cur_latent_seq_1 = np.asarray(cur_latent_seq_1,dtype=np.float32)
        print "shape of cur_latent_seq_1 is {}".format(cur_latent_seq_1.shape)
        latent_seq_1.append(cur_latent_seq_1)

        cur_latent_seq_2 = []
        for cc in range(200):
            cur_latent_seq_2.append(latent2)
        cur_latent_seq_2 = np.asarray(cur_latent_seq_2, dtype=np.float32)
        print "shape of cur_latent_seq_2 is {}".format(cur_latent_seq_2.shape)
        latent_seq_2.append(cur_latent_seq_2)

    latent_seq_1 = np.asarray(latent_seq_1)
    latent_seq_2 = np.asarray(latent_seq_2)
    print "shape of latent_seq_1 is {}".format(latent_seq_1.shape)
    print "shape of latent_seq_2 is {}".format(latent_seq_2.shape)
    latent_sequence = np.concatenate((latent_seq_1,interpolated_latent_seq,latent_seq_2),axis=-2)

    print 'shape of latent_sequence is {}'.format(latent_sequence.shape)
    print "shape of speed_1 {}".format(speed_1.shape)
    print "shape of speed_2 {}".format(speed_2.shape)

    speed_sequence = np.concatenate((speed_1,speed_2[:,:200,:]),axis=1)
    print "shape of speed_sequence is {}".format(speed_sequence.shape)
    print 'load decoder ...'
    # maxlength = sequence_1.shape[-2]+sequence_1.shape[-2]+sequence_2.shape[-2]
    maxlength=600
    decoder = decoder(latent_dim=50, max_len=maxlength, dof=dof, speed_dim=1,
                      hidden_dim_dec_list=[100, 100], activation_dec_list=['tanh', 'tanh'])

    decoder.load_weights('./decoder.h5')
    output_sequence = decoder.predict(x=[latent_sequence,speed_sequence])

    #compute norm of velocity of two successive frames,to see if the translation is linear
    output_sequence[0][200:400]
    norm_velocity = np.mean(np.sqrt(np.sum(np.square((output_sequence[:,201:401,4:]-output_sequence[:,200:400,4:])),axis=-1)),axis=0)
    plt.figure()
    plt.plot(norm_velocity)
    #plt.axis([0,200,0,0.025])
    plt.savefig('./results/norm_velocity.png')

    np.save('./results/norm_velocity.npz',norm_velocity)



