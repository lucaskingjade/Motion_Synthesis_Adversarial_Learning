from Seq_AAE_V1.datasets.EmilyaDataset.src.get_dataset import get_dataset
import numpy as np
from keras.layers import Input,RepeatVector,merge,LSTM,SimpleRNN,Dense
from keras.models import Model
from keras.models import model_from_yaml
from Seq_AAE_V1.synthesis_scripts.synthesis_utils import postprocess_seq,smooth_sequence,save_seq_2_bvh,get_index_from_labels,get_line_space_array
from Seq_AAE_V1.datasets.EmilyaDataset.src.get_dataset import get_label_by_name
from Seq_AAE_V1.datasets.EmilyaDataset.src.EmilyData_utils import compute_speed_xz
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

    max_vector = h5_handle['max_min_vectors']['max_vector'][:]
    min_vector = h5_handle['max_min_vectors']['min_vector'][:]
    max_vertical_position = h5_handle['vertical_position']['max_vertical_position'][:]
    min_vertical_position = h5_handle['vertical_position']['min_vertical_position'][:]
    print 'Successfully get sequences'
    import os
    root_path = os.getenv('Seq_AAE_V1')
    datafilename = root_path+ 'continuous_data/Nt1CS_Brian.bvh'
    #read bvh data from Nt1CS_Brian.bvh
    from Seq_AAE_V1.bvh_tools.bvhplay.skeleton import *
    bvh = readbvh(datafilename)
    bvh.read()
    keyframes = np.asarray(bvh.keyframes,dtype=np.float32)
    print "shape of keyframes is {}".format(keyframes.shape)
    actor_label = 0
    sequence_1 = keyframes[2383:2583]
    sequence_medium = keyframes[2583:2783]
    sequence_2 = keyframes[2783:2983]
    #save ground truth
    save_seq_2_bvh(sequence_medium,sequence_name='./results/ground_truth_medium_Nt1CS_Brian.bvh')

    #processing sequences
    max_vertical_vector = max_vertical_position[actor_label]
    min_vertical_vector = min_vertical_position[actor_label]
    sequence_1 = (sequence_1-min_vertical_vector)/(max_vertical_vector - min_vertical_vector)
    sequence_medium = (sequence_medium - min_vertical_vector) / (max_vertical_vector - min_vertical_vector)
    sequence_2 = (sequence_2 - min_vertical_vector) / (max_vertical_vector - min_vertical_vector)

    sequence_1 = (sequence_1-min_vector)/(max_vector-min_vector)
    sequence_medium = (sequence_medium - min_vector) / (max_vector - min_vector)
    sequence_2 = (sequence_2 - min_vector) / (max_vector - min_vector)

    #
    sequence_1,speed_1 = compute_speed_xz([sequence_1])
    sequence_medium,speed_medium= compute_speed_xz([sequence_medium])
    sequence_2,speed_2 = compute_speed_xz([sequence_2])

    print "shape of sequence_1 is {}".format(sequence_1.shape)
    print "shape of sequence_medium is {}".format(sequence_medium.shape)
    print "shape of sequence_2 is {}".format(sequence_2.shape)


    #load encoder and decoder
    print 'load encoder ...'
    dof = sequence_1.shape[-1]
    max_len = sequence_1.shape[-2]
    encoder = encoder(max_len=max_len, dof=dof,
                      hidden_dim_enc_list=[100, 100], activation_enc_list=['tanh', 'tanh'],
                      latent_dim=50, speed_dim=1)
    encoder.load_weights('./encoder.h5')

    latent_code_1 = encoder.predict(x=[sequence_1,speed_1])
    latent_code_medium = encoder.predict(x=[sequence_medium, speed_medium])
    latent_code_2 = encoder.predict(x=[sequence_2,speed_2])

    #construct a latent_seq
    nb_frames_interpolated= sequence_medium.shape[-2]
    print "nb_frames_interpolated = %d"%nb_frames_interpolated
    interploated_latent_seq = get_line_space_array(latent_code_1[0],latent_code_2[0],nb=nb_frames_interpolated)
    print interploated_latent_seq.shape

    latent_seq_1 = []
    for cc in range(sequence_1.shape[-2]):
        latent_seq_1.extend(latent_code_1)
    latent_seq_1 = np.asarray(latent_seq_1,dtype=np.float32)

    latent_seq_2 = []
    for cc in range(sequence_2.shape[-2]):
        latent_seq_2.extend(latent_code_2)
    latent_seq_2 = np.asarray(latent_seq_2, dtype=np.float32)

    latent_sequence = np.concatenate((latent_seq_1,interploated_latent_seq,latent_seq_2),axis=0)
    latent_sequence = latent_sequence.reshape((1,latent_sequence.shape[0],latent_sequence.shape[1]))
    print 'shape of latent_sequence is {}'.format(latent_sequence.shape)
    print "shape of speed_1 {}".format(speed_1.shape)
    print "shape of speed_medium {}".format(speed_medium.shape)
    print "shape of speed_2 {}".format(speed_2.shape)


    speed_sequence = np.concatenate((speed_1,speed_medium,speed_2),axis=1)
    print "shape of speed_sequence is {}".format(speed_sequence.shape)
    print 'load decoder ...'
    maxlength = sequence_1.shape[-2]+sequence_medium.shape[-2]+sequence_2.shape[-2]
    decoder = decoder(latent_dim=50, max_len=maxlength, dof=dof, speed_dim=1,
                      hidden_dim_dec_list=[100, 100], activation_dec_list=['tanh', 'tanh'])

    decoder.load_weights('./decoder.h5')
    output_sequence = decoder.predict(x=[latent_sequence,speed_sequence])


    #compute the MSE with groundtruth
    euclidean_distance = np.sqrt(np.sum(np.square((sequence_medium[0]- output_sequence[0][200:400,:])), axis=-1))
    plt.figure()
    plt.plot(euclidean_distance)
    plt.savefig('./results/interpolated_disctance.png')

    #compute norm of velocity of two successive frames,to see if the translation is linear
    output_sequence[0][200:400]
    norm_velocity = np.sum(np.square((output_sequence[0][201:401,:]-output_sequence[0][200:400,:])),axis=-1)
    plt.figure()
    plt.plot(norm_velocity)
    plt.axis([0,200,0,0.025])
    plt.savefig('./results/norm_velocity.png')

    np.save('./results/norm_velocity.npz',norm_velocity)

    #smooth
    output_sequence = smooth_sequence(output_sequence,
                                      window_length=41,
                                      polyorder=3)
    #postprocess seqeunces
    sequence = postprocess_seq(output_sequence, speed_sequence, max_vector=max_vector, min_vector=min_vector,
                               max_vertical_position=max_vertical_position, min_vertical_position=min_vertical_position,
                               actor=1)



    # smooth sequence
    sequence = smooth_sequence(sequence, window_length=41, polyorder=3)

    # save to file
    print "saving long sequence..."

    file_name = './results/SW_SD_interpolated.bvh'


    save_seq_2_bvh(sequence[0], file_name)

    #save frames every 20 frames
    interpolated_frames = sequence[0][200:400:20,:]
    print interpolated_frames.shape
    file_name = './results/SW_SD_interplated_snapshoot.bvh'
    save_seq_2_bvh(interpolated_frames,file_name,step=1000)

