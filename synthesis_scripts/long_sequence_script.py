# from Seq_AAE.generate_long_sequence import *
# from Seq_AAE.sequence_generation_utils import *
import os
# from Seq_AAE.dataset import get_mocap
# from Seq_AAE.mocap_utils import save_seq_2_bvh
from Seq_AAE_V1.datasets.EmilyaDataset.src.get_dataset import get_dataset
import numpy as np
from keras.layers import Input,RepeatVector,merge,LSTM,SimpleRNN,Dense
from keras.models import Model
from keras.models import model_from_yaml
from synthesis_utils import postprocess_seq,smooth_sequence,save_seq_2_bvh,get_index_from_labels
from Seq_AAE_V1.datasets.EmilyaDataset.src.get_dataset import get_label_by_name
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
    latent_input = Input(shape=(latent_dim,), name='latent_input')
    latent_input_seq = RepeatVector(max_len)(latent_input)
    speed_input = Input(shape=(max_len, speed_dim), name='speed_input2')
    decoded = merge([latent_input_seq, speed_input], mode='concat')

    for i, (dim, activation) in enumerate(zip(hidden_dim_dec_list, activation_dec_list)):
        decoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(decoded)

    decoded = SimpleRNN(output_dim = dof, activation='sigmoid', name='decoder_output', return_sequences=True)(
        decoded)
    return Model(input=[latent_input, speed_input], output=decoded, name='Decoder')




if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    activity_list = ['Being Seated', 'Lift', 'Simple Walk', 'Throw', 'Knocking on the Door', 'Move Books',
                     'Sitting Down', 'Walk with smth in the Hands']
    emotion_list = ['Anger', 'Anxiety', 'Joy', 'Neutral', 'Panic Fear', 'Pride', 'Sadness', 'Shame']
    # training_set, valid_set, test_set, max_min_vector, globalposition, max_len = get_mocap(activity_list, \
    #                                                                                        emotion_list, framelen=20,
    #                                                                                        load_from_pkl=True)

    #get dataset
    Actor_name = ['Brian', 'Elie', 'Florian', 'Hu', 'Janina', 'Jessica', 'Maria', 'Muriel', 'Robert', 'Sally',
                  'Samih', 'Tatiana']
    h5_handle, filesource = get_dataset(actor_list=Actor_name,window_width=200,shift_step=20,sampling_interval=None)

    parser = argparse.ArgumentParser()
    parser.add_argument('--emotion', help='specify the emotion name that you want to animate', type=str)
    parser.add_argument('--activity', help='specify the activity name that you want to animate', type=str)
    parser.add_argument('--index', help='specify which seqeunce you want to visualize', type=int)
    parser.add_argument('--dataset', help='choose a dataset: training,valid,test', type=str)
    parser.add_argument('--maxlength', help='specify the length of the generated sequence', type=int)
    parser.add_argument('--smooth',help='smooth the generated sequence',action='store_true')
    args = parser.parse_args()
    #get the true sequence
    # get the seqeunces of the specified dataset,activity and emotion.
    if args.dataset == 'training':
        X = h5_handle['training_set']['X']
        Y1 = h5_handle['training_set']['Y1'][:]
        Y2 = h5_handle['training_set']['Y2'][:]
        Y3 = h5_handle['training_set']['Y3'][:]
        speed_xz = h5_handle['training_set']['speed_xz']
    elif args.dataset == 'valid':
        X = h5_handle['valid_set']['X']
        Y1 = h5_handle['valid_set']['Y1'][:]
        Y2 = h5_handle['valid_set']['Y2'][:]
        Y3 = h5_handle['valid_set']['Y3'][:]
        speed_xz = h5_handle['valid_set']['speed_xz']
    elif args.dataset == 'test':
        X = h5_handle['test_set']['X']
        Y1 = h5_handle['test_set']['Y1'][:]
        Y2 = h5_handle['test_set']['Y2'][:]
        Y3 = h5_handle['test_set']['Y3'][:]
        speed_xz = h5_handle['test_set']['speed_xz']
    else:
        raise ValueError('the dataset doesnt exit')
    # get the sequence specified by the index
    max_vector = h5_handle['max_min_vectors']['max_vector'][:]
    min_vector = h5_handle['max_min_vectors']['min_vector'][:]
    max_vertical_position = h5_handle['vertical_position']['max_vertical_position'][:]
    min_vertical_position = h5_handle['vertical_position']['min_vertical_position'][:]

    #get the corresponding set of sequences
    activity_label = get_label_by_name(args.activity, whichlabel=1)
    emotion_label = get_label_by_name(args.emotion, whichlabel=2)
    print activity_label
    print emotion_label
    indices = get_index_from_labels(activity_label, emotion_label,None, Y1, Y2)


    print 'Successfully get sequences'
    if args.index !=None:
        index = args.index
    else:
        index = np.random.randint(low=0, high=len(indices) - 1)
        print "the index of sequence is %d" % index
    print indices
    input_seqeunce = X[indices[index]]
    speed_seq = speed_xz[indices[index]:indices[index]+1]
    print 'dim of shape of speed seq is {}'.format(speed_seq.shape)
    # load encoder and decoder
    print 'load encoder ...'
    # with open('./encoder.yaml','r') as f:
    #     encoder = model_from_yaml(f)
    dof = input_seqeunce.shape[-1]
    max_len = input_seqeunce.shape[-2]
    encoder = encoder(max_len=max_len,dof=dof,
                      hidden_dim_enc_list=[100,100],activation_enc_list=['tanh','tanh'],
                      latent_dim=50,speed_dim=1)
    encoder.load_weights('./encoder.h5')

    if len(input_seqeunce.shape) != 3:
        input_seqeunce = input_seqeunce.reshape((1, input_seqeunce.shape[0], input_seqeunce.shape[1]))


    true_latent_vector = encoder.predict(x=[input_seqeunce,speed_seq])
    #create a sequence of latent vector
    if args.maxlength ==None:
        max_len = 100
    else:
        max_len = args.maxlength
    #seq_latent_vector = np.asarray([true_latent_vector for i in range(max_len)]).reshape(1,max_len,50)
    print 'speed seq shape is {}'.format(speed_seq.shape)
    offset = max_len-speed_seq.shape[1]
    speed_seq = speed_seq.reshape(speed_seq.shape[1])
    new_speed_seq =[]
    new_speed_seq.extend(speed_seq)
    num_copy = offset/speed_seq.shape[-1]
    for cc in range(num_copy+1):
        if cc != num_copy:
            new_speed_seq.extend(speed_seq[:])
        else:
            if len(new_speed_seq)==max_len:
                break
            else:
                tmp = max_len-len(new_speed_seq)
                new_speed_seq.extend(speed_seq[:tmp])

    new_speed_seq = np.asarray(new_speed_seq).reshape(1,len(new_speed_seq),1)
    print "shape of extended speed is {}".format(new_speed_seq.shape)

    #decoder
    latent_dim = true_latent_vector.shape[1]
    print "latent_dim is %d"%latent_dim
    decoder = decoder(latent_dim=latent_dim,max_len=args.maxlength,dof=dof,speed_dim=1,
            hidden_dim_dec_list=[100,100],activation_dec_list=['tanh','tanh'])

    decoder.load_weights('./decoder.h5')
    output_sequence = decoder.predict(x=[true_latent_vector,new_speed_seq])

    if args.smooth :
        print "smoothing generated sequence"
        output_sequence = smooth_sequence(output_sequence,
                                         window_length=41,
                                         polyorder=3)
        #smooth speed sequence
    #     smoothed_speed = smooth_sequence(new_speed_seq,window_length=31,polyorder=3)
    #
    # #plot smoothed speed and unsmoothed speed
    # plt.figure()
    # plt.plot(new_speed_seq[0,:,0],label='smoothed')
    # plt.plot(smoothed_speed[0][:,0],label='unsmoothed')
    # plt.legend(bbox_to_anchor=(1.3, 1), loc=2, borderaxespad=0., fontsize=10)
    # plt.savefig('./results/speed_seq.png')
    # import pdb
    # pdb.set_trace()
    sequence = postprocess_seq(output_sequence,new_speed_seq,max_vector=max_vector,min_vector=min_vector,
                    max_vertical_position=max_vertical_position,min_vertical_position=min_vertical_position,
                    actor=1)


    #smooth sequence
    sequence = smooth_sequence(sequence, window_length=41, polyorder=3)

    #save to file
    print "saving long sequence..."
    act1_str = ''.join(c for c in args.activity if c.isupper())
    file_name = './results/'+act1_str + '_' + args.emotion + str(args.index)+'_longsequence_frame' +str(max_len)+ '.bvh'
    save_seq_2_bvh(sequence[0], file_name)
