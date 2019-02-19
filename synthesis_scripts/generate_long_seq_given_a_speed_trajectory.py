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
from Seq_AAE_V1.datasets.dataset import generate_positive_samples
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

    #get dataset
    Actor_name = ['Brian', 'Elie', 'Florian', 'Hu', 'Janina', 'Jessica', 'Maria', 'Muriel', 'Robert', 'Sally',
                  'Samih', 'Tatiana']
    h5_handle, filesource = get_dataset(actor_list=Actor_name,window_width=200,shift_step=20,sampling_interval=None)

    parser = argparse.ArgumentParser()
    parser.add_argument('--emotion', help='specify the emotion name that you want to animate', type=str)
    parser.add_argument('--activity', help='specify the activity name that you want to animate', type=str)
    parser.add_argument('--dataset', help='choose a dataset: training,valid,test', type=str)
    parser.add_argument('--maxlength', help='specify the length of the generated sequence', type=int)
    parser.add_argument('--smooth',help='smooth the generated sequence',action='store_true')
    parser.add_argument('--simulate', help='if use simulated speed sequence', action='store_true')
    parser.add_argument('--latentsource',help='where do the latent vectors come from',type=str)
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
        X = h5_handle['test_set']['X'][:]
        Y1 = h5_handle['test_set']['Y1'][:]
        Y2 = h5_handle['test_set']['Y2'][:]
        Y3 = h5_handle['test_set']['Y3'][:]
        speed_xz = h5_handle['test_set']['speed_xz'][:]
    else:
        raise ValueError('the dataset doesnt exit')
    # get the sequence specified by the index
    max_vector = h5_handle['max_min_vectors']['max_vector'][:]
    min_vector = h5_handle['max_min_vectors']['min_vector'][:]
    max_vertical_position = h5_handle['vertical_position']['max_vertical_position'][:]
    min_vertical_position = h5_handle['vertical_position']['min_vertical_position'][:]

    #dof
    dof = X.shape[-1]

    #get the corresponding set of sequences
    activity_label = get_label_by_name(args.activity, whichlabel=1)
    emotion_label = get_label_by_name(args.emotion, whichlabel=2)
    print activity_label
    print emotion_label
    #indices = get_index_from_labels(activity_label, emotion_label,None, Y1, Y2)
    indices = get_index_from_labels(activity_label, None, None, Y1, Y2)

    x_seq = X[indices[0:10]]
    speed_seq = speed_xz[indices[0:10]]
    print "shape of x_seq_2 is {}".format(x_seq.shape)
    print 'load encoder ...'
    max_len = x_seq.shape[-2]
    encoder = encoder(max_len=max_len,dof=dof,
                      hidden_dim_enc_list=[100,100],activation_enc_list=['tanh','tanh'],
                      latent_dim=50,speed_dim=1)
    encoder.load_weights('./encoder.h5')
    if len(x_seq.shape) != 3:
        raise ValueError()
    true_latent_vectors = encoder.predict(x=[x_seq,speed_seq])
    print true_latent_vectors[:2]
    print "shape of true_latent_vectors is {}".format(true_latent_vectors.shape)

    if args.maxlength ==None:
        max_len = 100
    else:
        max_len = args.maxlength
    # acquire a 1000 frames of speed sequence from dataset
    import os
    root_path = os.getenv('Seq_AAE_V1')
    datafilename = root_path + 'continuous_data/Nt1CS_Brian.bvh'
    # read bvh data from Nt1CS_Brian.bvh
    from Seq_AAE_V1.bvh_tools.bvhplay.skeleton import *

    bvh = readbvh(datafilename)
    bvh.read()
    keyframes = np.asarray(bvh.keyframes, dtype=np.float32)
    from Seq_AAE_V1.datasets.EmilyaDataset.src.EmilyData_utils import compute_speed_xz

    ###define activity cuts
    frame_cuts = [(3000,3200),None,(0,400),None,(4040,4440),(7600,8000),(2800,3000),(5180,5580)]


    interval = frame_cuts[int(activity_label)]
    len_frames = interval[1]-interval[0]
    if interval ==None:
        raise ValueError('activity label is wrong, please change to anther activity')

    seq = [keyframes[interval[0]:interval[1]]]
    tmp_x ,long_speed_seq =compute_speed_xz(seq)
    long_speed_seq = long_speed_seq.reshape(long_speed_seq.shape[1],long_speed_seq.shape[2])
    print "shape of long_speed_seq is {}".format(long_speed_seq.shape)
    tmp_long_speed_seq= long_speed_seq
    for i in range(args.maxlength/len_frames -1):
        tmp_long_speed_seq = np.concatenate((tmp_long_speed_seq,long_speed_seq))
    print "shape of tmp_long_speed_seq is {}".format(tmp_long_speed_seq.shape)

    ##plot speed
    fig = plt.figure()
    plt.plot(tmp_long_speed_seq)
    plt.savefig('./speed_seq.png')

    if args.simulate==True:
        ##get speed from a sin fucntion
        fs = 2000.
        f= 20.
        tmp_long_speed_seq = [0.5*np.sin(2. * np.pi * f * (i / fs))+1.1 for i in np.arange(fs)]
        tmp_long_speed_seq = np.asarray(tmp_long_speed_seq).reshape(2000,1)
        fig = plt.figure()
        plt.plot(tmp_long_speed_seq)
        plt.savefig('./fake_speed_seq.png')
    if args.latentsource =='prior':
        latent_dim = 50
        mean = np.zeros(latent_dim)
        covariance = np.eye(N=latent_dim) * 1.0
        latent_vector = generate_positive_samples(10, mean, covariance, 'Gaussian', seed=np.random.randint(0, 2000))
        true_latent_vectors = latent_vector

    #duplicate the long_speed_seq to ten sequences
    long_speed_seq = []
    for i in range(10):
        long_speed_seq.append(tmp_long_speed_seq)

    long_speed_seq = np.asarray(long_speed_seq)
    print "shape of long_speed_seq is {}".format(long_speed_seq.shape)


    latent_dim = true_latent_vectors.shape[1]
    print "latent_dim is %d"%latent_dim
    decoder = decoder(latent_dim=latent_dim,max_len=args.maxlength,dof=dof,speed_dim=1,
            hidden_dim_dec_list=[100,100],activation_dec_list=['tanh','tanh'])

    decoder.load_weights('./decoder.h5')
    output_sequence = decoder.predict(x=[true_latent_vectors,long_speed_seq])


    if args.smooth:
        print "smoothing generated sequence"
        output_sequence = smooth_sequence(output_sequence,
                                         window_length=41,
                                         polyorder=3)

        #smooth speed sequence
    sequence = postprocess_seq(output_sequence,long_speed_seq,max_vector=max_vector,min_vector=min_vector,
                    max_vertical_position=max_vertical_position,min_vertical_position=min_vertical_position,
                    actor=1)


    #smooth sequence
    sequence = smooth_sequence(sequence, window_length=41, polyorder=3)
    #save to file

    print "saving long sequence..."
    for i,seq in enumerate(sequence):
        act1_str = ''.join(c for c in args.activity if c.isupper())
        str_suffix=''
        if args.simulate==True:
            str_suffix = 'simulate_'
        if args.latentsource !=None:
            str_suffix = str_suffix+args.latentsource

        file_name = './results/' + act1_str + '_' + args.emotion + '_longsequence_frame_given_speed_' \
                   +str_suffix + str(max_len) + '_' + str(i) + '.bvh'
        save_seq_2_bvh(seq, file_name)
