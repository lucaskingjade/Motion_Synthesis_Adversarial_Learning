from Seq_AAE_V1.datasets.EmilyaDataset.src.get_dataset import get_dataset
import numpy as np
from keras.layers import Input,RepeatVector,merge,LSTM,SimpleRNN,Dense
from keras.models import Model
from keras.models import model_from_yaml
from synthesis_utils import postprocess_seq,smooth_sequence,save_seq_2_bvh,get_index_from_labels
from Seq_AAE_V1.datasets.EmilyaDataset.src.get_dataset import get_label_by_name


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
    import argparse
    from keras.models import load_model

    activity_list = ['Being Seated', 'Lift', 'Simple Walk', 'Throw', 'Knocking on the Door', 'Move Books',
                     'Sitting Down', 'Walk with smth in the Hands']
    emotion_list = ['Anger', 'Anxiety', 'Joy', 'Neutral', 'Panic Fear', 'Pride', 'Sadness', 'Shame']

    # get dataset
    Actor_name = ['Brian', 'Elie', 'Florian', 'Hu', 'Janina', 'Jessica', 'Maria', 'Muriel', 'Robert', 'Sally',
                  'Samih', 'Tatiana']
    h5_handle, filesource = get_dataset(actor_list=Actor_name, window_width=200, shift_step=20, sampling_interval=None)

    parser = argparse.ArgumentParser()
    parser.add_argument('--emotion1', help='specify the emotion name that you want to animate', type=str)
    parser.add_argument('--emotion2', help='specify the emotion name that you want to animate', type=str)
    parser.add_argument('--activity1', help='specify the activity name that you want to animate', type=str)
    parser.add_argument('--activity2', help='specify the activity name that you want to animate', type=str)
    parser.add_argument('--index1', help='specify which seqeunce you want to visualize', type=int)
    parser.add_argument('--index2', help='specify which seqeunce you want to visualize', type=int)
    parser.add_argument('--dataset', help='choose a dataset: training,valid,test', type=str)
    parser.add_argument('--number',help='the number of interpolated latent vectors between start and end points',type=int)

    args = parser.parse_args()

    #parsing arguments
    if args.emotion1 ==None or args.emotion2==None:
        raise ValueError("please specify two emotions")
    elif args.emotion1!=None and args.emotion2 !=None:
        emotion1 = args.emotion1
        emotion2 = args.emotion2
    if args.activity1 !=None and args.activity2 != None:
        activity1 = args.activity1
        activity2 = args.activity2
    else:
        raise ValueError("please specify two activities")
    #parsing source of dataset
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
    print 'Successfully get sequences'

    # get the corresponding set of sequences
    activity_label1 = get_label_by_name(args.activity1, whichlabel=1)
    emotion_label1 = get_label_by_name(args.emotion1, whichlabel=2)
    indices1 = get_index_from_labels(activity_label1, emotion_label1, None, Y1, Y2)

    activity_label2 = get_label_by_name(args.activity2, whichlabel=1)
    emotion_label2 = get_label_by_name(args.emotion2, whichlabel=2)
    indices2 = get_index_from_labels(activity_label2, emotion_label2, None, Y1, Y2)



    if args.index1 != None and args.index2 != None:
        index1 = args.index1
        index2 = args.index2
    else:
        print "randomly choose the indices of sequences"
        index1 = np.random.randint(low=0,high=len(indices1)-1)
        index2 = np.random.randint(low=0,high=len(indices2)-1)
        print "index1:%d"%index1
        print "index2:%d" % index2

    start_sequence = X[indices1[index1]:indices1[index1]+1]
    end_sequence = X[indices2[index2]:indices2[index2]+1]

    #load encoder and decoder
    print 'load encoder ...'
    encoder = load_model('./encoder.h5')
    print 'load decoder ...'
    decoder = load_model('./decoder.h5')

    if len(start_sequence.shape) != 3:
        start_sequence= start_sequence.reshape((1, start_sequence.shape[0], start_sequence.shape[1]))

    if len(end_sequence.shape) != 3:
        end_sequence = end_sequence.reshape((1, end_sequence.shape[0], end_sequence.shape[1]))

    start_latent = encoder.predict(start_sequence)
    end_latent = encoder.predict(end_sequence)
    output_sequences = generate_seq_by_interpolate_latent(decoder,start_latent[0],end_latent[0],nb=args.number)
    #postprocess seqeunces
    sequences = postprocess_seq(output_sequences,globalposition,max_min_vector)
    print 'len of output is {}'.format(output_sequences.shape)
    assert len(sequences) == args.number
    # save sequences
    act1_str = ''.join(c for c in activity1 if c.isupper())
    act2_str = ''.join(c for c in activity2 if c.isupper())
    for i, seq in enumerate(sequences):
        print 'save sequences to files'
        file_name = act1_str+'_'+emotion1+'_'+act2_str+'_'+emotion2+'_sequence_interpolate_latent_' + str(i) + '.bvh'
        save_seq_2_bvh(seq, file_name)

    #save start sequence and end_sequence
    start_sequence = postprocess_seq(start_sequence,globalposition,max_min_vector)
    end_sequence = postprocess_seq(end_sequence,globalposition,max_min_vector)
    print "saving start and end sequences..."
    file_name = act1_str + '_' + emotion1 + '_' + act2_str + '_' + emotion2 + '_start_seq'  + '.bvh'
    save_seq_2_bvh(start_sequence[0],file_name)
    file_name = act1_str + '_' + emotion1 + '_' + act2_str + '_' + emotion2 + '_end_seq' + '.bvh'
    save_seq_2_bvh(end_sequence[0],file_name)


