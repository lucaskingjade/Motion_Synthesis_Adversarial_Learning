import numpy as np
from keras.models import load_model,model_from_yaml
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def reconstruction(encoder,decoder,input_seqs,input_speeds,max_vector,min_vector,max_vertical_position,min_vertical_position,actor=0,
                   **kwargs):
    latent_codes = encoder.predict(x=[input_seqs, input_speeds])
    generated_seqs = decoder.predict(x=[latent_codes,input_speeds])
    #postprocess dataset
    #plot root vertical orientation
    #generated_seqs[6][:,1] = input_sequences[6][:,1].copy()
    # plot_orientation(input_sequences[6],generated_seqs[6],column=1)
    # plot_orientation(input_sequences[6], generated_seqs[6], column=46)
    # generated_seqs[6][:,46] = savgol_filter(generated_seqs[6][:,46],window_length=21,polyorder=3)
    # plot_orientation(input_sequences[6], generated_seqs[6], column=46,name='smoothed')
    # plot_orientation(input_sequences[6], generated_seqs[6], column=47)
    # generated_seqs[6][:, 47] = savgol_filter(generated_seqs[6][:, 47], window_length=21, polyorder=3)
    # plot_orientation(input_sequences[6], generated_seqs[6], column=47, name='smoothed')
    # plot_orientation(input_sequences[6], generated_seqs[6], column=48)
    # generated_seqs[6][:, 48] = savgol_filter(generated_seqs[6][:, 48], window_length=21, polyorder=3)
    # plot_orientation(input_sequences[6], generated_seqs[6], column=48, name='smoothed')
    #generated_seqs[6] = smooth_sequence(generated_seqs[6])
    if 'smooth' not in kwargs.keys():
        smooth = False
    else:
        smooth = kwargs['smooth']
    if smooth :
        generated_seqs = smooth_sequence(generated_seqs,
                                         window_length=kwargs['window_length'],
                                         polyorder=kwargs['polyorder'])
    #plot_orientation(input_seqs[0], generated_seqs[0],column=3, name='unsmooth')
    #plot_orientation(input_seqs[0], generated_seqs[0],column=3, name='smooth')
    sequences = postprocess_seq(generated_seqs,input_speeds,max_vector,min_vector,
                                max_vertical_position,min_vertical_position,actor=actor)

    original_seqs = postprocess_seq(input_seqs,input_speeds,max_vector,min_vector,
                               max_vertical_position,min_vertical_position,actor=actor)
    #plot_orientation(original_seqs[0], sequences[0], column=3, name='y_rotation')
    #plot_orientation(original_seqs[0],sequences[0],column=0,name='x_position')
    #plot_orientation(original_seqs[0], sequences[0],column=2, name='z_position')
    for i, seq in enumerate(sequences):
        filename = 'reconstructed_'+kwargs['savename']+'_'+str(i)+'.bvh'
        save_seq_2_bvh(seq,filename,step=1)
    for i ,seq in enumerate(original_seqs):
        filename = 'original_' + kwargs['savename'] + '_' + str(i) + '.bvh'
        save_seq_2_bvh(seq, filename, step=1)

def smooth_sequence(sequences,**kwargs):
    smoothed_sequence = []

    for seq in sequences:
        cur_seq = savgol_filter(seq, window_length=kwargs['window_length'],polyorder=kwargs['polyorder'],axis=-2)
        smoothed_sequence.append(cur_seq)
    return smoothed_sequence



def plot_orientation(sequence1,sequence2,column=1,**kwargs):
    plt.figure()
    plt.plot(sequence1[:,column],label='original root orientation')
    plt.plot(sequence2[:,column],label='reconstructed root orientation')
    plt.legend(bbox_to_anchor=(1.3, 1), loc=2, borderaxespad=0.,fontsize=10)
    if 'name' in kwargs.keys():
        filename = 'orientation_'+str(column)+'_'+kwargs['name']+'.png'
        plt.savefig(filename)
    else:
        plt.savefig('orientation_'+str(column)+'.png')
    plt.close()

def plot_curve_coefficient(sequences,name_list=['original','unsmoothed','smoothed'],dim=1,**kwargs):
    plt.figure()
    ax = plt.subplot(111)
    for name,seq in zip(name_list,sequences):
        print name
        print 'shape {}'.format(seq.shape)
        plt.plot(seq[0][:,dim],label=name)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    plt.legend(bbox_to_anchor=(1.05, 1.0-0.06), loc=2, borderaxespad=0., fontsize=10)
    plt.savefig('curve_orientation_'+str(dim)+'_'+kwargs['savename']+'.png')
    plt.close()

def save_seq_2_bvh(sequence, sequence_name,step=1):
    print "sequence_name: %s" % sequence_name
    nb_frames = int(sequence.shape[0])
    print 'length of sequence = %f' % nb_frames
    filepath = os.getenv('Seq_AAE_V1')+'Training/Seq_AAE/Expr_Emilya/generation/'
    with open(filepath+ 'bvh_no_frame.bvh', 'rb') as fl:
        lines = fl.readlines()
        lines[-1] = lines[-1].replace(str(0.08333),str(0.008333*step))
        lines[-2] = lines[-2].replace(str(50), str(nb_frames))
    save_name = sequence_name
    with open(save_name, 'wb') as fl:
        fl.writelines(lines)
        for i in range(len(sequence)):
            fl.writelines(' '.join([str(sequence[i][j]) for j in range(len(sequence[i]))]))
            fl.writelines('\n')
    fl.close()



def postprocess_seq(sequences,speeds, max_vector, min_vector, max_vertical_position,min_vertical_position,actor=0):
    #denormalize the vertical position according to specific actor
    sequences = denormalize_vertical_position(sequences,max_vertical_position,min_vertical_position,actor)
    print "length of sequence2 is {}".format(len(sequences))
    #denormalize the rest 69 DOFs according to max_vector and min_vector
    sequences = denormalize(sequences,max_vector,min_vector)

    #direction can be evaluated by vertical orientation
    directions = compute_direction(sequences)
    #convert position to speed
    init_position = np.asarray([0.0,0.0],dtype=np.float32)
    xz_position = convert_speed_to_position(init_position,directions,speeds)
    sequences = add_xz_positions(xz_position,sequences)

    return sequences


def add_xz_positions(xz_position_seq,sequences):
    rval = []
    for positions, seq in zip(xz_position_seq,sequences):
      new_seq = np.zeros((seq.shape[0],seq.shape[1]+2),dtype=np.float32)
      new_seq[:,0] = positions[:,0].copy()
      new_seq[:,2] = positions[:,1].copy()
      new_seq[:,1] = seq[:,0].copy()
      new_seq[:,3:] = seq[:,1:].copy()
      rval.append(new_seq)

    rval=np.asarray(rval,dtype=np.float32)
    return rval


def denormalize_vertical_position(sequences,max_vertical_position,min_vertical_position,actor=0):
    rval = []
    max_vector = max_vertical_position[actor]
    min_vector = min_vertical_position[actor]
    for seq in sequences:
      new_seq = np.zeros(seq.shape,dtype=np.float32)
      new_seq[:] =seq*(max_vector[1:-1]-min_vector[1:-1])+min_vector[1:-1]
      rval.append(new_seq)

    rval= np.asarray(rval,dtype=np.float32)
    return rval


def denormalize(sequences,max_vector,min_vector):
    rval = []
    for seq in sequences:
      new_seq = np.zeros(seq.shape,dtype=np.float32)
      new_seq[:] = seq*(max_vector[2:] - min_vector[2:]) + min_vector[2:]
      rval.append(new_seq)

    rval = np.asarray(rval,dtype=np.float32)

    return rval


def compute_direction(sequences):
    rval = []
    for seq in sequences:
      orientation_radians = np.radians(seq[:,1])
      direction =np.zeros((seq.shape[0],2),dtype=np.float32)
      direction[:,0] = np.sin(orientation_radians)
      direction[:,1] = np.cos(orientation_radians)
      rval.append(direction)

    rval = np.asarray(rval,dtype=np.float32)

    return rval


def convert_speed_to_position(init_position,directions_seq,speed_seqs):
    rval = []
    for directions, speeds in zip(directions_seq,speed_seqs):
      T = speeds.shape[0]
      position_sequence = np.zeros((T, 2), dtype=np.float32)
      for i, (n,v) in enumerate(zip(directions,speeds)):
           if i ==0:
                position_sequence[i,:] = init_position[:].copy()
           elif i == T-1:
                position_sequence[i,:] = v * n+position_sequence[i-1,:]
           else:
                #position_sequence[i,:] = 2*speeds[i-1]*directions[i-1]+position_sequence[i-2,:]
                cur_direction = (directions[i - 2] +directions[i])/np.linalg.norm((directions[i - 2] +directions[i]))
                position_sequence[i, :] = 2 * speeds[i - 1] * cur_direction+ position_sequence[i - 2, :]

      rval.append(position_sequence)

    rval = np.asarray(rval,dtype=np.float32)
    return rval


def get_index_from_labels(activity_label,emotion_label,actor_label,Y1,Y2,Y3=None):
    index1 = np.where(Y1==activity_label)[0]
    index2 = np.where(Y2 == emotion_label)[0]
    if emotion_label==None:
        index2 = range(len(Y1))
    if Y3==None:
        index3= range(len(Y1))
    else:
        index3 = np.where(Y3 == actor_label)[0]
    index12 = set(index1).intersection(index2)
    index123 = set(index12).intersection(index3)
    print list(index123)
    return list(index123)

def get_data_from_h5(dataset_name,h5_handle):
    if dataset_name=='training':
        dataset = h5_handle['training_set']
    elif dataset_name =='valid':
        dataset=h5_handle['valid_set']
    elif dataset_name=='test':
        dataset =h5_handle['test_set']
    else:
        raise ValueError()

    X = dataset['X'][:]
    speed_sequences = dataset['speed_xz'][:]
    Y1= dataset['Y1'][:]
    Y2 = dataset['Y2'][:]
    Y3 = dataset['Y3'][:]

    max_vector = h5_handle['max_min_vectors']['max_vector'][:]
    min_vector = h5_handle['max_min_vectors']['min_vector'][:]
    max_vertical_position = h5_handle['vertical_position']['max_vertical_position'][:]
    min_vertical_position = h5_handle['vertical_position']['min_vertical_position'][:]
    return X,speed_sequences,Y1,Y2,Y3,max_vector,min_vector,max_vertical_position,min_vertical_position


def accelerate_speed(speed,multiple):
    length = speed[0].shape[0]
    a = np.arange(start=0.0,stop=length,step=1.0)
    assert len(a)==length
    new_speed =np.zeros(speed.shape)
    print "new_speed shape is {}".format(new_speed.shape)
    for i,c in enumerate(speed[0]):
        #new_speed[0][i]= np.min((c*multiple,1.0))
        new_speed[0][i] = c * multiple

    return new_speed

def get_line_space_array(array1, array2, nb):
    c = np.array([np.linspace(i, j, nb) for i, j in zip(array1, array2)]).transpose()
    print c
    return c


# def generate_seq_by_interpolate_latent(decoder, start_latent, end_latent, nb):
#     latent_vectors = get_line_space_array(start_latent, end_latent, nb)
#     sequences = decoder.predict(latent_vectors)
#     return sequences





if __name__=='__main__':
    import h5py
    with open('./encoder.yaml','r') as file:
        encoder = model_from_yaml(file)
    encoder.load_weights('./encoder.h5')
    with open('./decoder.yaml','r') as file:
        decoder = model_from_yaml(file)
    decoder.load_weights('./decoder.h5')

    #path of dataset

    with open('./meta_data.txt') as file:
        filepath = file.readlines()[-1]
        root_path = os.getenv('Seq_AAE_V1') + 'datasets/EmilyaDataset/'
        if filepath[-3:] !='.h5':
            print "didn't find datasource path in meta_data.txt file"
            filepath = root_path+'alldataset_frame200_shift20_without_sampling.h5'
        else:
            print 'find data source in meta_data.txt'
            path, filename = os.path.split(filepath)
            filepath = root_path+filename

    handle = h5py.File(filepath,'r')
    input_sequences = handle['valid_set']['X'][0:20]
    input_speeds = handle['valid_set']['speed_xz'][0:20]
    Y1=handle['valid_set']['Y1'][0:20]
    Y2 = handle['valid_set']['Y2'][0:20]
    Y3 = handle['valid_set']['Y3'][0:20]
    print "Y1={}\n,Y2={},\nY3={}".format(Y1,Y2,Y3)
    max_vector = handle['max_min_vectors']['max_vector'][:]
    min_vector = handle['max_min_vectors']['min_vector'][:]
    max_vertical_position = handle['vertical_position']['max_vertical_position'][:]
    min_vertical_position = handle['vertical_position']['min_vertical_position'][:]
    reconstruction(encoder,decoder,input_sequences,input_speeds,max_vector,min_vector,max_vertical_position,min_vertical_position,actor=0)






