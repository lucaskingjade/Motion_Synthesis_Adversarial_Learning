from EmilyData_utils import get_hdf5_file_name,get_label_by_name,load_data_from_npz,normalization_joint_angles
from EmilyData_utils import normalization_vertical_position_per_subject,truncate_long_sequence,compute_speed_xz,split_dataset
from EmilyData_utils import normalization_velocity,compute_velocity_xz_plane,compute_velocity_feature,down_sampling,count_distribution_length_frames
from EmilyData_utils import plot_distribution_length,compute_velocity_xz_plane_with_normalization,extract_activity,incremental_orientation
import os.path as path
import h5py
import numpy as np
import os

def get_dataset( actor_list, window_width=200, shift_step=20, sampling_interval=None):
    # search hdf5 file in ../ directory
    if sampling_interval ==None:
        sampling_interval=1
    if window_width % sampling_interval != 0:
        raise ValueError('window_width cannot be divided exactly by the sampling_interval')

    filename = get_hdf5_file_name(window_width, shift_step, sampling_interval)
    if path.isfile(filename):
        print 'Successfully find the dataset file!'
        hdf5_file = h5py.File(filename, mode='r')
        print 'Successully get data set!'
        return hdf5_file, filename
    else:
        print "Didn't find the dataset file. Creating hdf5 file from npz..."
        root_path= os.getenv('Seq_AAE_V1')
        path_dataset = root_path + 'datasets/EmilyaDataset/'
        npz_name = path_dataset + 'Emilya_Dataset.npz'
        X, Y1, Y2, Y3 = load_data_from_npz(npz_name)
        actor_labels = np.asarray([get_label_by_name(actor,whichlabel=3) for actor in actor_list])

        # normalization for global position coordinate of y axis
        X, max_vertical_position_each_actor, min_vertical_position_each_actor \
            = normalization_vertical_position_per_subject(X, Y3, actor_labels)

        # normalization for joint angles
        X, max_vector, min_vector = normalization_joint_angles(X)
        # split it into training set, validation set and test set
        training_set, valid_set, test_set = split_dataset(X, Y1, Y2, Y3, ratio_train=0.6, ratio_valid=0.2)

        # truncate long sequences into short sequences
        training_set = truncate_long_sequence(training_set, window_size=window_width, shift_step=shift_step)
        valid_set = truncate_long_sequence(valid_set, window_size=window_width, shift_step=shift_step)
        test_set = truncate_long_sequence(test_set, window_size=window_width, shift_step=shift_step)
        # compute speed in the xz plane
        X, training_speed_xz = compute_speed_xz(training_set[0])
        training_set = (X, training_set[1], training_set[2], training_set[3])
        X, valid_speed_xz = compute_speed_xz(valid_set[0])
        valid_set = (X, valid_set[1], valid_set[2], valid_set[3])
        X, test_speed_xz = compute_speed_xz(test_set[0])
        test_set = (X, test_set[1], test_set[2], test_set[3])

        # save datasets, speed_xz,max_vertical_position_each_actor,min_vertical_position_each_actor,
        # max_vector,min_vector to hdf5 file
        file_handle = h5py.File(filename, 'w')
        training_set_group = file_handle.create_group('training_set')
        valid_set_group = file_handle.create_group('valid_set')
        test_set_group = file_handle.create_group('test_set')
        vertical_position_group = file_handle.create_group('vertical_position')
        max_min_vector_group = file_handle.create_group('max_min_vectors')

        training_set_group.create_dataset('X', data=training_set[0])
        training_set_group.create_dataset('Y1', data=training_set[1])
        training_set_group.create_dataset('Y2', data=training_set[2])
        training_set_group.create_dataset('Y3', data=training_set[3])
        training_set_group.create_dataset('speed_xz', data=training_speed_xz)

        valid_set_group.create_dataset('X', data=valid_set[0])
        valid_set_group.create_dataset('Y1', data=valid_set[1])
        valid_set_group.create_dataset('Y2', data=valid_set[2])
        valid_set_group.create_dataset('Y3', data=valid_set[3])
        valid_set_group.create_dataset('speed_xz', data=valid_speed_xz)

        test_set_group.create_dataset('X', data=test_set[0])
        test_set_group.create_dataset('Y1', data=test_set[1])
        test_set_group.create_dataset('Y2', data=test_set[2])
        test_set_group.create_dataset('Y3', data=test_set[3])
        test_set_group.create_dataset('speed_xz', data=test_speed_xz)

        vertical_position_group.create_dataset('max_vertical_position', data=max_vertical_position_each_actor)
        vertical_position_group.create_dataset('min_vertical_position', data=min_vertical_position_each_actor)

        max_min_vector_group.create_dataset('max_vector', data=max_vector)
        max_min_vector_group.create_dataset('min_vector', data=min_vector)

        file_handle.flush()
        file_handle.close()

        hdf5_file = h5py.File(filename, 'r')
        return hdf5_file, filename


#define a function for getting a dataset which include velocity feature
def get_dataset_with_velocity( actor_list, window_width=200, shift_step=20, sampling_interval=None):
    # search hdf5 file in ../ directory
    if sampling_interval == None:
        sampling_interval = 1
    if window_width % sampling_interval != 0:
        raise ValueError('window_width cannot be divided exactly by the sampling_interval')

    filename = get_hdf5_file_name(window_width, shift_step, sampling_interval)
    filename = filename[:-3]+'_with_velocity.h5'
    print 'searching the file %s in the dataset path'%filename
    if path.isfile(filename):
        print 'Successfully find the dataset file!'
        hdf5_file = h5py.File(filename, mode='r')
        print 'Successully get data set!'
        return hdf5_file, filename
    else:
        print "Didn't find the dataset file. Creating hdf5 file from npz..."
        root_path = os.getenv('Seq_AAE_V1')
        npz_name = root_path + 'datasets/EmilyaDataset/Emilya_Dataset.npz'
        X, Y1, Y2, Y3 = load_data_from_npz(npz_name)
        actor_labels = np.asarray([get_label_by_name(actor, whichlabel=3) for actor in actor_list])

        # normalization for global position coordinate of y axis
        X, max_vertical_position_each_actor, min_vertical_position_each_actor \
            = normalization_vertical_position_per_subject(X, Y3, actor_labels)

        # normalization for joint angles
        X, max_vector, min_vector = normalization_joint_angles(X)
        # split it into training set, validation set and test set
        training_set, valid_set, test_set = split_dataset(X, Y1, Y2, Y3, ratio_train=0.6, ratio_valid=0.2)

        # truncate long sequences into short sequences
        training_set = truncate_long_sequence(training_set, window_size=window_width, shift_step=shift_step)
        valid_set = truncate_long_sequence(valid_set, window_size=window_width, shift_step=shift_step)
        test_set = truncate_long_sequence(test_set, window_size=window_width, shift_step=shift_step)
        # compute speed in the xz plane

        X, training_speed_xz = compute_speed_xz(training_set[0])
        delta_X = compute_velocity_feature(X)
        delta_X,max_velocity_vector,min_velocity_vector = normalization_velocity(delta_X,return_max_min=True)
        training_set = (X, delta_X, training_set[1], training_set[2], training_set[3])

        X, valid_speed_xz = compute_speed_xz(valid_set[0])
        delta_X = compute_velocity_feature(X)
        delta_X= normalization_velocity(delta_X,max_velocity_vector,min_velocity_vector)
        valid_set = (X, delta_X, valid_set[1], valid_set[2], valid_set[3])

        X, test_speed_xz = compute_speed_xz(test_set[0])
        delta_X = compute_velocity_feature(X)
        delta_X = normalization_velocity(delta_X,max_velocity_vector,min_velocity_vector)
        test_set = (X, delta_X, test_set[1], test_set[2], test_set[3])

        # save datasets, speed_xz,max_vertical_position_each_actor,min_vertical_position_each_actor,
        # max_vector,min_vector to hdf5 file
        file_handle = h5py.File(filename, 'w')
        training_set_group = file_handle.create_group('training_set')
        valid_set_group = file_handle.create_group('valid_set')
        test_set_group = file_handle.create_group('test_set')
        vertical_position_group = file_handle.create_group('vertical_position')
        max_min_vector_group = file_handle.create_group('max_min_vectors')
        max_min_velocity_vector_group = file_handle.create_group('max_min_velocity_vectors')

        training_set_group.create_dataset('X', data=training_set[0])
        training_set_group.create_dataset('delta_X', data=training_set[1])
        training_set_group.create_dataset('Y1', data=training_set[2])
        training_set_group.create_dataset('Y2', data=training_set[3])
        training_set_group.create_dataset('Y3', data=training_set[4])
        training_set_group.create_dataset('speed_xz', data=training_speed_xz)

        valid_set_group.create_dataset('X', data=valid_set[0])
        valid_set_group.create_dataset('delta_X', data=valid_set[1])
        valid_set_group.create_dataset('Y1', data=valid_set[2])
        valid_set_group.create_dataset('Y2', data=valid_set[3])
        valid_set_group.create_dataset('Y3', data=valid_set[4])
        valid_set_group.create_dataset('speed_xz', data=valid_speed_xz)

        test_set_group.create_dataset('X', data=test_set[0])
        test_set_group.create_dataset('delta_X', data=test_set[1])
        test_set_group.create_dataset('Y1', data=test_set[2])
        test_set_group.create_dataset('Y2', data=test_set[3])
        test_set_group.create_dataset('Y3', data=test_set[4])
        test_set_group.create_dataset('speed_xz', data=test_speed_xz)

        vertical_position_group.create_dataset('max_vertical_position', data=max_vertical_position_each_actor)
        vertical_position_group.create_dataset('min_vertical_position', data=min_vertical_position_each_actor)

        max_min_vector_group.create_dataset('max_vector', data=max_vector)
        max_min_vector_group.create_dataset('min_vector', data=min_vector)
        max_min_velocity_vector_group.create_dataset('max_velocity_vector',data=max_velocity_vector)
        max_min_velocity_vector_group.create_dataset('min_velocity_vector', data=min_velocity_vector)

        file_handle.flush()
        file_handle.close()

        hdf5_file = h5py.File(filename, 'r')
        return hdf5_file, filename


#get_dataset_with_velocity_and_horizontal_velocity
#this is used for horizontal position regression
def get_dataset_with_velocity_and_horizontal_velocity(actor_list, window_width=200, shift_step=20, sampling_interval=None):
    # search hdf5 file in ../ directory
    if sampling_interval == None:
        sampling_interval = 1
    if window_width % sampling_interval != 0:
        raise ValueError('window_width cannot be divided exactly by the sampling_interval')

    filename = get_hdf5_file_name(window_width, shift_step, sampling_interval)
    filename = filename[:-3] + '_with_velocity_horizontal_velocity.h5'
    print 'searching the file %s in the dataset path' % filename
    if path.isfile(filename):
        print 'Successfully find the dataset file!'
        hdf5_file = h5py.File(filename, mode='r')
        print 'Successully get data set!'
        return hdf5_file, filename
    else:
        print "Didn't find the dataset file. Creating hdf5 file from npz..."
        root_path = os.getenv('Seq_AAE_V1')
        npz_name = root_path + 'datasets/EmilyaDataset/Emilya_Dataset.npz'
        X, Y1, Y2, Y3 = load_data_from_npz(npz_name)
        actor_labels = np.asarray([get_label_by_name(actor, whichlabel=3) for actor in actor_list])

        # normalization for global position coordinate in y axis
        X, max_vertical_position_each_actor, min_vertical_position_each_actor \
            = normalization_vertical_position_per_subject(X, Y3, actor_labels)

        # normalization for joint angles
        X, max_vector, min_vector = normalization_joint_angles(X)
        # split it into training set, validation set and test set
        training_set, valid_set, test_set = split_dataset(X, Y1, Y2, Y3, ratio_train=0.6, ratio_valid=0.2)

        # truncate long sequences into short sequences
        training_set = truncate_long_sequence(training_set, window_size=window_width, shift_step=shift_step)
        valid_set = truncate_long_sequence(valid_set, window_size=window_width, shift_step=shift_step)
        test_set = truncate_long_sequence(test_set, window_size=window_width, shift_step=shift_step)
        # compute speed in the xz plane

        X, train_velocity_xz_plane= compute_velocity_xz_plane(training_set[0])
        delta_X= compute_velocity_feature(X,keep_T=True)
        delta_X, max_velocity_vector, min_velocity_vector = normalization_velocity(delta_X, return_max_min=True)
        #concatenate X and delta_X
        X = np.concatenate((X,delta_X),axis=-1)
        assert X.shape[0] == delta_X.shape[0]
        assert X.shape[1] == delta_X.shape[1]
        assert X.shape[2] == delta_X.shape[2]*2

        training_set = (X, train_velocity_xz_plane,training_set[1], training_set[2], training_set[3])

        X, valid_velocity_xz_plane = compute_velocity_xz_plane(valid_set[0])
        delta_X = compute_velocity_feature(X,keep_T=True)
        delta_X = normalization_velocity(delta_X, max_velocity_vector, min_velocity_vector)
        #concatenate X and delta_X
        X = np.concatenate((X,delta_X),axis=-1)
        valid_set = (X, valid_velocity_xz_plane, valid_set[1], valid_set[2], valid_set[3])

        X, test_velocity_xz_plane= compute_velocity_xz_plane(test_set[0])
        delta_X = compute_velocity_feature(X,keep_T=True)
        delta_X = normalization_velocity(delta_X, max_velocity_vector, min_velocity_vector)
        #concatenate X and delta X
        X = np.concatenate((X,delta_X),axis=-1)
        test_set = (X, test_velocity_xz_plane, test_set[1], test_set[2], test_set[3])

        # save datasets, speed_xz,max_vertical_position_each_actor,min_vertical_position_each_actor,
        # max_vector,min_vector to hdf5 file
        file_handle = h5py.File(filename, 'w')
        training_set_group = file_handle.create_group('training_set')
        valid_set_group = file_handle.create_group('valid_set')
        test_set_group = file_handle.create_group('test_set')
        vertical_position_group = file_handle.create_group('vertical_position')
        max_min_vector_group = file_handle.create_group('max_min_vectors')
        max_min_velocity_vector_group = file_handle.create_group('max_min_velocity_vectors')

        training_set_group.create_dataset('X', data=training_set[0])
        training_set_group.create_dataset('velocity_xz_plane', data=training_set[1])
        #training_set_group.create_dataset('delta_X', data=training_set[1])
        training_set_group.create_dataset('Y1', data=training_set[2])
        training_set_group.create_dataset('Y2', data=training_set[3])
        training_set_group.create_dataset('Y3', data=training_set[4])
        #training_set_group.create_dataset('speed_xz', data=training_speed_xz)

        valid_set_group.create_dataset('X', data=valid_set[0])
        #valid_set_group.create_dataset('delta_X', data=valid_set[1])
        valid_set_group.create_dataset('velocity_xz_plane',data=valid_set[1])
        valid_set_group.create_dataset('Y1', data=valid_set[2])
        valid_set_group.create_dataset('Y2', data=valid_set[3])
        valid_set_group.create_dataset('Y3', data=valid_set[4])
        #valid_set_group.create_dataset('speed_xz', data=valid_speed_xz)

        test_set_group.create_dataset('X', data=test_set[0])
        #test_set_group.create_dataset('delta_X', data=test_set[1])
        test_set_group.create_dataset('velocity_xz_plane',data=test_set[1])
        test_set_group.create_dataset('Y1', data=test_set[2])
        test_set_group.create_dataset('Y2', data=test_set[3])
        test_set_group.create_dataset('Y3', data=test_set[4])
        #test_set_group.create_dataset('speed_xz', data=test_speed_xz)

        vertical_position_group.create_dataset('max_vertical_position', data=max_vertical_position_each_actor)
        vertical_position_group.create_dataset('min_vertical_position', data=min_vertical_position_each_actor)

        max_min_vector_group.create_dataset('max_vector', data=max_vector)
        max_min_vector_group.create_dataset('min_vector', data=min_vector)
        max_min_velocity_vector_group.create_dataset('max_velocity_vector', data=max_velocity_vector)
        max_min_velocity_vector_group.create_dataset('min_velocity_vector', data=min_velocity_vector)

        file_handle.flush()
        file_handle.close()

        hdf5_file = h5py.File(filename, 'r')
        return hdf5_file, filename

# get data with position on xz plane. For eliminate the global information,we use the velocity on xz plane to replace
# the position information
def get_dataset_with_velocity_xz(actor_list, activity_list, window_width=200, shift_step=20, sampling_interval=None,suffix=''):
    # search hdf5 file in ../ directory
    if sampling_interval == None:
        sampling_interval = 1
    if window_width % sampling_interval != 0:
        raise ValueError('window_width cannot be divided exactly by the sampling_interval')

    filename = get_hdf5_file_name(window_width, shift_step, sampling_interval)
    filename = filename[:-3] + '_with_xz_velocity'+suffix+'.h5'
    print 'searching the file %s in the dataset path' % filename
    if path.isfile(filename):
        print 'Successfully find the dataset file!'
        hdf5_file = h5py.File(filename, mode='r')
        print 'Successully get data set!'
        return hdf5_file, filename
    else:
        print "Didn't find the dataset file. Creating hdf5 file from npz..."
        root_path = os.getenv('Seq_AAE_V1')
        npz_name = root_path + 'datasets/EmilyaDataset/Emilya_Dataset.npz'
        X, Y1, Y2, Y3 = load_data_from_npz(npz_name)
        activity_labels = np.asarray([get_label_by_name(activity, whichlabel=1) for activity in activity_list])
        actor_labels = np.asarray([get_label_by_name(actor, whichlabel=3) for actor in actor_list])
        if len(activity_labels)<8:
            X,Y1,Y2,Y3 = extract_activity(X,Y1,Y2,Y3,activity_labels)

        # # count distribution of length
        # distribution_length = count_distribution_length_frames(X, Y1)
        # plot_distribution_length(distribution_length)
        # split it into training set, validation set and test set
        training_set, valid_set, test_set = split_dataset(X, Y1, Y2, Y3, ratio_train=0.6, ratio_valid=0.2)
        # downsampling the sequence into 60 frames per second
        training_set[0], training_set[1], training_set[2], training_set[3]= \
            down_sampling(training_set[0], training_set[1], training_set[2], training_set[3], step=sampling_interval)
        valid_set[0], valid_set[1], valid_set[2], valid_set[3] = \
            down_sampling(valid_set[0], valid_set[1], valid_set[2], valid_set[3], step=sampling_interval)
        test_set[0], test_set[1], test_set[2], test_set[3] = \
            down_sampling(test_set[0], test_set[1], test_set[2], test_set[3], step=sampling_interval)

        # truncate long sequences into short sequences
        training_set = truncate_long_sequence(training_set, window_size=window_width, shift_step=shift_step)
        valid_set = truncate_long_sequence(valid_set, window_size=window_width, shift_step=shift_step)
        test_set = truncate_long_sequence(test_set, window_size=window_width, shift_step=shift_step)

        # normalization for global position coordinate in y axis
        training_set[0], max_vertical_position_each_actor, min_vertical_position_each_actor \
            = normalization_vertical_position_per_subject(training_set[0], training_set[3], actor_labels)
        valid_set[0] = normalization_vertical_position_per_subject(valid_set[0], valid_set[3], actor_labels,
                                                                      max_vertical_position_list=max_vertical_position_each_actor,
                                                                      min_vertical_position_list=min_vertical_position_each_actor)

        test_set[0] = normalization_vertical_position_per_subject(test_set[0], test_set[3], actor_labels,
                                                                   max_vertical_position_list=max_vertical_position_each_actor,
                                                                   min_vertical_position_list=min_vertical_position_each_actor)

        #representing the vertical orientation by the incremental change
        training_set[0],train_orientations = incremental_orientation(training_set[0])
        valid_set[0] ,valid_orientations= incremental_orientation(valid_set[0])
        test_set[0] ,test_orientations= incremental_orientation(test_set[0])

        # normalization for joint angles
        training_set[0], max_vector, min_vector = normalization_joint_angles(training_set[0])
        valid_set[0] = normalization_joint_angles(valid_set[0],max_vector,min_vector)
        test_set[0] = normalization_joint_angles(test_set[0],max_vector,min_vector)

        # compute speed in the xz plane
        X, train_velocity_xz_plane,train_new_theta_0, max_magnitude_velocity,min_magnitude_velocity,max_theta,min_theta = \
            compute_velocity_xz_plane_with_normalization(training_set[0])

        training_set = (X, train_velocity_xz_plane, train_new_theta_0, train_orientations, training_set[1], training_set[2], training_set[3])

        X, valid_velocity_xz_plane,valid_new_theta_0= \
            compute_velocity_xz_plane_with_normalization(valid_set[0],
                                                         max_magnitude_velocity,
                                                         min_magnitude_velocity,
                                                         max_theta,min_theta)
        valid_set = (X, valid_velocity_xz_plane,valid_new_theta_0, valid_orientations,valid_set[1], valid_set[2], valid_set[3])

        X, test_velocity_xz_plane,test_new_theta_0= \
            compute_velocity_xz_plane_with_normalization(test_set[0],
                                                         max_magnitude_velocity,
                                                         min_magnitude_velocity,
                                                         max_theta,min_theta)
        test_set = (X, test_velocity_xz_plane, test_new_theta_0, test_orientations, test_set[1], test_set[2], test_set[3])

        # save datasets, speed_xz,max_vertical_position_each_actor,min_vertical_position_each_actor,
        # max_vector,min_vector to hdf5 file
        file_handle = h5py.File(filename, 'w')
        training_set_group = file_handle.create_group('training_set')
        valid_set_group = file_handle.create_group('valid_set')
        test_set_group = file_handle.create_group('test_set')
        vertical_position_group = file_handle.create_group('vertical_position')
        max_min_vector_group = file_handle.create_group('max_min_vectors')
        max_min_xz_velocity_magnitude_group = file_handle.create_group('max_min_xz_velocity_magnitude')
        max_min_theta = file_handle.create_group('max_min_theta')

        training_set_group.create_dataset('X', data=training_set[0])
        training_set_group.create_dataset('velocity_xz_plane', data=training_set[1])
        training_set_group.create_dataset('train_new_theta_0', data=training_set[2])
        training_set_group.create_dataset('train_orientation', data=training_set[3])
        training_set_group.create_dataset('Y1', data=training_set[4])
        training_set_group.create_dataset('Y2', data=training_set[5])
        training_set_group.create_dataset('Y3', data=training_set[6])
        max_min_xz_velocity_magnitude_group.create_dataset('max_magnitude_velocity',data=max_magnitude_velocity)
        max_min_xz_velocity_magnitude_group.create_dataset('min_magnitude_velocity', data=min_magnitude_velocity)
        max_min_theta.create_dataset('max_theta',data=max_theta)
        max_min_theta.create_dataset('min_theta', data=min_theta)

        valid_set_group.create_dataset('X', data=valid_set[0])
        valid_set_group.create_dataset('velocity_xz_plane',data=valid_set[1])
        valid_set_group.create_dataset('valid_new_theta_0', data=valid_set[2])
        valid_set_group.create_dataset('valid_orientation', data=valid_set[3])
        valid_set_group.create_dataset('Y1', data=valid_set[4])
        valid_set_group.create_dataset('Y2', data=valid_set[5])
        valid_set_group.create_dataset('Y3', data=valid_set[6])

        test_set_group.create_dataset('X', data=test_set[0])
        test_set_group.create_dataset('velocity_xz_plane',data=test_set[1])
        test_set_group.create_dataset('test_new_theta_0', data=test_set[2])
        test_set_group.create_dataset('test_orientation', data=test_set[3])
        test_set_group.create_dataset('Y1', data=test_set[4])
        test_set_group.create_dataset('Y2', data=test_set[5])
        test_set_group.create_dataset('Y3', data=test_set[6])

        vertical_position_group.create_dataset('max_vertical_position', data=max_vertical_position_each_actor)
        vertical_position_group.create_dataset('min_vertical_position', data=min_vertical_position_each_actor)

        max_min_vector_group.create_dataset('max_vector', data=max_vector)
        max_min_vector_group.create_dataset('min_vector', data=min_vector)


        file_handle.flush()
        file_handle.close()

        hdf5_file = h5py.File(filename, 'r')
        return hdf5_file, filename

if __name__ == "__main__":
    #cell testing.
    Actor_name = ['Brian', 'Elie', 'Florian', 'Hu', 'Janina', 'Jessica', 'Maria', 'Muriel', 'Robert', 'Sally', 'Samih',
                  'Tatiana']
    activity_list = ["Being Seated","Simple Walk","Walk with smth in the Hands","Move Books"]
    #activity_list = ["Simple Walk"]
    file_handle, source_name = get_dataset_with_velocity_xz(Actor_name,activity_list=activity_list,
                                                            window_width=200,shift_step=20,
                                                            sampling_interval=2,suffix='_BsSwWhMb_Incremental_orientation')
    file_handle.close()

