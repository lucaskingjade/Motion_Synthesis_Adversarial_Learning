'''
In this file, datasets class will be defined.
'''
from EmilyaDataset.src.get_dataset import *
from EmilyaDataset.src.posture_dataset import get_postures_EmilyDataset
import math
from Seq_AAE_V1.synthesis_scripts.synthesis_utils import denormalize,denormalize_vertical_position

class Basic_Dataset(object):
    def __init__(self):
        pass
    def extract_data(self):
        pass

class Emilya_Dataset(Basic_Dataset):

    def __init__(self,window_width = 200, shift_step=20,sampling_interval=None,with_velocity=False,number=None,nb_valid=None,nb_test=None):
        Actor_name = ['Brian', 'Elie', 'Florian', 'Hu', 'Janina', 'Jessica', 'Maria', 'Muriel', 'Robert', 'Sally',
                      'Samih','Tatiana']

        args = locals().copy()
        del args['self']
        self.__dict__.update(args)

        if with_velocity == False:
            self.hdf_handle, self.file_source =get_dataset(Actor_name,window_width,shift_step,sampling_interval)
        elif with_velocity == True:
            self.hdf_handle,self.file_source = get_dataset_with_velocity(Actor_name,window_width,shift_step,sampling_interval)
        else:
            raise ValueError('value of with_velocity is illegal')

        if self.sampling_interval is None:
            self.sampling_interval = 1

        self.extract_data(n=self.number,nb_valid=self.nb_valid,nb_test=self.nb_test)


    def postprocess(self,seqs,max_vector,min_vector):
        sequences = denormalize(seqs, max_vector, min_vector)
        new_seq = []
        for seq in sequences:
            position_coordinates = np.asarray([[1.,90.0,1.]]*seq.shape[0])
            new_seq.append(np.concatenate((position_coordinates,seq),axis=1))
        new_seq = np.asarray(new_seq)
        #sequences = add_xz_positions(xz_position, sequences)
        return new_seq


    def extract_data(self,n,nb_valid=None,nb_test=None):
        # if nb_test is None:
        #     nb_test = n
        # if nb_valid is None:
        #     nb_valid = n
        self.train_X = self.hdf_handle['training_set']['X'][:n]
        self.train_Y1 = self.hdf_handle['training_set']['Y1'][:n]
        self.train_Y2 = self.hdf_handle['training_set']['Y2'][:n]
        self.train_Y3 = self.hdf_handle['training_set']['Y3'][:n]
        self.train_speed = self.hdf_handle['training_set']['speed_xz'][:n]

        self.valid_X = self.hdf_handle['valid_set']['X'][:nb_valid]
        self.valid_Y1 = self.hdf_handle['valid_set']['Y1'][:nb_valid]
        self.valid_Y2 = self.hdf_handle['valid_set']['Y2'][:nb_valid]
        self.valid_Y3 = self.hdf_handle['valid_set']['Y3'][:nb_valid]
        self.valid_speed = self.hdf_handle['valid_set']['speed_xz'][:nb_valid]

        self.test_X = self.hdf_handle['test_set']['X'][:nb_test]
        self.test_Y1 = self.hdf_handle['test_set']['Y1'][:nb_test]
        self.test_Y2 = self.hdf_handle['test_set']['Y2'][:nb_test]
        self.test_Y3 = self.hdf_handle['test_set']['Y3'][:nb_test]
        self.test_speed = self.hdf_handle['test_set']['speed_xz'][:nb_test]

        self.max_vector = self.hdf_handle['max_min_vectors']['max_vector'][:]
        self.min_vector = self.hdf_handle['max_min_vectors']['min_vector'][:]

        self.max_len = self.train_X[0].shape[0]
        self.dof = self.train_X[0].shape[1]

        if self.with_velocity==True:
            self.train_delta_X = self.hdf_handle['training_set']['delta_X'][:n]
            self.valid_delta_X = self.hdf_handle['valid_set']['delta_X'][:n]
            self.test_delta_X = self.hdf_handle['test_set']['delta_X'][:n]
            self.max_velocity_vector = self.hdf_handle['max_min_velocity_vectors']['max_velocity_vector'][:]
            self.min_velocity_vector = self.hdf_handle['max_min_velocity_vectors']['min_velocity_vector'][:]
        print "fininshing data extracting"


#Emilya Dataset with Positions of root joint
class Emilya_Dataset_Velocity_xz(Basic_Dataset):

    def __init__(self,window_width = 200, shift_step=20,sampling_interval=None,with_velocity=False):
        Actor_name = ['Brian', 'Elie', 'Florian', 'Hu', 'Janina', 'Jessica', 'Maria', 'Muriel', 'Robert', 'Sally',
                      'Samih','Tatiana']

        args = locals().copy()
        del args['self']
        self.__dict__.update(args)

        if with_velocity == False:
            self.hdf_handle, self.file_source =get_dataset(Actor_name,window_width,shift_step,sampling_interval)
        elif with_velocity == True:
            self.hdf_handle,self.file_source = get_dataset_with_velocity(Actor_name,window_width,shift_step,sampling_interval)
        else:
            raise ValueError('value of with_velocity is illegal')

        self.extract_data()

    def extract_data(self):
        n = None
        self.train_X = self.hdf_handle['training_set']['X'][:n]
        self.train_Y1 = self.hdf_handle['training_set']['Y1'][:n]
        self.train_Y2 = self.hdf_handle['training_set']['Y2'][:n]
        self.train_Y3 = self.hdf_handle['training_set']['Y3'][:n]
        self.train_position_xz = self.hdf_handle['training_set']['velocity_xz_plane'][:n]
        self.max_magnitude_velocity = self.hdf_handle['max_min_xz_velocity_magnitude']['max_magnitude_velocity'][:]
        self.min_magnitude_velocity = self.hdf_handle['max_min_xz_velocity_magnitude']['min_magnitude_velocity'][:]

        self.valid_X = self.hdf_handle['valid_set']['X'][:n]
        self.valid_Y1 = self.hdf_handle['valid_set']['Y1'][:n]
        self.valid_Y2 = self.hdf_handle['valid_set']['Y2'][:n]
        self.valid_Y3 = self.hdf_handle['valid_set']['Y3'][:200]
        self.valid_position_xz = self.hdf_handle['valid_set']['velocity_xz_plane'][:n]

        self.test_X = self.hdf_handle['test_set']['X'][:n]
        self.test_Y1 = self.hdf_handle['test_set']['Y1'][:n]
        self.test_Y2 = self.hdf_handle['test_set']['Y2'][:n]
        self.test_Y3 = self.hdf_handle['test_set']['Y3'][:n]
        self.test_position_xz = self.hdf_handle['test_set']['velocity_xz_plane'][:n]

        self.max_len = self.train_X[0].shape[0]
        self.dof = self.train_X[0].shape[1]

        if self.with_velocity==True:
            self.train_delta_X = self.hdf_handle['training_set']['delta_X'][:n]
            self.valid_delta_X = self.hdf_handle['valid_set']['delta_X'][:n]
            self.test_delta_X = self.hdf_handle['test_set']['delta_X'][:n]
            self.max_velocity_vector = self.hdf_handle['max_min_velocity_vectors']['max_velocity_vector'][:]
            self.min_velocity_vector = self.hdf_handle['max_min_velocity_vectors']['min_velocity_vector'][:]
        print "fininshing data extracting"





def generate_positive_samples(nb, mean, covariance, nb_mixture=None, type='Gaussian', seed=None):
    if not seed is None:
        np.random.seed(seed)
    if type == 'Mixture' and (nb_mixture == None or nb_mixture <= 1):
        raise ValueError('input argument:type is wrong!')
    if type == 'Gaussian':
        latent_var_prior = np.random.multivariate_normal(mean, covariance, size=nb)
    elif type == 'Mixture':
        latent_var_prior = np.random.multivariate_normal(mean, covariance, size=nb)
        latent_var_prior[:, :2] = mixture_gaussian_sampling(nb, nb_mixture)
    elif type == 'Mixture_Uniform':
        latent_var_prior = np.random.uniform(-4.0, 4.0, size=(nb, len(mean)))
        latent_var_prior[:, :2] = mixture_gaussian_sampling(nb, nb_mixture)
    elif type == 'Mixture_10':
        latent_var_prior = np.random.multivariate_normal(mean, covariance, size=nb)
        latent_var_prior[:, :2] = mixture_gaussian_sampling(nb, nb_mixture, alpha=0.05, beta=1.0, gamma=5.0)
    else:
        raise ValueError('There is no such type of prior %s' % type)
    return latent_var_prior

def mixture_gaussian_sampling(number,num_mixture,alpha=0.05,beta = 1.0,gamma=2.0):
    number_per_mixture = number/(num_mixture)
    #assert number%num_mixture==0
    dim = 2.5
    covariance = np.eye(dim)
    covariance[1,1] =alpha
    covariance=covariance *beta
    degree_step = 360./num_mixture
    mean = np.asarray([gamma,0.])
    samples = np.zeros((number,2),dtype=np.float32)
    for i in range(num_mixture):
        theta_degree = degree_step*i
        theta = math.radians(theta_degree)
        transform_mat = np.asarray([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=np.float32).transpose()
        start = i*number_per_mixture
        end = start+number_per_mixture
        samples_mixture= np.dot(np.random.multivariate_normal(mean,covariance,size=number_per_mixture),transform_mat)
        samples[start:end,:] = samples_mixture
    #shuffle samples
    samples = np.random.permutation(samples)
    return samples




