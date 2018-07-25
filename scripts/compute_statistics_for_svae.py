import numpy as np
from keras.models import model_from_yaml
from Seq_AAE_V1.datasets.dataset import generate_positive_samples

# def convert_indices_2_onehot(targets, nb_labels):
#     tmp_targets = targets.astype(int)
#     ohm = np.zeros((tmp_targets.shape[0], nb_labels), dtype=np.float32)
#     ohm[np.arange(tmp_targets.shape[0]), tmp_targets] = 1.0
#     return ohm

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq2seq", default=False,
                        action='store_true')
    parser.add_argument("--nb_total", default=None,
                        type=int)
    parser.add_argument("--nb_query", default=None,
                        type=int)
    parser.add_argument("--remove_label", default=None,
                        type=int)
    parser.add_argument("--which_epoch", default=200,
                        type=int)

    parser.add_argument('--query_difference',default=False,action='store_true')
    parser.add_argument('--normalized',default=False,action='store_true')
    args = parser.parse_args()
    from sklearn.neighbors import KDTree
    model_path ='../decoder_epoch'+str(args.which_epoch)+'.yaml'

    nb_total = args.nb_total
    nb_query = args.nb_query
    with open(model_path,mode='r') as f:
        decoder = model_from_yaml(f)
    decoder.load_weights(model_path[:-4]+'h5')

    #load data
    from Seq_AAE_V1.datasets.dataset import Emilya_Dataset,Emilya_Dataset_With_Context

    data_obj = Emilya_Dataset(window_width=200, shift_step=20, sampling_interval=None, with_velocity=False,
                              number=None,nb_valid=None, nb_test=None)
    print data_obj.test_X.shape
    print data_obj.valid_X.shape
    X_1 = data_obj.valid_X[:, :, 1:]
    X_2 = data_obj.test_X[:,:,1:]
    X = np.concatenate((X_1,X_2),axis=0)
    print "size of all real seqs is {}".format(X.shape)
    if nb_total is None:
        nb_total = X.shape[0]
        nb_query = int(nb_total/10)
    Y1 = np.concatenate((data_obj.valid_Y1[:],data_obj.test_Y1[:]),axis=0)
    Y2 = np.concatenate((data_obj.valid_Y2[:],data_obj.test_Y2[:]),axis=0)
    # Y1 = convert_indices_2_onehot(Y1,8)
    # Y2 = convert_indices_2_onehot(Y2,8)

    print "nb_total ={}".format(nb_total)
    print "nb_query ={}".format(nb_query)
    ##generate noise vectors
    if args.seq2seq is True:
        mean_cov = np.load('mean_cov.npz','r')
        names = mean_cov.files
        mean = mean_cov[names[1]]
        covariance = mean_cov[names[0]]
    else:
        mean = np.zeros(50)
        covariance = np.eye(N=50) * 1.0
    print 'shape of mean {}'.format(mean.shape)
    print 'shape of cov {}'.format(covariance.shape)
    noise_vectors = generate_positive_samples(nb_total,
                                                   mean, covariance,
                                                   'Gaussian',seed=2345)

    generated_seqs,covariance = decoder.predict(x=noise_vectors,batch_size=1000)
    ##compute the (x_t-x_t-1)
    if args.query_difference is not True:
        generated_seqs=generated_seqs[:,::10,:]
        generated_seqs = generated_seqs.reshape(generated_seqs.shape[0],generated_seqs.shape[1]*generated_seqs.shape[2])
        X=X[:,::10,:]
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        ##compute the mean max min distance from generated seqs to real sequences.
        print 'compute the first distance '
        np.random.seed(1234)
        X = np.random.permutation(X)
        kdtree = KDTree(X[:nb_total, :], leaf_size=10)

        distances, noise_neighbours = kdtree.query(generated_seqs[:nb_query], k=1)
        mean_distance_1 = np.mean(distances)
        max_distance_1 = np.max(distances)
        min_distance_1 = np.min(distances)
        std_distance_1 = np.std(distances)
        if args.normalized is True:
            X_norm =  np.mean(np.sum(np.square(X),axis=-1),axis=0)
        else:
            X_norm = 1.
        mean_distance_normalized_1 = mean_distance_1 / X_norm
        max_distance_normalized_1 = max_distance_1 / X_norm
        min_distance_normalized_1 = min_distance_1 / X_norm
        # copmute the distance from real data to generated data

        np.random.seed(1223)
        X = np.random.permutation(X)
        print 'compute the second distance '
        kdtree = KDTree(generated_seqs[:nb_total, :], leaf_size=10)
        distances, noise_neighbours = kdtree.query(X[:nb_query,:], k=1)

        mean_distance_2 = np.mean(distances)
        max_distance_2 = np.max(distances)
        min_distance_2 = np.min(distances)
        std_distance_2 = np.std(distances)
        mean_distance_normalized_2= mean_distance_2/X_norm
        max_distance_normalized_2 = max_distance_2 / X_norm
        min_distance_normalized_2 = min_distance_2 / X_norm
        print 'mean distance from generated seq to real data is {},{}'.format(mean_distance_1,mean_distance_normalized_1)
        print 'mean distance from real data to generated data is {},{}'.format(mean_distance_2,mean_distance_normalized_2)
        print 'max distance from generated seq to real data is {},{}'.format(max_distance_1,max_distance_normalized_1)
        print 'max distance from real data to generated data is {},{}'.format(max_distance_2,max_distance_normalized_2)
        print 'min distance from generated seq to real data is {},{}'.format(min_distance_1,min_distance_normalized_1)
        print 'min distance from real data to generated data is {},{}'.format(min_distance_2,min_distance_normalized_2)
        print 'std distance from generated seq to real data is {}'.format(std_distance_1)
        print 'std distance from real data to generated data is {}'.format(std_distance_2)

    else:
        difference = generated_seqs[:, 1:, :] - generated_seqs[:, :-1, :]
        X_difference = X[:,1:,:]-X[:,:-1,:]
        difference = difference[:,::10,:]
        X_difference = X_difference[:,::10,:]
        print 'shape of generated difference {}'.format(difference.shape)
        print 'shape of X_difference {}'.format(X_difference.shape)
        X_difference= X_difference.reshape(X_difference.shape[0],
                                                X_difference.shape[1] * X_difference.shape[2])
        difference = difference.reshape(difference.shape[0],difference.shape[1]*difference.shape[2])

        print 'compute the first distance '
        np.random.seed(1234)
        X_difference = np.random.permutation(X_difference)
        kdtree = KDTree(X_difference[:nb_total], leaf_size=10)
        distances, noise_neighbours = kdtree.query(difference[:nb_query], k=1)
        mean_distance_1 = np.mean(distances)
        max_distance_1 = np.max(distances)
        min_distance_1 = np.min(distances)
        std_distance_1 = np.std(distances)
        if args.normalized is True:
            X_norm =  np.mean(np.sum(np.square(X_difference),axis=-1),axis=0)
        else:
            X_norm = 1.
        mean_distance_normalized_1 = mean_distance_1 / X_norm
        max_distance_normalized_1 = max_distance_1 / X_norm
        min_distance_normalized_1 = min_distance_1 / X_norm

        np.random.seed(1223)
        X_difference = np.random.permutation(X_difference)
        print 'compute the second distance '
        kdtree = KDTree(difference[:nb_total, :], leaf_size=10)
        distances, noise_neighbours = kdtree.query(X_difference[:nb_query, :], k=1)
        mean_distance_2 = np.mean(distances)
        max_distance_2 = np.max(distances)
        min_distance_2 = np.min(distances)
        std_distance_2 = np.std(distances)
        mean_distance_normalized_2 = mean_distance_2 / X_norm
        max_distance_normalized_2 = max_distance_2 / X_norm
        min_distance_normalized_2 = min_distance_2 / X_norm

        print 'mean differential distance from generated seq to real data is {},{}'.format(mean_distance_1,mean_distance_normalized_1)
        print 'mean differential distance from real data to generated data is {},{}'.format(mean_distance_2,mean_distance_normalized_2)
        print 'max differential distance from generated seq to real data is {},{}'.format(max_distance_1,max_distance_normalized_1)
        print 'max differential distance from real data to generated data is {},{}'.format(max_distance_2,max_distance_normalized_2)
        print 'min differential distance from generated seq to real data is {},{}'.format(min_distance_1,min_distance_normalized_1)
        print 'min differential distance from real data to generated data is {},{}'.format(min_distance_2,min_distance_normalized_2)
        print 'std differential distance from generated seq to real data is {}'.format(std_distance_1)
        print 'std differential distance from real data to generated data is {}'.format(std_distance_2)

    if args.seq2seq is True:
        print "Mean and Cov are computed from real data"


