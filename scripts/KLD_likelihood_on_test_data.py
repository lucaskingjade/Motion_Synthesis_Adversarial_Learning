##first use generated models to generate some sequences or load generated sequences from a file
from Seq_AAE_V1.datasets.dataset import generate_positive_samples
from Seq_AAE_V1.datasets.dataset import Emilya_Dataset, Emilya_Dataset_With_Context
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
def convert_indices_2_onehot(targets, nb_labels):
    tmp_targets = targets.astype(int)
    ohm = np.zeros((tmp_targets.shape[0], nb_labels), dtype=np.float32)
    ohm[np.arange(tmp_targets.shape[0]), tmp_targets] = 1.0
    return ohm


if __name__=='__main__':
    import argparse
    import numpy as np
    from keras.models import model_from_yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_data", default=False, action='store_true')
    parser.add_argument("--seq2seq", default=False, action='store_true')
    parser.add_argument("--svae", default=False, action='store_true')
    parser.add_argument("--which_epoch", default=200,
                        type=int)
    parser.add_argument("--nb_generated", default=10000,
                        type=int)
    parser.add_argument("--nb_test", default=10000,
                        type=int)
    parser.add_argument('--activity_emotin_condition', default=0, type=int)
    parser.add_argument('--bandwidth', default=None,type=float)

    args = parser.parse_args()

    ##generate noise vectors
    if args.seq2seq is True:
        mean_cov = np.load('mean_cov.npz', 'r')
        names = mean_cov.files
        mean = mean_cov[names[1]]
        covariance = mean_cov[names[0]]
    else:
        mean = np.zeros(50)
        covariance = np.eye(N=50) * 1.0
    print 'shape of mean {}'.format(mean.shape)
    print 'shape of cov {}'.format(covariance.shape)
    noise_vectors = generate_positive_samples(args.nb_generated,
                                              mean, covariance,
                                              'Gaussian', seed=2345)
    data_obj = Emilya_Dataset(window_width=200, shift_step=20, sampling_interval=None, with_velocity=False,
                              number=None, nb_valid=None, nb_test=None)
    print data_obj.test_X.shape
    X = data_obj.test_X[:, :, 1:]
    print "size of all real seqs is {}".format(X.shape)
    Y1 = data_obj.test_Y1[:]
    Y2 = data_obj.test_Y2[:]
    Y1 = convert_indices_2_onehot(Y1, 8)
    Y2 = convert_indices_2_onehot(Y2, 8)
    #if it's ERD then use 70 dimension
    X_test = X[:args.nb_test,::10,:]
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    if args.load_data is False:
        # load model and randomly generated some seqs
        if args.svae is False:
            model_path = '../decoder' + str(args.which_epoch) + '.yaml'
        else:
            model_path = '../decoder_epoch' + str(args.which_epoch) + '.yaml'
        activity_emotion_condition = args.activity_emotin_condition  # default 0: no condition; 1: activity conditional 2:emotion conditional
        with open(model_path, mode='r') as f:
            decoder = model_from_yaml(f)
        decoder.load_weights(model_path[:-4] + 'h5')

        if activity_emotion_condition == 0:
            generated_seqs = decoder.predict(x=noise_vectors, batch_size=1000)
            ##compute the (x_t-x_t-1)
        elif activity_emotion_condition == 1:
            generated_seqs = decoder.predict(x=[noise_vectors, Y1[:args.nb_generated]], batch_size=1000)
        elif activity_emotion_condition == 2:
            generated_seqs = decoder.predict(x=[noise_vectors, Y2[:args.nb_generated]], batch_size=1000)
    else:

        data = np.load('./generated_seqs.npz')
        generated_seqs = data[data.files[0]]

    if args.svae is True:
        generated_seqs = generated_seqs[0]
    generated_seqs = generated_seqs[:,::10,:]
    generated_seqs=generated_seqs.reshape(generated_seqs.shape[0],generated_seqs.shape[1]*generated_seqs.shape[2])

    if args.bandwidth is None:
        ##grid search
        params = {'bandwidth': np.logspace(-1, 0., 10)}
        grid = GridSearchCV(KernelDensity(), params, cv=3, verbose=1)
        X_search = np.random.permutation(X)[:10000,::10,:]
        X_search = X_search.reshape(X_search.shape[0],X_search.shape[1]*X_search.shape[2])
        grid_result = grid.fit(X_search)


        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        for params, mean_score, scores in grid_result.grid_scores_:
            print("scores.mean:%f (score.std:%f) with: %r" % (scores.mean(), scores.std(), params))
        #bandwidth = 0.25
        bandwidth = grid_result.best_params_
    else:
        bandwidth = args.bandwidth

    ParzenWindow = KernelDensity(bandwidth=bandwidth, algorithm='auto', kernel='gaussian', metric='euclidean')
    print "shape of generated_seqs is {}".format(generated_seqs.shape)
    print "shape of X_test is {}".format(X_test.shape)
    ParzenWindow.fit(generated_seqs)
    print "finish fiting"
    probability_test_set =ParzenWindow.score_samples(X_test)
    print "compute score"
    k = 10
    N_pb = probability_test_set.shape[0]
    mean_probability_list= []
    for i in range(k):
        start = i * N_pb / 10
        end = start + (N_pb / 10)
        mean_probability_list.append(np.mean(probability_test_set[start:end]))
        # std_saae_list.append(np.std(probability_saae_test_set[start:end]))
    mean_saae = np.mean(mean_probability_list)
    std_saae = np.std(mean_probability_list)
    print "10-fold, mean of pb:{0},std:{1}".format(mean_saae, std_saae)
    print 'bandwidth is {}'.format(args.bandwidth)


