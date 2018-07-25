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
    parser.add_argument("--which_epoch", default=390,
                        type=int)
    parser.add_argument("--what_encoder", default=0,
                        type=int)
    parser.add_argument("--nb_generated", default=10000,
                        type=int)
    parser.add_argument("--nb_test", default=10000,
                        type=int)
    parser.add_argument('--bandwidth', default=0.09,type=float)

    args = parser.parse_args()

    data_obj = Emilya_Dataset(window_width=200, shift_step=20, sampling_interval=None, with_velocity=False,
                              number=None, nb_valid=None, nb_test=None)
    print data_obj.test_X.shape
    X_valid = data_obj.valid_X[:args.nb_generated,:,1:]
    Y1_valid = data_obj.valid_Y1[:args.nb_generated]
    Y2_valid = data_obj.valid_Y2[:args.nb_generated]

    X = data_obj.test_X[:, :, 1:]
    Y1 = data_obj.test_Y1[:]
    Y2 = data_obj.test_Y2[:]
    print "size of all real seqs is {}".format(X.shape)
    X_test = X[:args.nb_test, ::10, :]
    Y1_test = Y1[:args.nb_test]
    Y2_test = Y2[:args.nb_test]
    # X_for_transform = X[args.nb_test:args.nb_test+args.nb_generated,:,:]
    # Y2_for_transform = Y2[args.nb_test:args.nb_test+args.nb_generated]

    ##load encoder
    which_epoch =args.which_epoch
    encoder_name = '../encoder' + str(which_epoch) + '.yaml'
    with open(encoder_name, mode='r') as fl:
        encoder = model_from_yaml(fl)
    encoder.load_weights(encoder_name[:-4] + 'h5')
    # latent_codes = encoder.predict(X_test, batch_size=1000)
    # content_latents = latent_codes[:, :50]
    # style_latents = latent_codes[:, 50:]

    latent_codes = encoder.predict(X_valid, batch_size=1000)
    content_latents = latent_codes[:, :50]
    style_latents = latent_codes[:, 50:]

    ##load classifier, select good sequences from test set which are able to be correctly classified.
    # load emotion classifier
    path_classifier = '/data1/home/wang.qi/CODE/python/Seq_AAE_V1/Training/Classifier/Expr_Emilya/expr001/'

    with open(path_classifier + 'emotion_classifier.yaml') as fl:
        classifier = model_from_yaml(fl)

    classifier.load_weights(path_classifier + 'emotion_classifier.h5')
    classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    predicted_labels = classifier.predict(X_valid, verbose=0, batch_size=1000)
#    good_test_sequences = []
    good_style_latent = []
    for i in range(8):
        indices1 = np.where(Y1_valid== i)[0]
        for j in range(8):
            indices2 = np.where(Y2_valid== j)[0]
            indices = list(set(indices1).intersection(indices2))
            #cur_predicted_labels = np.asarray([predicted_labels[c] for c in indices])
            cur_predicted_labels = predicted_labels[indices]
            k = np.argmax(cur_predicted_labels[:, j], axis=0)
            print 'best score is {}'.format(cur_predicted_labels[k])
 #           good_test_sequences.append(X[indices[k]])
            good_style_latent.append(style_latents[indices[k]])

    new_latent = []
    new_Y2 = []
    for i, xx in enumerate(X_valid):
        act_label = Y1_valid[i]
        j = int(act_label * 8)
        cur_style = good_style_latent[j:j + 8]
        cur_latent = np.concatenate((np.asarray([content_latents[i]] * 8), cur_style), axis=-1)
        new_latent.extend(cur_latent)
        new_Y2.extend([0., 1., 2., 3., 4., 5., 6., 7.])

    target_latents = np.asarray(new_latent)[:args.nb_generated]
    #target_Y2 = np.asarray(new_Y2)
    print target_latents.shape

    # load decoder
    decoder_name = encoder_name.replace('encoder', 'decoder')
    with open(decoder_name) as fl:
        decoder = model_from_yaml(fl)

    decoder.load_weights(decoder_name[:-4] + 'h5')
    target_seqs = decoder.predict(target_latents, batch_size=1000)

    generated_seqs = target_seqs[:,::10,:]
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
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


