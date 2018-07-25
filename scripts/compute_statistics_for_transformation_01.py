import numpy as np
from keras.models import model_from_yaml
from Seq_AAE_V1.datasets.dataset import generate_positive_samples,Emilya_Dataset

def convert_indices_2_onehot(targets, nb_labels):
    tmp_targets = targets.astype(int)
    ohm = np.zeros((tmp_targets.shape[0], nb_labels), dtype=np.float32)
    ohm[np.arange(tmp_targets.shape[0]), tmp_targets] = 1.0
    return ohm

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--which_epoch", default=390,
                        type=int)
    parser.add_argument("--nb_total", default=None,
                        type=int)
    parser.add_argument("--nb_query", default=None,
                        type=int)
    parser.add_argument('--query_difference',default=False, action ='store_true')
    parser.add_argument('--normalized',default=False, action='store_true')
    args = parser.parse_args()
    print 'query_difference = {}'.format(args.query_difference)
    print 'normalized = {}'.format(args.normalized)
    from sklearn.neighbors import KDTree

    data_obj = Emilya_Dataset(window_width=200, shift_step=20, sampling_interval=None, with_velocity=False,
                              number=None, nb_valid=None, nb_test=None)
    print data_obj.test_X.shape
    X_valid = data_obj.valid_X[:args.nb_total*2,:, 1:]
    Y1_valid = data_obj.valid_Y1[:args.nb_total*2]
    Y2_valid = data_obj.valid_Y2[:args.nb_total*2]

    X = data_obj.test_X[:args.nb_total * 2, :, 1:]
    Y1 = data_obj.test_Y1[:args.nb_total * 2]
    Y2 = data_obj.test_Y2[:args.nb_total * 2]
    # X_valid = X[:10000, :, :]
    # Y1_valid = Y1[:10000]
    # Y2_valid = Y2[:10000]
    # print "size of all real seqs is {}".format(X.shape)
    # X_test = X[:args.nb_test, ::10, :]
    # Y1_test = Y1[:args.nb_test]
    # Y2_test = Y2[:args.nb_test]

    #load encoder
    which_epoch = args.which_epoch
    encoder_name = '../encoder' + str(which_epoch) + '.yaml'
    with open(encoder_name, mode='r') as fl:
        encoder = model_from_yaml(fl)
    encoder.load_weights(encoder_name[:-4] + 'h5')

    latent_codes = encoder.predict(X_valid, batch_size=1000)
    content_latents = latent_codes[:, :50]
    style_latents = latent_codes[:, 50:]

    #load classifier
    # load emotion classifier
    path_classifier = '/data1/home/wang.qi/CODE/python/Seq_AAE_V1/Training/Classifier/Expr_Emilya/expr001/'

    with open(path_classifier + 'emotion_classifier.yaml') as fl:
        classifier = model_from_yaml(fl)

    classifier.load_weights(path_classifier + 'emotion_classifier.h5')
    classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    predicted_labels = classifier.predict(X_valid, verbose=0, batch_size=1000)

    #    good_test_sequences = []
    good_style_latent = []
    nb_best_style = 3
    for i in range(8):
        indices1 = np.where(Y1_valid == i)[0]
        for j in range(8):
            indices2 = np.where(Y2_valid == j)[0]
            indices = list(set(indices1).intersection(indices2))
            indices = np.asarray(indices)
            # cur_predicted_labels = np.asarray([predicted_labels[c] for c in indices])
            cur_predicted_labels = predicted_labels[indices]
            #k = np.argmax(cur_predicted_labels[:, j], axis=0)
            sorted_indices = np.argsort(cur_predicted_labels[:,j],axis=0)
            k = sorted_indices[-nb_best_style:]
            #print k
            #good_test_sequences.append(X[indices[k]])
            good_style_latent.extend(style_latents[indices[k]])

    new_latent = []
    source_Y1 = []
    new_Y2 = []
    source_Y2 = []
    print "good_style_latent length is {}".format(len(good_style_latent))
    for i, xx in enumerate(X_valid):
        act_label = Y1_valid[i]
        em_label = int(Y2_valid[i])
        j = int(act_label * 8*nb_best_style)
        cur_style = good_style_latent[j:j + 8*nb_best_style]
        del cur_style[em_label:em_label+nb_best_style]
        cur_style = np.asarray(cur_style)
        # print "shape of cur_style"
        # print cur_style.shape
        # print "shape of content_latents"
        # print np.asarray([content_latents[i]*7*nb_best_style]).shape
        cur_latent = np.concatenate((np.asarray([content_latents[i]] * 7*nb_best_style), cur_style), axis=-1)
        new_latent.extend(cur_latent)
        em_list = [0., 1., 2., 3., 4., 5., 6., 7.]
        source_Y2.extend([em_label]*7*nb_best_style)
        source_Y1.extend([act_label]*7*nb_best_style)
        del em_list[em_label]
        ###here I need to flatten this list
        new_Y2.extend([[c]*nb_best_style for c in em_list])

    print "original new_Y2 shape is {}".format(np.asarray(new_Y2).shape)
    new_Y2 = np.asarray(new_Y2).flatten()
    assert len(new_Y2)== len(new_latent)
    target_latents = np.asarray(new_latent)[:]
    target_Y2 = np.asarray(new_Y2)[:]
    source_Y2 = np.asarray(source_Y2)[:]
    source_Y1 = np.asarray(source_Y1)[:]
    print target_latents.shape
    print "target Y2 shape is {}".format(target_Y2.shape)

    # load decoder
    decoder_name = encoder_name.replace('encoder', 'decoder')
    with open(decoder_name) as fl:
        decoder = model_from_yaml(fl)

    decoder.load_weights(decoder_name[:-4] + 'h5')
    del X_valid,Y1_valid,Y2_valid
    target_seqs = decoder.predict(target_latents, batch_size=1000)

    generated_seqs=target_seqs
    nb_total = args.nb_total
    nb_query = args.nb_query
    #query between the same emotion
    skip_number = 3
    if args.query_difference is not True:
        X_base = X[:nb_total, ::10, skip_number:]
        Y1_base = Y1[:nb_total]
        Y2_base = Y2[:nb_total]
        X_query = X[nb_total:(nb_total + nb_query), ::10, skip_number:]
        Y1_query = Y1[nb_total:(nb_total + nb_query)]
        Y2_query = Y2[nb_total:(nb_total + nb_query)]

        generated_seqs_base = generated_seqs[:nb_total, ::10, skip_number:]
        target_Y2_base = target_Y2[:nb_total]
        generated_seqs_query = generated_seqs[nb_total:(nb_total + nb_query), ::10, skip_number:]
        target_Y2_query = target_Y2[nb_total:(nb_total + nb_query)]
        # print "X.shape is {}".format(X.shape)
        X_base = X_base.reshape(X_base.shape[0], X_base.shape[1] * X_base.shape[2])
        X_query = X_query.reshape(X_query.shape[0], X_query.shape[1] * X_query.shape[2])

        generated_seqs_base = generated_seqs_base.reshape(generated_seqs_base.shape[0],
                                                          generated_seqs_base.shape[1] * generated_seqs_base.shape[2])

        generated_seqs_query = generated_seqs_query.reshape(generated_seqs_query.shape[0],
                                                            generated_seqs_query.shape[1] * generated_seqs_query.shape[
                                                                2])

        assert len(X_base) == len(Y2_base)
        assert len(X_query) == len(Y2_query)
        assert len(generated_seqs_base) == len(target_Y2_base)
        assert len(generated_seqs_query) == len(target_Y2_query)

        print 'compute the first distance '
        kdtree = KDTree(X_base, leaf_size=10)
        distances, noise_neighbours = kdtree.query(generated_seqs_query, k=1)
        mean_distance_1 = np.mean(distances)
        max_distance_1 = np.max(distances)
        min_distance_1 = np.min(distances)
        std_distance_1 = np.std(distances)

        print 'compute the second distance '
        kdtree = KDTree(generated_seqs_base, leaf_size=10)
        distances, noise_neighbours = kdtree.query(X_query, k=1)
        mean_distance_2 = np.mean(distances)
        max_distance_2 = np.max(distances)
        min_distance_2 = np.min(distances)
        std_distance_2 = np.std(distances)

        if args.normalized is True:
            X_norm = np.mean(np.sum(np.square(X_base), axis=-1), axis=0)
            # normalized_distances = distances/X_norm
        else:
            X_norm = 1.

        mean_distance_normalized_1 = mean_distance_1 / X_norm
        max_distance_normalized_1 = max_distance_1 / X_norm
        min_distance_normalized_1 = min_distance_1 / X_norm

        mean_distance_normalized_2 = mean_distance_2 / X_norm
        max_distance_normalized_2 = max_distance_2 / X_norm
        min_distance_normalized_2 = min_distance_2 / X_norm

        print 'mean distance from generated seq to real data is {},{}'.format(mean_distance_1, mean_distance_normalized_1)
        print 'mean distance from real data to generated data is {},{}'.format(mean_distance_2, mean_distance_normalized_2)
        print 'max distance from generated seq to real data is {},{}'.format(max_distance_1, max_distance_normalized_1)
        print 'max distance from real data to generated data is {},{}'.format(max_distance_2, max_distance_normalized_2)
        print 'min distance from generated seq to real data is {},{}'.format(min_distance_1, min_distance_normalized_1)
        print 'min distance from real data to generated data is {},{}'.format(min_distance_2, min_distance_normalized_2)

    else:
        X_base = X[:nb_total, :, skip_number:]
        Y1_base = Y1[:nb_total]
        Y2_base = Y2[:nb_total]
        X_query = X[nb_total:(nb_total + nb_query), :, skip_number:]
        Y1_query = Y1[nb_total:(nb_total + nb_query)]
        Y2_query = Y2[nb_total:(nb_total + nb_query)]
        generated_seqs_base = generated_seqs[:nb_total, :, skip_number:]
        target_Y2_base = target_Y2[:nb_total]
        generated_seqs_query = generated_seqs[nb_total:(nb_total + nb_query), :, skip_number:]
        target_Y2_query = target_Y2[nb_total:(nb_total + nb_query)]

        difference_base = generated_seqs_base[:, 1:, skip_number:] - generated_seqs_base[:, :-1, skip_number:]
        X_difference_base = X_base[:, 1:, skip_number:] - X_base[:, :-1, skip_number:]
        difference_query = generated_seqs_query[:, 1:, skip_number:] - generated_seqs_query[:, :-1, skip_number:]
        X_difference_query = X_query[:, 1:, skip_number:] - X_query[:, :-1, skip_number:]

        difference_base = difference_base[:, ::10, :]
        X_difference_base = X_difference_base[:, ::10, :]
        difference_query = difference_query[:, ::10, :]
        X_difference_query = X_difference_query[:, ::10, :]
        print 'shape of generated difference {}'.format(difference_base.shape)
        print 'shape of X_difference {}'.format(X_difference_query.shape)
        X_difference_base = X_difference_base.reshape(X_difference_base.shape[0],
                                                      X_difference_base.shape[1] * X_difference_base.shape[2])
        difference_base = difference_base.reshape(difference_base.shape[0],
                                                  difference_base.shape[1] * difference_base.shape[2])
        X_difference_query = X_difference_query.reshape(X_difference_query.shape[0],
                                                        X_difference_query.shape[1] * X_difference_query.shape[2])
        difference_query = difference_query.reshape(difference_query.shape[0],
                                                    difference_query.shape[1] * difference_query.shape[2])

        print 'compute the first distance '
        # np.random.seed(1234)
        # tmp_X_difference = np.random.permutation(X_difference)
        # tmp_difference = np.random.permutation(difference)
        kdtree = KDTree(X_difference_base, leaf_size=10)
        distances, noise_neighbours = kdtree.query(difference_query, k=1)
        mean_distance_1 = np.mean(distances)
        max_distance_1 = np.max(distances)
        min_distance_1 = np.min(distances)
        std_distance_1 = np.std(distances)

        print 'compute the second distance '
        kdtree = KDTree(difference_base[:nb_total, :], leaf_size=10)
        distances, noise_neighbours = kdtree.query(X_difference_query[:nb_query, :], k=1)
        mean_distance_2 = np.mean(distances)
        max_distance_2 = np.max(distances)
        min_distance_2 = np.min(distances)
        std_distance_2 = np.std(distances)

        if args.normalized is True:
            X_norm = np.mean(np.sum(np.square(X_difference_base), axis=-1), axis=0)
            # normalized_distances = distances/X_norm
        else:
            X_norm = 1.

        mean_distance_normalized_1 = mean_distance_1 / X_norm
        max_distance_normalized_1 = max_distance_1 / X_norm
        min_distance_normalized_1 = min_distance_1 / X_norm

        mean_distance_normalized_2 = mean_distance_2 / X_norm
        max_distance_normalized_2 = max_distance_2 / X_norm
        min_distance_normalized_2 = min_distance_2 / X_norm

        print 'mean differential distance from generated seq to real data is {},{}'.format(mean_distance_1,
                                                                                           mean_distance_normalized_1)
        print 'mean differential distance from real data to generated data is {},{}'.format(mean_distance_2,
                                                                                            mean_distance_normalized_2)
        print 'max differential distance from generated seq to real data is {},{}'.format(max_distance_1,
                                                                                          max_distance_normalized_1)
        print 'max differential distance from real data to generated data is {},{}'.format(max_distance_2,
                                                                                           max_distance_normalized_2)
        print 'min differential distance from generated seq to real data is {},{}'.format(min_distance_1,
                                                                                          min_distance_normalized_1)
        print 'min differential distance from real data to generated data is {},{}'.format(min_distance_2,
                                                                                           min_distance_normalized_2)



