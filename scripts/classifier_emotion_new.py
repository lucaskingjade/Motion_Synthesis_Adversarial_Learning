#evluate the emotion accuracy on the stylized sequences
from Seq_AAE_V1.datasets.dataset import Emilya_Dataset


def convert_indices_2_onehot(targets, nb_labels):
    tmp_targets = targets.astype(int)
    ohm = np.zeros((tmp_targets.shape[0], nb_labels), dtype=np.float32)
    ohm[np.arange(tmp_targets.shape[0]), tmp_targets] = 1.0
    return ohm


if __name__=='__main__':
    from keras.models import model_from_yaml
    import numpy as np
    #load data
    dataset_obj = Emilya_Dataset(window_width=200,shift_step=20,
                                 sampling_interval=None,
                                 with_velocity=False,
                                 number=None, nb_valid=None, nb_test=None)

    test_X = dataset_obj.test_X[:,:,1:]
    test_Y1 = dataset_obj.test_Y1
    test_Y2 = dataset_obj.test_Y2

    ##load encoder
    which_epoch = 390
    encoder_name ='../encoder'+str(which_epoch)+'.yaml'
    with open(encoder_name,mode='r') as fl:
        encoder = model_from_yaml(fl)
    encoder.load_weights(encoder_name[:-4]+'h5')
    latent_codes = encoder.predict(test_X,batch_size=100)
    content_latents = latent_codes[:,:50]
    style_latents = latent_codes[:,50:]

    ##load classifier, select good sequences from test set which are able to be correctly classified.
    # load emotion classifier
    path_classifier = '/data1/home/wang.qi/CODE/python/Seq_AAE_V1/Training/Classifier/Expr_Emilya/expr001/'

    with open(path_classifier + 'emotion_classifier.yaml') as fl:
        classifier = model_from_yaml(fl)

    classifier.load_weights(path_classifier + 'emotion_classifier.h5')
    classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    predicted_labels = classifier.predict(test_X, verbose=0, batch_size=1000)
    good_test_sequences = []
    good_style_latent = []
    for i in range(8):
        indices1 = np.where(test_Y1 == i)[0]
        for j in range(8):
            indices2 = np.where(test_Y2 == j)[0]
            indices = list(set(indices1).intersection(indices2))
            #cur_predicted_labels = np.asarray([predicted_labels[c] for c in indices])
            cur_predicted_labels = predicted_labels[indices]
            k = np.argmax(cur_predicted_labels[:, j], axis=0)
            print 'best score is {}'.format(cur_predicted_labels[k])
            good_test_sequences.append(test_X[indices[k]])
            good_style_latent.append(style_latents[indices[k]])

    good_test_sequences = np.asarray(good_test_sequences)
    print 'length of good_test_sequences is %d' % len(good_test_sequences)

    new_latent = []
    new_Y2 = []
    for i , xx in enumerate(test_X):
        act_label = test_Y1[i]
        j = int(act_label*8)
        cur_style = good_style_latent[j:j+8]
        cur_latent = np.concatenate((np.asarray([content_latents[i]]*8),cur_style),axis=-1)
        new_latent.extend(cur_latent)
        new_Y2.extend([0.,1.,2.,3.,4.,5.,6.,7.])

    target_latents =np.asarray(new_latent)
    target_Y2 = np.asarray(new_Y2)
    print target_latents.shape


    #load decoder
    decoder_name = encoder_name.replace('encoder','decoder')
    with open(decoder_name) as fl:
        decoder = model_from_yaml(fl)

    decoder.load_weights(decoder_name[:-4]+'h5')
    sequences = decoder.predict(target_latents,batch_size=1000)
    print ' begin compute classification accuracy'

    target_Y2 = convert_indices_2_onehot(target_Y2,8)
    loss,accuracy= classifier.evaluate(sequences,target_Y2,batch_size=1000)
    print 'Loss:%f ,Accuracy:%f'%(loss,accuracy)
    target_predicted_labels = classifier.predict(sequences,batch_size =1000)
    target_predicted_labels = np.argmax(target_predicted_labels,axis=-1)
    from sklearn.metrics import confusion_matrix
    target_Y2 = np.argmax(target_Y2,axis=-1)
    cnf_matrix=confusion_matrix(target_Y2,target_predicted_labels)
    # Compute confusion matrix
    np.set_printoptions(precision=2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from Seq_AAE_V1.experiments_esann.plot_confusion_matrix import plot_confusion_matrix
    # Plot non-normalized confusion matrix
    plt.figure()
    class_names = ['Anger', 'Anxiety', 'Joy', 'Neutral', 'Panic Fear', 'Pride', 'Sadness', 'Shame']
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
    plt.savefig('confusion_matrix_without_normailzation.png')
    plt.close()
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    plt.savefig('confusion_matrix_with_normailzation.png')









