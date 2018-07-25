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

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--which_epoch", default=200,
                        type=int)
    parser.add_argument("--what_encoder",default=0,type=int)

    args = parser.parse_args()

    #load data
    dataset_obj = Emilya_Dataset(window_width=200,shift_step=20,
                                 sampling_interval=None,
                                 with_velocity=False,
                                 number=None, nb_valid=None, nb_test=None)

    test_X = dataset_obj.test_X[:,:,1:]
    test_Y2 = dataset_obj.test_Y2[:]
    test_Y2 = convert_indices_2_onehot(test_Y2,nb_labels=8)
    del dataset_obj
    ##load encoder
    which_epoch = args.which_epoch
    encoder_name ='../encoder'+str(which_epoch)+'.yaml'
    with open(encoder_name,mode='r') as fl:
        encoder = model_from_yaml(fl)
    encoder.load_weights(encoder_name[:-4]+'h5')
    if args.what_encoder==0:
        latent_codes = encoder.predict(test_X,batch_size=100)
    else:
        latent_codes = encoder.predict([test_X, test_Y2],batch_size=100)

    # load decoder
    print 'load decoder'
    decoder_name = encoder_name.replace('encoder', 'decoder')
    with open(decoder_name) as fl:
        decoder = model_from_yaml(fl)
    decoder.load_weights(decoder_name[:-4] + 'h5')
    target_Y2=[]
    target_seqs = []
    target_latents = []
    test_Y2 = np.argmax(test_Y2,axis=-1)
    for i in range(8):
        indices = np.where(test_Y2!=i)[0]
        cur_labels = np.asarray(len(indices)*[i])
        cur_labels = convert_indices_2_onehot(cur_labels,8)
        target_latents.extend(latent_codes[indices])
        #cur_sequences =decoder.predict([latent_codes[indices],cur_labels],batch_size =1000)
        #target_seqs.extend(cur_sequences)
        target_Y2.extend(len(indices)*[i])
    print 'begin predict sequesnce'
    target_latents = np.asarray(target_latents)
    target_Y2 = np.asarray(target_Y2)
    target_Y2 = convert_indices_2_onehot(target_Y2,8)
    print target_latents.shape
    print target_Y2.shape
    target_seqs = decoder.predict([target_latents,target_Y2])
    assert len(target_Y2) == len(target_seqs)

    ##load classifier, select good sequences from test set which are able to be correctly classified.
    # load emotion classifier
    path_classifier = '/data1/home/wang.qi/CODE/python/Seq_AAE_V1/Training/Classifier/Expr_Emilya/expr001/'
    #path_classifier = '/Users/qiwang/Documents/PythonCodes_wq/Keras_Projects/Seq_AAE_V1/Training/Classifier/Expr_Emilya/expr001/'

    with open(path_classifier + 'emotion_classifier.yaml') as fl:
        classifier = model_from_yaml(fl)

    classifier.load_weights(path_classifier + 'emotion_classifier.h5')
    classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    loss,accuracy= classifier.evaluate(target_seqs,target_Y2,batch_size=1000)
    print 'Loss:%f ,Accuracy:%f'%(loss,accuracy)
    target_predicted_labels = classifier.predict(target_seqs,batch_size =1000)
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









