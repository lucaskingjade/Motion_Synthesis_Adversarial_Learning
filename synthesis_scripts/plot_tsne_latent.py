from sklearn.manifold import TSNE
from keras.layers import Input,LSTM,merge,Dense
from keras.models import Model
from Seq_AAE_V1.datasets.EmilyaDataset.src.get_dataset import get_dataset
import numpy as np
from keras.models import model_from_yaml
def convert_indices_2_onehot(targets, nb_labels):
    tmp_targets = targets.astype(int)
    ohm = np.zeros((tmp_targets.shape[0], nb_labels), dtype=np.float32)
    ohm[np.arange(tmp_targets.shape[0]), tmp_targets] = 1.0
    return ohm

def plot_latent_space( latent_codes, filename, Y1=None, Y2=None, dim_x=0, dim_y=1):
    if Y1 is None:
        Y1 = np.zeros(len(latent_codes), dtype=np.float32)
    if Y2 is None:
        Y2 = np.zeros(Y1.shape, dtype=np.float32)
    nb_act = len(np.unique(Y1))
    nb_em = len(np.unique(Y2))
    fig = plt.figure(figsize=(8, 8))
    color = plt.cm.rainbow(np.linspace(0, 1, nb_act * nb_em))
    # print np.unique(Y)
    for l, c in zip(range(nb_act * nb_em), color):
        y1 = l / nb_em
        y2 = l % nb_em
        idx_1 = np.where(Y1 == y1)[0]
        idx_2 = np.where(Y2 == y2)[0]
        idx = list(set(idx_1).intersection(idx_2))
        plt.scatter(latent_codes[idx, dim_x], latent_codes[idx, dim_y], c=c, label=l, s=8, linewidths=0)
        plt.xlim([-20.0,20.0])
        plt.ylim([-20.0,20.0])
    plt.legend(fontsize=15)
    plt.savefig(filename)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity_emotion", default=1,
                        type=int)
    parser.add_argument("--which_epoch", default=200,
                        type=int)
    parser.add_argument("--number_samples", default=1000,
                        type=int)
    parser.add_argument("--iters", default=1000,
                        type=int)
    parser.add_argument("--perplexity", default=100,
                        type=int)


    args = parser.parse_args()


    from Seq_AAE_V1.datasets.dataset import Emilya_Dataset, Emilya_Dataset_With_Context

    data_obj = Emilya_Dataset(window_width=200, shift_step=20, sampling_interval=None, with_velocity=False,
                              number=args.number_samples, nb_valid=args.number_samples,
                              nb_test=args.number_samples)
    print data_obj.test_X.shape
    print data_obj.valid_X.shape
    train_X = data_obj.train_X[:, :, 1:]
    train_Y1 = data_obj.train_Y1[:]
    train_Y2 = data_obj.train_Y2[:]
    valid_X = data_obj.valid_X[:, :, 1:]
    valid_Y1= data_obj.valid_Y1[:]
    valid_Y2 = data_obj.valid_Y2[:]
    test_X = data_obj.test_X[:, :, 1:]
    test_Y1 = data_obj.test_Y1[:]
    test_Y2 = data_obj.test_Y2[:]
    train_Y1 = convert_indices_2_onehot(train_Y1,nb_labels=8)
    train_Y2 = convert_indices_2_onehot(train_Y2, nb_labels=8)
    valid_Y1 = convert_indices_2_onehot(valid_Y1, nb_labels=8)
    valid_Y2 = convert_indices_2_onehot(valid_Y2, nb_labels=8)
    test_Y1 = convert_indices_2_onehot(test_Y1, nb_labels=8)
    test_Y2 = convert_indices_2_onehot(test_Y2, nb_labels=8)

    ##load encoder
    model_path ='../encoder'+str(args.which_epoch)+'.yaml'
    with open(model_path, mode='r') as f:
        encoder = model_from_yaml(f)
    encoder.load_weights(model_path[:-4] + 'h5')
    if args.activity_emotion==1:
        print 'activity_emotion is {}'.format(args.activity_emotion)
        train_latent = encoder.predict(x=[train_X,train_Y1],batch_size=1000)
        valid_latent = encoder.predict(x=[valid_X,valid_Y1],batch_size=1000)
        test_latent = encoder.predict(x=[test_X,test_Y1],batch_size=1000)
    elif args.activity_emotion==2:
        print 'activity_emotion is {}'.format(args.activity_emotion)
        train_latent = encoder.predict(x=[train_X, train_Y2], batch_size=1000)
        valid_latent = encoder.predict(x=[valid_X, valid_Y2], batch_size=1000)
        test_latent = encoder.predict(x=[test_X, test_Y2], batch_size=1000)
    else:
        print 'activity_emotion is {}'.format(args.activity_emotion)
        train_latent = encoder.predict(x=train_X, batch_size=1000)
        valid_latent = encoder.predict(x=valid_X, batch_size=1000)
        test_latent = encoder.predict(x=test_X, batch_size=1000)


    import numpy as np

    tsne = TSNE(n_components=2,perplexity=args.perplexity,n_iter=args.iters,verbose=2)
    print "begin to fit model"
    np.set_printoptions(suppress=True)
    new_train_latent = tsne.fit_transform(train_latent)
    new_valid_latent = tsne.fit_transform(valid_latent)
    new_test_latent = tsne.fit_transform(test_latent)
    print "finishing fitting"
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_latent_space(new_train_latent,filename='train_t_sne_Y1.png',Y1=train_Y1)
    plot_latent_space(new_valid_latent,filename='valid_t_sne_Y1.png',Y1=valid_Y1)
    plot_latent_space(new_test_latent,filename='test_t_sne_Y1.png',Y1=test_Y1)
    # plot_latent_space(new_X,filename='t_sne_Y2.png',Y2=Y2)
    #
    # plot_latent_space(new_X,filename='t_sne_Y1Y2.png',Y1=Y1, Y2=Y2)
