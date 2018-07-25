#

def convert_indices_2_onehot(targets, nb_labels):
    tmp_targets = targets.astype(int)
    ohm = np.zeros((tmp_targets.shape[0], nb_labels), dtype=np.float32)
    ohm[np.arange(tmp_targets.shape[0]), tmp_targets] = 1.0
    return ohm


if __name__=='__main__':
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--which_epoch", default=200,
                        type=int)
    parser.add_argument('--activity_emotion',default=0,type=int)
    args = parser.parse_args()

    from Seq_AAE_V1.datasets.dataset import Emilya_Dataset,Emilya_Dataset_With_Context
    data_obj = Emilya_Dataset(window_width=200, shift_step=20, sampling_interval=None, with_velocity=False,
                          number=None,nb_valid=None, nb_test=None)
    from keras.models import model_from_yaml

    encoder_path = '../encoder'+str(args.which_epoch)+'.yaml'
    with open(encoder_path,'r') as f:
        encoder  = model_from_yaml(f)
    encoder.load_weights(encoder_path[:-4]+'h5')
    X = data_obj.valid_X[:,:,1:]
    Y1 = data_obj.valid_Y1[:]
    Y2 = data_obj.valid_Y2[:]
    Y1 = convert_indices_2_onehot(Y1, 8)
    Y2 = convert_indices_2_onehot(Y2, 8)
    if args.activity_emotion==0:
        latent_codes = encoder.predict(X,batch_size=1000,verbose=0)
    elif args.activity_emotion==1:
        latent_codes = encoder.predict([X,Y1], batch_size=1000, verbose=0)
    elif args.activity_emotion == 2:
        latent_codes = encoder.predict([X, Y2], batch_size=1000, verbose=0)
    mean = np.mean(latent_codes,axis=0)
    cov = np.cov(latent_codes,rowvar=False)
    print "shape of mean is {}".format(mean.shape)
    print "shape of cov is {}".format(cov.shape)
    np.savez('mean_cov.npz',mean,cov)