import numpy as np
from keras.models import model_from_yaml
import os
import sys
root_path = os.getenv('Seq_AAE_V1')
sys.path.append(root_path)
from Seq_AAE_V1.datasets.dataset import Emilya_Dataset

dataset_obj = Emilya_Dataset(window_width=200,shift_step=20,
                             sampling_interval=None,
                             with_velocity=False,
                             number=200,nb_valid=200,nb_test=10000)
X=dataset_obj.test_X[:,:,1:]
path_model='./Encoder200.yaml'
with open(path_model,'r') as f:
    encoder = model_from_yaml(f)
encoder.load_weights(path_model[:-4]+'h5')
latent_codes = encoder.predict(X,batch_size=1000,verbose=0)
mean_z= np.mean(latent_codes,axis=0)
cov_d=np.cov(latent_codes,rowvar=False)
np.savez('mean_cov.npz',[mean_z,cov_d])