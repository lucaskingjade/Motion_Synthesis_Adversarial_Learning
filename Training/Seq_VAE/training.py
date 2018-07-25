import sys
import os

root_path = os.getenv('Seq_AAE_V1')
sys.path.append(root_path)

from Seq_AAE_V1.datasets.dataset import Emilya_Dataset, Emilya_Dataset_With_Context
from Seq_AAE_V1.models.Seq_VAE.seq_vae import Sequence_VAE
data_obj = Emilya_Dataset(window_width=200, shift_step=20, sampling_interval=None, with_velocity=False,number=None)

model = Sequence_VAE(latent_dim=50,latent_activation='tanh',
                                    latent_BN = False,
                                    hidden_dim_enc_list=[100,100],activation_enc_list=['tanh','tanh'],
                                    hidden_dim_dec_list=None,activation_dec_list=None,
                                    batch_size=200,max_epoch=401,
                                    optimiser_autoencoder='rmsprop',
                                    output_type = 'rnn',
                                    lr_autoencoder=1e-3,
                                    decay_autoencoder=0.0,
                                    momentum_autoencoder=0.0,
                                    prior_noise_type='Gaussian',
                                    data_obj=data_obj,
                                    lambda_reguler=[0.],
                                    regulariser = ['null'],
                                    alpha_rate=1e-2,alpha_bias=5.0,
                                    begin_tune_alpha_epoch=0.,
                                    recurrent_dropout_outputlayer=0.0,
                                    fixed_alpha=1.0)
#model.set_up_dataset()
model.training(data_obj)
