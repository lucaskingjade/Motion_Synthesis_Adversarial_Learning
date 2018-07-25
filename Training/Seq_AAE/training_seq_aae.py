import sys
import os
root_path = os.getenv('Seq_AAE_V1')
sys.path.append(root_path)
from Seq_AAE_V1.datasets.dataset import Emilya_Dataset
from Seq_AAE_V1.models.Seq_AAE.seq_aae_new_loss import Sequence_Adversrial_Autoencoder_with_New_Loss

dataset_obj = Emilya_Dataset(window_width=200,shift_step=20,sampling_interval=None,with_velocity=False)

model = Sequence_Adversrial_Autoencoder_with_New_Loss(latent_dim=50,
                                        latent_activation='tanh',
                                        latent_BN=False,
                                        hidden_dim_enc_list=[100,100],
                                        activation_enc_list=['tanh','tanh'],
                                        hidden_dim_dec_list=None,
                                        activation_dec_list=None,
                                        hidden_dim_dis_list=[100,40],
                                        activation_dis_list=['relu','relu'],
                                        dropout_dis_list=[0.0,0.0],
                                        batch_size=200,
                                        max_epoch=401,
                                        optimiser_autoencoder='rmsprop',
                                        optimiser_dis='Adam',
                                        lr_autoencoder=0.001, lr_dis=0.001,
                                        decay_autoencoder=0.0, decay_dis=0.0,
                                        momentum_autoencoder=0.0, momentum_dis=0.0,
                                        prior_noise_type='Gaussian',
                                        loss_weights=[1.0,0.001],
                                        train_disc=True,
                                        checkpoint_epochs=100,
                                        custom_loss=False,
                                        loss_weight_mse_v=1.0)
model.training(dataset_obj)


