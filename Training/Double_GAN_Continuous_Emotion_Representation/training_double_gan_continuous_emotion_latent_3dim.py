import sys
import os
root_path = os.getenv('Seq_AAE_V1')
sys.path.append(root_path)
from Seq_AAE_V1.datasets.dataset import Emilya_Dataset
from Seq_AAE_V1.models.Double_GAN_Continuous_Emotion_Representation.double_gan_continuous_emotion_latent import Double_GAN_Continuous_Emotion_Latent

dataset_obj = Emilya_Dataset(window_width=200,shift_step=20,
                             sampling_interval=None,with_velocity=False,
                             number=None,nb_valid=None,nb_test=None)
model = Double_GAN_Continuous_Emotion_Latent(
                                latent_dim=50,aux_latent_dim=3,
                                latent_activation='tanh',latent_BN=False,
                                hidden_dim_enc_list=[100,100],activation_enc_list=['tanh','tanh'],
                                hidden_dim_dec_list=None,activation_dec_list=None,
                                hidden_dim_dis_list=[100,40],activation_dis_list=['relu','relu'],
                                hidden_dim_classifier1_list=[100,40],activation_classifier1_list=['relu','relu'],
                                hidden_dim_classifier2_list=[100, 40],activation_classifier2_list=['relu', 'relu'],
                                dropout_dis_list=[0.0,0.0],
                                dropout_classifier1_list=[0.0,0.0],
                                dropout_classifier2_list=[0.0,0.0],
                                batch_size=200,max_epoch=401,
                                optimiser_autoencoder='rmsprop',optimiser_dis='adam',
                                optimiser_classifier1='adam',optimiser_classifier2='adam',
                                lr_autoencoder=0.001,lr_dis=0.001,decay_autoencoder=0.0,decay_dis=0.0,
                                lr_classifier1=0.001,lr_classifier2=0.001,
                                decay_classifier1=0.0,decay_classifier2=0.0,
                                momentum_autoencoder=0.0,momentum_dis=0.0,
                                momentum_classifier1=0.0,momentum_classifier2=0.0,
                                prior_noise_type='Gaussian',
                                loss_weights=[1.0,0.001,0.0005,0.01],
                                train_disc=True,
                                train_classifier1=True,
                                custom_loss=True,
                                loss_weight_mse_v=100,
                                symetric_autoencoder=False,
                                checkpoint_epochs=30,
                                nb_to_generate=10,
                                condition_activity_or_emotion=2,
                                nb_label=8,fully_condition=False,
                                is_annealing_beta=True,
                                beta_anneal_rate=0.1,
                                bias_beta=9.)
model.training(dataset_obj)


