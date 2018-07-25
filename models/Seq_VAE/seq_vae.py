##This is a model of Sequential adversarial Autoencoder combining a Posture AAE model.

#This is an original model, so we will remove some unnesscessary parts from the SAAE, such as
#the context input,label input etc. We only use the simplest version of SAAE
from sklearn.base import BaseEstimator
import numpy as np
import sys
print "recursionlimit is {}".format(sys.getrecursionlimit())
from keras.layers import Input,LSTM,RepeatVector,Dense,SimpleRNN,GRU
from keras.models import Model
from keras.optimizers import Adam,RMSprop,SGD,Nadam,Adadelta,Adamax
from keras.layers import TimeDistributed,merge,BatchNormalization,Dropout
from keras.models import model_from_yaml
from keras.regularizers import l2,l1
from Seq_AAE_V1.callbacks.custom_initialiser import custom_wq_glorot_uniform,Scalar_Init
from keras.layers.core import Lambda
#from prior_distribution import mixture_gaussian_sampling
import keras.backend as K
from keras.utils.vis_utils import plot_model
from keras import objectives
import matplotlib
import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap,LogNorm,PowerNorm
from Seq_AAE_V1.datasets.dataset import generate_positive_samples
from Seq_AAE_V1.synthesis_scripts.synthesis_utils import save_seq_2_bvh
from Seq_AAE_V1.custom_layers.custom_layers import Scaling

class Sequence_VAE(BaseEstimator):
    def __init__(self, latent_dim=30,latent_activation='tanh', latent_BN=False,
                 hidden_dim_enc_list=[100, 100], activation_enc_list=['tanh', 'tanh'],
                 hidden_dim_dec_list=None, activation_dec_list=None,
                 output_type = 'dense',
                 batch_size=300, max_epoch=200,
                 optimiser_autoencoder='rmsprop',
                 lr_autoencoder=0.001,
                 decay_autoencoder=0.0,
                 momentum_autoencoder=0.0,
                 prior_noise_type='Gaussian',
                 nb_to_generate = 20,
                 data_obj = None,epoch_checkpoint=100,
                 lambda_reguler=0.,
                 regulariser = 'null',
                 default_hyper_parameter=True,epsilon_std=1.0,
                 alpha_rate=0.,begin_tune_alpha_epoch=0,alpha_bias=5.0,
                 recurrent_dropout_outputlayer=0.0,
                 fixed_alpha=None):

        args = locals().copy()
        del args['self']
        self.__dict__.update(args)
        print args.keys()
        self.save_configuration(args)
        if not data_obj is None:
            self.set_up_dataset(self.data_obj)

    def set_up_model_as_none(self):
        self.encoder = None
        self.decoder = None

    def set_up_model(self):
        #load pretrained models
        if self.fixed_alpha is not None:
            self.alpha = K.variable(self.fixed_alpha,name='alpha')
        else:
            self.alpha = K.variable(0., name='alpha')
        if self.hidden_dim_dec_list is None:
            self.hidden_dim_dec_list = self.hidden_dim_enc_list[::-1]
        if self.activation_dec_list is None:
            self.activation_dec_list = self.activation_enc_list[::-1]

        self.encoder = self.Encoder()
        self.decoder = self.Decoder()
        self.svae = self.SVAE()
        self.compile()
        self.visualize_model(self.encoder)
        self.visualize_model(self.decoder)
        self.visualize_model(self.svae)

    def visualize_model(self, model):
        model.summary()
        plot_model(model,
                   to_file='%s.png' % model.name,
                   show_shapes=True,
                   show_layer_names=True)


    def save_configuration(self, arguments):
        with open('meta_data.txt', 'w') as file:
            file.writelines("========Meta Data========\r\n")
            for key in arguments.keys():
                file.writelines(key + ' : ' + str(arguments[key]) + '\r\n')
            file.writelines('===========END===========\r\n')

    def set_up_dataset(self, dataset_obj):
        # save datasource to meta_data.txt
        with open('meta_data.txt', 'a') as file:
            file.writelines(dataset_obj.file_source)

        if hasattr(dataset_obj, 'max_len'):
            self.max_len = dataset_obj.max_len
        else:
            raise ValueError("Attribute 'max_len' doesn't exist in dataset obj ")

        if hasattr(dataset_obj, 'dof'):
            self.dof = dataset_obj.dof-1
        else:
            raise ValueError("Attribute 'dof' doesn't exist in dataset obj ")

        self.train_X = dataset_obj.train_X[:,:,1:]
        self.train_Y1 = dataset_obj.train_Y1
        self.train_Y2 = dataset_obj.train_Y2
        self.train_Y3 = dataset_obj.train_Y3
        #self.train_speed = dataset_obj.train_speed
        print "training set size: {}".format(self.train_X.shape)

        self.valid_X = dataset_obj.valid_X[:,:,1:]
        self.valid_Y1 = dataset_obj.valid_Y1
        self.valid_Y2 = dataset_obj.valid_Y2
        self.valid_Y3 = dataset_obj.valid_Y3
        #self.valid_speed = dataset_obj.valid_speed

        self.test_X = dataset_obj.test_X[:,:,1:]
        self.test_Y1 = dataset_obj.test_Y1
        self.test_Y2 = dataset_obj.test_Y2
        self.test_Y3 = dataset_obj.test_Y3
        #self.test_speed = dataset_obj.test_speed
        self.max_vector = dataset_obj.max_vector[1:]
        self.min_vector = dataset_obj.min_vector[1:]

        self.postprocess = dataset_obj.postprocess
        self.sampling_interval = dataset_obj.sampling_interval
        #set up noise vector to generate seqeunces

        if self.prior_noise_type == 'Gaussian':
            mean = np.zeros(self.latent_dim)
            covariance = np.eye(N=self.latent_dim) * 1.0
            self.noise_vectors = generate_positive_samples(self.nb_to_generate,
                                                           mean, covariance,
                                                           'Gaussian',seed=1234)

        else:
            raise NotImplementedError()

    def set_up_noise_examples(self):

        if self.prior_noise_type == 'Gaussian':
            mean = np.zeros(self.latent_dim)
            covariance = np.eye(N=self.latent_dim) * 1.0
            self.positive_examples_training = generate_positive_samples(len(self.train_X), mean, covariance, 'Gaussian',
                                                                        seed=None)
            self.positive_examples_valid = generate_positive_samples(len(self.valid_X), mean, covariance, 'Gaussian',
                                                                     seed=np.random.randint(0, 2000))
            self.positive_examples_test = generate_positive_samples(len(self.test_X), mean, covariance, 'Gaussian',
                                                                    seed=np.random.randint(0, 2000))

        else:
            raise NotImplementedError()


    def Encoder(self):
        self.motion_input = Input(shape=(self.max_len, self.dof), name='encoder_input')
        encoded = self.motion_input
        for i, (dim, activation) in enumerate(zip(self.hidden_dim_enc_list, self.activation_enc_list)):
            if self.regulariser =='l2':
                encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True,kernel_regularizer=l2(self.lambda_reguler))(encoded)
            elif self.regulariser == 'l1':
                encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True,
                               kernel_regularizer=l1(self.lambda_reguler))(encoded)
            else:
                encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(encoded)

        ##output of encoder
        encoded = LSTM(output_dim=self.latent_dim, activation=self.latent_activation, name='encoded_layer',
                       return_sequences=False)(encoded)
        self.z_mean = Dense(output_dim=self.latent_dim, activation='linear', name='z_mean')(encoded)
        self.z_log_sigma = Dense(output_dim=self.latent_dim, activation='linear', name='z_log_sigma')(encoded)
        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,))(
            [self.z_mean, self.z_log_sigma])
        return Model(inputs=self.motion_input, outputs=self.z, name='Encoder')

    def Decoder(self):
        latent_input = Input(shape=(self.latent_dim,), name='latent_input')
        decoded = RepeatVector(self.max_len)(latent_input)
        for i, (dim, activation) in enumerate(zip(self.hidden_dim_dec_list, self.activation_dec_list)):
            decoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(decoded)

        if self.output_type=='dense':
            decoded_mean = TimeDistributed(Dense(units=self.dof, activation='sigmoid'),
                                           name='decoded_mean')(decoded)
            decoded_log_sigma = TimeDistributed(Dense(units=self.dof, activation='linear'),
                                                name='decoded_log_sigma')(decoded)
        elif self.output_type =='rnn':
            decoded_mean = SimpleRNN(units=self.dof, activation='sigmoid',
                                     recurrent_dropout=self.recurrent_dropout_outputlayer,
                                     return_sequences=True,
                                     name='decoded_mean')(decoded)
            decoded_log_sigma = SimpleRNN(units=self.dof, activation='linear',
                                     recurrent_dropout=self.recurrent_dropout_outputlayer,
                                          return_sequences=True,
                                          name='decoded_log_sigma')(decoded)
        elif self.output_type =='lstm':
            decoded_mean = LSTM(units=self.dof, activation='sigmoid',
                                recurrent_dropout=self.recurrent_dropout_outputlayer,
                                return_sequences=True,
                                name='decoded_mean')(decoded)
            decoded_log_sigma = LSTM(units=self.dof,activation='linear',
                                     recurrent_dropout=self.recurrent_dropout_outputlayer,
                                     return_sequences=True,
                                     name='decoded_log_sigma')(decoded)

        return Model(inputs=latent_input, outputs=[decoded_mean,decoded_log_sigma], name='Decoder')

    def SVAE(self):
        self.svae_mean,self.svae_log_sigma = self.decoder(self.z)
        return Model(inputs=self.motion_input,outputs=[self.svae_mean,self.svae_log_sigma],name='SVAE')

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], K.shape(z_mean)[-1]),
                                  mean=0., stddev=self.epsilon_std)
        ##modifide by wangqi
        return z_mean + K.exp(z_log_sigma) * epsilon

    def compile(self):
        if self.optimiser_autoencoder == 'sgd':
            optimizer_ae = SGD(lr=self.lr_autoencoder,
                               decay=self.decay_autoencoder,
                               momentum=self.momentum_autoencoder)
        elif self.optimiser_autoencoder == 'rmsprop':
            optimizer_ae = RMSprop(lr=self.lr_autoencoder,decay=self.decay_autoencoder)
        elif self.optimiser_autoencoder == 'adam':
            optimizer_ae = Adam(lr=self.lr_autoencoder,decay=self.decay_autoencoder)
        elif self.optimiser_autoencoder == 'nadam':
            if self.default_hyper_parameter == True:
                optimizer_ae = Nadam()
            else:
                optimizer_ae = Nadam(lr=self.lr_autoencoder,schedule_decay=self.decay_autoencoder)
        elif self.optimiser_autoencoder == 'adadelta':
            #recommend leave all the parameters at default values
            if self.default_hyper_parameter == True:
                optimizer_ae = Adadelta()
            else:
                optimizer_ae = Adadelta(lr=self.lr_autoencoder, decay=self.decay_autoencoder)

        self.encoder.compile(optimizer='sgd',loss='mse')
        self.decoder.compile(optimizer='sgd',loss='mse')
        self.svae.compile(optimizer=optimizer_ae,loss=[self.loss_svae,None],
                          metrics=['mse',self.neg_log_likelihood,self.KL_divergence])

    def KL_divergence(self, y_true, y_pred):
        kl_loss = - 0.5 * K.sum(1 + 2 * self.z_log_sigma - K.square(self.z_mean) - K.exp(2 * self.z_log_sigma), axis=-1)
        return kl_loss

    def neg_log_likelihood(self,y_true,y_pred):
        pxzx = 1. * K.sum(K.sum((-(0.5 * np.log(2 * np.pi) + 0.5 * self.svae_log_sigma) -
                                 0.5 * ((y_true - y_pred) ** 2 / K.exp(self.svae_log_sigma))), axis=-1), axis=-1)
        return -1.*pxzx

    def loss_svae(self,y_true, y_pred):
        pxzx = 1. * K.sum(K.sum((-(0.5 * np.log(2 * np.pi) + 0.5 * self.svae_log_sigma) -
                      0.5 * ((y_true - y_pred)**2 / K.exp(self.svae_log_sigma))),axis=-1),axis=-1)
        kl_loss = - 0.5 * K.sum(1 + 2*self.z_log_sigma - K.square(self.z_mean) - K.exp(2*self.z_log_sigma),axis=-1)
        total_loss = -1.* pxzx + self.alpha*kl_loss
        return total_loss

    def init_loss_history_list(self):
        self.loss_history = {"training_total_loss": [],
                             "training_metric_mse": [],
                             "training_metric_neg_log_px":[],
                             "training_metric_kld": [],
                             "valid_total_loss": [],
                             "valid_metric_mse": [],
                             "valid_metric_neg_log_px": [],
                             "valid_metric_kld": [],
                             "alpha": [],
                             }
    def print_loss_history(self):
        for loss_key in sorted(self.loss_history.keys()):
            if len(self.loss_history[loss_key])==0:
                continue
            print "%s:%f" % (loss_key, self.loss_history[loss_key][-1])

    def plot_latent_space(self, latent_codes, filename, Y1=None, Y2=None, dim_x=0, dim_y=1):
        if Y1 is None:
            Y1 = np.zeros(len(latent_codes), dtype=np.float32)
        if Y2 is None:
            Y2 = np.zeros(Y1.shape, dtype=np.float32)
        nb_act = len(np.unique(Y1))
        act_unique = np.unique(Y1)
        nb_em = len(np.unique(Y2))
        fig = plt.figure(figsize=(8, 8))
        color = plt.cm.rainbow(np.linspace(0, 1, nb_act * nb_em))
        # print np.unique(Y)
        for l, c in zip(range(nb_act * nb_em), color):
            y1 = l / nb_em
            y2 = l % nb_em
            idx_1 = np.where(Y1 == act_unique[y1])[0]
            idx_2 = np.where(Y2 == y2)[0]
            idx = list(set(idx_1).intersection(idx_2))
            plt.scatter(latent_codes[idx, dim_x], latent_codes[idx, dim_y], c=c, label=l, s=8, linewidths=0)

        plt.legend(fontsize=15)
        plt.savefig(filename)
        plt.close(fig)

    def plot_2d_histogram_latent_codes(self,latent_codes, bins=None, suffix=''):
        if isinstance(latent_codes, list):
            latent_codes = np.asarray(latent_codes)
        if len(latent_codes.shape) ==3:
            latent_codes = latent_codes.reshape((latent_codes.shape[0]*latent_codes.shape[1],
                                                 latent_codes.shape[2]))
        dim_latent = latent_codes.shape[-1]
        nb_plots = dim_latent / 2
        if dim_latent % 2 == 1:
            nb_plots = nb_plots + 1
        cmap = matplotlib.cm.jet
        fig = plt.figure(figsize=(5, 4 * nb_plots))
        if bins is None:
            bins = np.arange(-5, 5, step=0.2)

        for i in range(nb_plots):
            plt.subplot(nb_plots, 1, i + 1)
            if i != nb_plots - 1:
                plt.hist2d(latent_codes[:, i * 2], latent_codes[:, i * 2 + 1], bins=bins, norm=LogNorm(), cmap=cmap)
            else:
                if i * 2 + 1 == dim_latent:
                    plt.hist2d(latent_codes[:, i * 2], latent_codes[:, i * 2 + 1], bins=bins, norm=LogNorm(), cmap=cmap)
                else:
                    plt.hist2d(latent_codes[:, i * 2 - 1], latent_codes[:, i * 2], bins=bins, norm=LogNorm(), cmap=cmap)
            plt.colorbar()

        plt.savefig('./hist2d_latent_dim' + suffix + '.png')
        plt.close(fig)

    def plot_1d_histogram_latent_codes(self,latent_codes,bins = None,suffix=''):
        if isinstance(latent_codes, list):
            latent_codes = np.asarray(latent_codes)
        latent_codes = latent_codes.flatten()
        if bins is None:
            bins = np.arange(-5, 5, step=1)
        fig = plt.figure(figsize=(5,5))
        plt.hist(latent_codes,bins=bins,normed=True)
        plt.savefig('./hist1d_latent_dim' + suffix + '.png')
        plt.close(fig)

    def batch_generator(self,iterable, batch_size=1,shuffle=False):
        l = len(iterable)
        if shuffle ==True:
            indices = np.random.permutation(len(iterable))
        else:
            indices = np.arange(0,stop=len(iterable))
        for ndx in range(0,l,batch_size):
            cur_indices = indices[ndx:min(ndx+batch_size,l)]
            yield  iterable[cur_indices]

    def training(self,dataset_obj):
        self.set_up_dataset(dataset_obj=dataset_obj)
        self.set_up_model()
        self.init_loss_history_list()
        iterations = 0
        batch_num = self.train_X.shape[0]/self.batch_size +1
        for epoch in range(self.max_epoch):
            print("\r\nEpoch seen: {}").format(epoch)
            self.cur_lr_svae= self.svae.optimizer.lr * (1. / (
                1 + self.svae.optimizer.decay * self.svae.optimizer.iterations))
            print "current learning rate of autoencoder: {}".format(self.cur_lr_svae.eval())
            print "alpha = {}".format(self.alpha.eval())
            for train_batch in self.batch_generator(self.train_X,batch_size=self.batch_size):
                iterations += 1
                if self.fixed_alpha is None:
                    if iterations >= self.begin_tune_alpha_epoch*batch_num:
                        #print "Update the value of alpha"
                        #new_alpha_value = K.minimum(self.alpha + epoch* self.alpha_rate, 1.).eval()
                        new_alpha_value = K.sigmoid(self.alpha_rate * (
                            (iterations - self.begin_tune_alpha_epoch*batch_num)/self.batch_size - (self.alpha_bias / self.alpha_rate))).eval()
                        K.set_value(self.alpha, new_alpha_value)

                self.svae.train_on_batch(x=train_batch,y=train_batch)
            if self.fixed_alpha is None:
                self.loss_history['alpha'].append(new_alpha_value)

            self.compute_loss_and_plot_latent_space(epoch,whichdataset='training')
            self.compute_loss_and_plot_latent_space(epoch, whichdataset='valid')
            self.print_loss_history()
            self.plot_loss()
            #save checkpoints
            if epoch%self.epoch_checkpoint ==0:
                random_generated_motion, random_generated_sigma = self.decoder.predict(x=self.noise_vectors)
                reconstruted_motion, reconstructed_sigm= self.svae.predict(x = self.test_X[:self.nb_to_generate])
                self.save_generated_seqs(random_generated_motion,max_vector=self.max_vector,
                                         min_vector=self.min_vector,suffix='random_generated',epoch=epoch)
                if epoch==0:
                    self.save_generated_seqs(self.test_X[:self.nb_to_generate],max_vector=self.max_vector,
                                             min_vector=self.min_vector,suffix='test_motion')
                self.save_generated_seqs(reconstruted_motion,max_vector=self.max_vector,
                                         min_vector=self.min_vector,suffix='reconstruted_test',epoch=epoch)

                self.save_models(epoch=epoch)
        np.savez('saved_loss_history.npz',self.loss_history)

    def fit(self,X,y=None):
        ##set up model
        # self.set_up_dataset(self.data_obj)
        # self.set_up_model()
        # self.svae.fit(x=self.train_X,y=self.train_X,batch_size=self.batch_size,
        #               epochs=self.max_epoch,validation_data=[self.valid_X,self.valid_X],verbose=0,callbacks=)
        self.set_up_dataset(self.data_obj)
        self.set_up_model()
        iterations = 0
        batch_num = self.train_X.shape[0] / self.batch_size + 1
        for epoch in range(self.max_epoch):
            print("\r\nEpoch seen: {}").format(epoch)
            if self.optimiser_autoencoder !='nadam':
                self.cur_lr_svae = self.svae.optimizer.lr * (1. / (
                    1 + self.svae.optimizer.decay * self.svae.optimizer.iterations))
                print "current learning rate of autoencoder: {}".format(self.cur_lr_svae.eval())
            print "alpha = {}".format(self.alpha.eval())
            for train_batch in self.batch_generator(self.train_X, batch_size=self.batch_size):
                iterations += 1
                if self.fixed_alpha is None:
                    if iterations >= self.begin_tune_alpha_epoch * batch_num:
                        # print "Update the value of alpha"
                        # new_alpha_value = K.minimum(self.alpha + epoch* self.alpha_rate, 1.).eval()
                        new_alpha_value = K.sigmoid(self.alpha_rate * (
                            (iterations - self.begin_tune_alpha_epoch * batch_num) / self.batch_size - (
                            self.alpha_bias / self.alpha_rate))).eval()
                        K.set_value(self.alpha, new_alpha_value)
                self.svae.train_on_batch(x=train_batch, y=train_batch)

        return self

    def score(self,X,y=None):
        totalloss,loss, mse,neg_log_px, kld = self.svae.evaluate(x=self.valid_X,y=self.valid_X,batch_size=200,verbose=0)
        return -1.*loss


    def compute_loss_and_plot_latent_space(self, epoch, whichdataset='training'):
        if whichdataset == 'training':
            dataset = self.train_X
            Y1 = self.train_Y1
            Y2 = self.train_Y2
        elif whichdataset == 'valid':
            dataset = self.valid_X
            Y1 = self.valid_Y1
            Y2 = self.valid_Y2
        elif whichdataset == 'test':
            dataset = self.test_X
            Y1 = self.test_Y1
            Y2 = self.test_Y
        else:
            raise ValueError('wrong argument whichdataset')
        num_samples = len(dataset)
        latent_codes = self.encoder.predict(x=dataset, batch_size=1000, verbose=0)
        # plot latent space
        if epoch % 30 == 0:
            filename = 'Epoch' + str(epoch) + '_' + whichdataset + '01.png'
            self.plot_latent_space(latent_codes, filename, Y1, dim_x=0, dim_y=1)
            filename = 'Epoch' + str(epoch) + '_' + whichdataset + '23.png'
            self.plot_latent_space(latent_codes, filename, Y1, dim_x=2, dim_y=3)
            # plot histogram of latent space#
            self.plot_2d_histogram_latent_codes(latent_codes,suffix='_epoch'+str(epoch))

        total_loss,loss, mse,neg_log_px,kld = self.svae.evaluate(x=dataset,y=dataset,verbose=0)
        assert total_loss==loss
        if whichdataset == 'training':
            self.loss_history["training_total_loss"].append(loss)
            self.loss_history["training_metric_mse"].append(mse)
            self.loss_history["training_metric_neg_log_px"].append(neg_log_px)
            self.loss_history["training_metric_kld"].append(kld)

        elif whichdataset == 'valid':
            self.loss_history["valid_total_loss"].append(loss)
            self.loss_history["valid_metric_mse"].append(mse)
            self.loss_history["valid_metric_neg_log_px"].append(neg_log_px)
            self.loss_history["valid_metric_kld"].append(kld)

        elif whichdataset == 'test':
            raise NotImplementedError()

    def plot_loss(self):
        # plot mse
        fig = plt.figure(figsize=(5, 5*4))
        legend_str = []
        plt.subplot(4,1,1)
        plt.plot(self.loss_history["training_total_loss"])
        legend_str.append('training_total_loss' + ':%f' % self.loss_history["training_total_loss"][-1])
        plt.plot(self.loss_history["valid_total_loss"])
        legend_str.append('valid_total_loss' + ':%f' % self.loss_history["valid_total_loss"][-1])
        plt.legend(legend_str)

        plt.subplot(4, 1, 2)
        legend_str = []
        plt.plot(self.loss_history['training_metric_mse'])
        legend_str.append('training_metric_mse:%f' % self.loss_history['training_metric_mse'][-1])
        plt.plot(self.loss_history['valid_metric_mse'])
        legend_str.append(
            'valid_metric_mse:%f' % self.loss_history['valid_metric_mse'][-1])
        plt.legend(legend_str, fontsize=10)
        # plot loss of discriminator
        plt.subplot(4, 1, 3)
        legend_str = []
        plt.plot(self.loss_history['training_metric_neg_log_px'])
        legend_str.append('training_metric_neg_log_px:%f' % self.loss_history['training_metric_neg_log_px'][-1])
        plt.plot(self.loss_history['valid_metric_neg_log_px'])
        legend_str.append(
            'valid_metric_neg_log_px:%f' % self.loss_history['valid_metric_neg_log_px'][-1])
        plt.legend(legend_str, fontsize=10)

        plt.subplot(4, 1, 4)
        legend_str = []
        plt.plot(self.loss_history['training_metric_kld'])
        legend_str.append('training_metric_kld:%f' % self.loss_history['training_metric_kld'][-1])
        plt.plot(self.loss_history['valid_metric_kld'])
        legend_str.append(
            'valid_metric_kld:%f' % self.loss_history['valid_metric_kld'][-1])
        if len(self.loss_history['alpha'])>0:
            plt.plot(self.loss_history['alpha'])
            legend_str.append(
                'alpha:%f' % self.loss_history['alpha'][-1])
        plt.legend(legend_str, fontsize=10)
        plt.tight_layout()
        plt.savefig('./learning_curve.png')
        plt.close(fig)

    from mpl_toolkits.mplot3d import Axes3D

    def save_generated_seqs(self,seqs,max_vector,min_vector,suffix = '',epoch=None):
        sequences = self.postprocess(seqs,max_vector,min_vector)
        for i, seq in enumerate(sequences):
            if epoch is None:
                filename = './generated_seq' + '_num_' + str(i) + suffix + '.bvh'
            else:
                filename = './generated_seq_epoch_' + str(epoch) + '_num_' + str(i) + suffix + '.bvh'
            save_seq_2_bvh(seq, sequence_name=filename, step=self.sampling_interval)

    def save_models(self,epoch =None):
        if epoch is None:
            suffix_str=''
        else:
            suffix_str = '_epoch'+str(epoch)

        self.encoder.save_weights('encoder'+suffix_str+'.h5')
        with open('encoder'+suffix_str+'.yaml', 'w') as yaml_file:
            yaml_file.write(self.encoder.to_yaml())

        self.decoder.save_weights('decoder'+suffix_str+'.h5')
        with open('decoder'+suffix_str+'.yaml', 'w') as yaml_file:
            yaml_file.write(self.decoder.to_yaml())

        self.svae.save_weights('svae'+suffix_str+'.h5')
        with open('svae'+suffix_str+'.yaml', 'w') as yaml_file:
            yaml_file.write(self.svae.to_yaml())



if __name__=='__main__':
    import sys
    import os
    root_path = os.getenv('Seq_AAE_V1')
    sys.path.append(root_path)
    from Seq_AAE_V1.datasets.dataset import Emilya_Dataset
    data_obj = Emilya_Dataset(window_width=200, shift_step=20, sampling_interval=None, with_velocity=False,
                              number=200,nb_test=200)

    model = Sequence_VAE(data_obj=data_obj,batch_size=32,
                                        hidden_dim_enc_list=[100],
                                        hidden_dim_dec_list=[100],
                                        nb_to_generate=50,alpha_rate=1e-2,
                         lr_autoencoder=1e-4,fixed_alpha=1.0)

    model.training(data_obj)