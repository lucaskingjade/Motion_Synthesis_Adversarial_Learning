'''
Sequential Autoencoder using the velocity along xz plane as contextual variables
'''
__author__ = 'qiwang'
import numpy as np
from keras.layers import Input,LSTM,RepeatVector,Dense,SimpleRNN,GRU
from keras.models import Model
from keras.optimizers import Adam,RMSprop,SGD
from keras.layers import TimeDistributed,merge,BatchNormalization,Dropout
#from prior_distribution import mixture_gaussian_sampling
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Seq_AAE_V1.datasets.dataset import generate_positive_samples
from sklearn.base import BaseEstimator
import keras.backend as K
from Seq_AAE_V1.synthesis_scripts.synthesis_utils import save_seq_2_bvh
from matplotlib.colors import Colormap,LogNorm,PowerNorm

class Sequence_Adversrial_Autoencoder_with_New_Loss(BaseEstimator):

    def __init__(self,latent_dim=50,latent_activation='tanh',latent_BN=False,
                 hidden_dim_enc_list=[100,100],activation_enc_list=['tanh','tanh'],
                 hidden_dim_dec_list=None,activation_dec_list=None,
                 hidden_dim_dis_list=[100,40],activation_dis_list=['relu','relu'],
                 dropout_dis_list=[0.0,0.0],
                 batch_size=200,max_epoch=200,
                 optimiser_autoencoder='rmsprop',optimiser_dis='sgd',
                 lr_autoencoder=0.001,lr_dis=0.01,
                 decay_autoencoder=0.0,decay_dis=0.0,
                 momentum_autoencoder=0.0,momentum_dis=0.0,
                 prior_noise_type='Gaussian',
                 loss_weights=[1.0,0.0],train_disc=True,
                 custom_loss=False,
                 loss_weight_mse_v= 1.0,
                 checkpoint_epochs=100,symetric_autoencoder=False,
                 nb_to_generate=30):

        args = locals().copy()
        del args['self']
        self.__dict__.update(args)
        print args.keys()
        self.save_configuration(args)

    def set_up_model(self):
        if self.hidden_dim_dec_list is None:
            self.hidden_dim_dec_list = self.hidden_dim_enc_list[::-1]
        if self.activation_dec_list is None:
            self.activation_dec_list = self.activation_enc_list[::-1]

        self.alpha = K.variable(self.loss_weight_mse_v, name='alpha')
        if self.symetric_autoencoder==True:
            self.encoder = self.Encoder_symetric()
        else:
            self.encoder = self.Encoder()
        self.decoder = self.Decoder()
        self.discriminator = self.Discriminator()
        self.autoencoder_with_discriminator = self.Autoencoder_with_Discmt()
        self.compile()
        #summary of model
        self.encoder.summary()
        self.decoder.summary()
        self.autoencoder_with_discriminator.summary()
        self.discriminator.summary()


    def save_configuration(self,arguments):
        with open('meta_data.txt','w') as file:
            file.writelines("========Meta Data========\r\n")
            for key in arguments.keys():
                file.writelines(key+' : '+ str(arguments[key])+'\r\n')
            file.writelines('===========END===========\r\n')


    def set_up_dataset(self,dataset_obj):
        #save datasource to meta_data.txt
        with open('meta_data.txt','a') as file:
            file.writelines(dataset_obj.file_source)

        if hasattr(dataset_obj,'max_len'):
            self.max_len = dataset_obj.max_len
        else:
            raise ValueError("Attribute 'max_len' doesn't exist in dataset obj ")

        if hasattr(dataset_obj,'dof'):
            self.dof = dataset_obj.dof-1
        else:
            raise ValueError("Attribute 'dof' doesn't exist in dataset obj ")

        self.train_X = dataset_obj.train_X[:,:,1:]
        self.train_Y1 = dataset_obj.train_Y1
        self.train_Y2 = dataset_obj.train_Y2
        self.train_Y3 = dataset_obj.train_Y3
        #self.train_speed = dataset_obj.train_speed

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
        # set up noise vector to generate seqeunces

        if self.prior_noise_type == 'Gaussian':
            mean = np.zeros(self.latent_dim)
            covariance = np.eye(N=self.latent_dim) * 1.0
            self.noise_vectors = generate_positive_samples(self.nb_to_generate,
                                                           mean, covariance,
                                                           'Gaussian', seed=1234)

        else:
            raise NotImplementedError()


    def set_up_noise_examples(self):

        if self.prior_noise_type=='Gaussian':
            mean = np.zeros(self.latent_dim)
            covariance = np.eye(N=self.latent_dim) * 1.0
            self.positive_examples_training = generate_positive_samples(len(self.train_X),mean,covariance,'Gaussian',seed=np.random.randint(0,2000))
            self.positive_examples_valid = generate_positive_samples(len(self.valid_X), mean, covariance, 'Gaussian',seed=np.random.randint(0,2000))
            self.positive_examples_test = generate_positive_samples(len(self.test_X), mean, covariance, 'Gaussian',seed=np.random.randint(0,2000))

        else:
            raise NotImplementedError()

    def Encoder_symetric(self):
        motion_input = Input(shape=(self.max_len, self.dof), name='encoder_input')
        encoded = motion_input
        for i, (dim, activation) in enumerate(zip(self.hidden_dim_enc_list, self.activation_enc_list)):
            if i <len(self.hidden_dim_enc_list)-1:
                encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(encoded)
            else:
                encoded = LSTM(output_dim=dim, activation=activation, return_sequences=False)(encoded)

        encoded = Dense(output_dim=self.latent_dim, activation='linear')(encoded)

        return Model(input=motion_input, output=encoded, name='Encoder')

    def Encoder(self):
        motion_input = Input(shape = (self.max_len,self.dof),name='encoder_input')
        encoded = motion_input
        for i, (dim, activation) in enumerate(zip(self.hidden_dim_enc_list, self.activation_enc_list)):
                encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(encoded)
        if self.latent_BN==True:
            encoded = LSTM(output_dim=self.latent_dim, activation=self.latent_activation, name='encoded_layer',
                           return_sequences=False)(encoded)
            encoded = Dense(output_dim=self.latent_dim,activation='linear')(encoded)
            encoded = BatchNormalization(name='latent_BN')(encoded)
        else:
            encoded = LSTM(output_dim=self.latent_dim,activation=self.latent_activation,name = 'encoded_layer',return_sequences=False)(encoded)
            encoded = Dense(output_dim=self.latent_dim, activation='linear')(encoded)

        return Model(input=motion_input, output = encoded, name='Encoder')


    def Decoder(self):
        latent_input = Input(shape=(self.latent_dim,),name='latent_input')
        latent_input_seq = RepeatVector(self.max_len)(latent_input)
        decoded = latent_input_seq
        for i,(dim, activation) in enumerate(zip(self.hidden_dim_dec_list,self.activation_dec_list)):
            decoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(decoded)

        decoded = SimpleRNN(output_dim=self.dof,activation='sigmoid',name='decoder_output',return_sequences=True)(decoded)
        return Model(input = latent_input, output=decoded,name='Decoder')

    def Discriminator(self):
        input = Input(shape=(self.latent_dim,),name='discmt_input')
        for i,(dim,activation,dropout) in enumerate(zip(self.hidden_dim_dis_list, self.activation_dis_list,self.dropout_dis_list)):
            if i ==0:
                discmt = Dense(dim,activation=activation)(input)
            else:
                discmt = Dropout(dropout)(discmt)
                discmt = Dense(dim,activation=activation)(discmt)

        discmt = Dense(1,activation='sigmoid',name='discmt_output')(discmt)
        return Model(input=input,output=discmt,name='Discmt')


    def Autoencoder_with_Discmt(self):
        autoencoder = self.decoder(self.encoder.output)
        self.discriminator.trainable = False
        aux_output_discmt = self.discriminator(self.encoder.output)
        return Model(input=self.encoder.input, output=[autoencoder, aux_output_discmt], name='Autoencoder_with_Dis')


    def compile(self):
        if self.optimiser_autoencoder =='sgd':
            optimizer_ae = SGD(lr=self.lr_autoencoder,decay=self.decay_autoencoder)
        elif self.optimiser_autoencoder =='rmsprop':
            optimizer_ae = RMSprop(lr=self.lr_autoencoder,decay=self.decay_autoencoder)
        elif self.optimiser_autoencoder =='adam':
            optimizer_ae = Adam(lr=self.lr_autoencoder,decay=self.decay_autoencoder)

        if self.optimiser_dis =='sgd':
            optimizer_discmt = SGD(lr=self.lr_dis,decay=self.decay_dis)
        elif self.optimiser_dis =='Adam':
            optimizer_discmt = Adam(lr=self.lr_dis,decay=self.decay_dis)

        self.discriminator.trainable = True
        self.discriminator.compile(optimizer_discmt,loss='binary_crossentropy',metrics=['accuracy'])
        self.discriminator.trainable = False

        if self.custom_loss==True:
            self.autoencoder_with_discriminator.compile(optimizer_ae, \
                                                        loss={'Decoder': self.loss_mse_velocity_loss, \
                                                              'Discmt': 'binary_crossentropy'}, \
                                                        loss_weights=self.loss_weights,
                                                        metrics={'Decoder':'mse'})

        else:
            self.autoencoder_with_discriminator.compile(optimizer_ae,\
                                                        loss={'Decoder':'mse',\
                                                               'Discmt':'binary_crossentropy'},\
                                                        loss_weights = self.loss_weights,
                                                        metrics={'Decoder': 'mse'})

    def loss_mse_velocity_loss(self,y_true,y_pred):
        mse = K.mean(K.square(y_pred-y_true))
        mse_v = K.mean(K.square((y_pred[:,1:,:] - y_pred[:,0:-1,:])-(y_true[:,1:,:] - y_true[:,0:-1,:])))

        return mse+self.alpha*mse_v

    def init_loss_history_list(self):

        self.loss_history = {"training_loss_discriminator": [],
                             "training_loss_mse_autoencoder": [],
                             "training_loss_crossentropy_encoder":[],
                             "training_metric_mse":[],
                             "training_metric_mse_v":[],
                             "valid_loss_discriminator": [],
                             "valid_loss_mse_autoencoder": [],
                             "valid_loss_crossentropy_encoder":[],
                             "valid_metric_mse": [],
                             "valid_metric_mse_v": [],
                             }


    def print_loss_history(self):
        for loss_key in sorted(self.loss_history.keys()):
            print "%s:%f"%(loss_key,self.loss_history[loss_key][-1])

    def plot_latent_space(self,latent_codes,filename,Y1=None,Y2=None,dim_x=0,dim_y=1):
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
            # plt.xlim([-5.0,5.0])
            # plt.ylim([-5.0,5.0])
        plt.legend(fontsize=15)
        plt.savefig(filename)



    def compute_loss_and_plot_latent_space(self,epoch,whichdataset='training'):
        if whichdataset=='training':
            dataset = self.train_X
            Y1 = self.train_Y1
            Y2 = self.train_Y2
            positive_examples = self.positive_examples_training
        elif whichdataset =='valid':
            dataset = self.valid_X
            Y1 = self.valid_Y1
            Y2 = self.valid_Y2
            positive_examples = self.positive_examples_valid
        elif whichdataset == 'test':
            dataset = self.test_X
            Y1 = self.test_Y1
            Y2 = self.test_Y2
            positive_examples = self.positive_examples_test
        else:
            raise ValueError('wrong argument whichdataset')
        num_samples = len(dataset)
        latent_codes = self.encoder.predict(x=dataset,batch_size=1000,verbose=0)
        #plot latent space
        if epoch %self.checkpoint_epochs==0:
            filename = 'Epoch' + str(epoch) + '_' + whichdataset + '01.png'
            self.plot_latent_space(latent_codes,filename,Y1,dim_x=0,dim_y=1)
            filename = 'Epoch' + str(epoch) + '_' + whichdataset + '23.png'
            self.plot_latent_space(latent_codes, filename, Y1, dim_x=2, dim_y=3)
            self.plot_2d_histogram_latent_codes(latent_codes,suffix=whichdataset+'_epoch' + str(epoch))

        X = np.concatenate((positive_examples, latent_codes),axis=0)
        Y = [1.]*num_samples +[0.]*num_samples
        loss_dis, accuracy_dis = self.discriminator.evaluate(X,Y,batch_size=1000,verbose=0)

        Y_hat = np.asarray([1.] * num_samples)

        total_loss, loss_autoencoder,loss_crossentropy_encoder,metric_mse = \
            self.autoencoder_with_discriminator.evaluate(x=dataset,y={'Decoder':dataset,'Discmt':Y_hat},batch_size=1000,verbose=0)
        metric_mse_v = (10000*(loss_autoencoder-metric_mse)/self.alpha).eval()
        if whichdataset=='training':
            self.loss_history["training_loss_discriminator"].append(loss_dis)
            self.loss_history["training_loss_mse_autoencoder"].append(loss_autoencoder)
            self.loss_history["training_loss_crossentropy_encoder"].append(loss_crossentropy_encoder)
            self.loss_history["training_metric_mse"].append(metric_mse)
            self.loss_history["training_metric_mse_v"].append(metric_mse_v)

        elif whichdataset=='valid':
            self.loss_history["valid_loss_discriminator"].append(loss_dis)
            self.loss_history["valid_loss_mse_autoencoder"].append(loss_autoencoder)
            self.loss_history["valid_loss_crossentropy_encoder"].append(loss_crossentropy_encoder)
            self.loss_history["valid_metric_mse"].append(metric_mse)
            self.loss_history["valid_metric_mse_v"].append(metric_mse_v)

        elif whichdataset=='test':
            raise NotImplementedError()


    def plot_loss(self):
        # plot mse
        fig = plt.figure(figsize=(5, 5*5))
        plt.subplot(5,1,1)
        legend_str =[]
        plt.plot(self.loss_history["training_loss_mse_autoencoder"])
        legend_str.append('training_mse'+':%f' % self.loss_history["training_loss_mse_autoencoder"][-1])
        plt.plot(self.loss_history["valid_loss_mse_autoencoder"])
        legend_str.append('valid_mse'+':%f' % self.loss_history["valid_loss_mse_autoencoder"][-1])
        plt.legend(legend_str,fontsize=10,loc='best',fancybox=True,framealpha=0.5)
        plt.xlabel('Epoch Number')
        plt.ylabel('loss (mse+alpha*mse_v)')

        plt.subplot(5, 1, 2)
        legend_str = []
        plt.plot(self.loss_history['training_metric_mse'])
        legend_str.append('training_metric_mse:%f' % self.loss_history['training_metric_mse'][-1])
        plt.plot(self.loss_history['valid_metric_mse'])
        legend_str.append(
            'valid_metric_mse:%f' % self.loss_history['valid_metric_mse'][-1])
        plt.legend(legend_str, fontsize=10, loc='best', fancybox=True, framealpha=0.5)
        plt.xlabel('Epoch Number')
        plt.ylabel('MSE')

        plt.subplot(5, 1, 3)
        legend_str = []
        plt.plot(self.loss_history['training_metric_mse_v'])
        legend_str.append('training_metric_mse_v:%f' % self.loss_history['training_metric_mse_v'][-1])
        plt.plot(self.loss_history['valid_metric_mse_v'])
        legend_str.append(
            'valid_metric_mse_v:%f' % self.loss_history['valid_metric_mse_v'][-1])
        plt.legend(legend_str, fontsize=10, loc='best', fancybox=True, framealpha=0.5)
        plt.xlabel('Epoch Number')
        plt.ylabel('mse_v')

        #plot loss of discriminator
        plt.subplot(5,1,4)
        legend_str = []
        plt.plot(self.loss_history['training_loss_discriminator'])
        legend_str.append('training_discriminator:%f'%self.loss_history['training_loss_discriminator'][-1])
        plt.plot(self.loss_history['training_loss_crossentropy_encoder'])
        legend_str.append('training_loss_crossentropy_encoder:%f' % self.loss_history['training_loss_crossentropy_encoder'][-1])
        plt.legend(legend_str,fontsize=10,loc='best',fancybox=True,framealpha=0.5)
        plt.xlabel('Epoch Number')
        plt.ylabel('adversarial loss on training set')
        # plot loss of discriminator
        plt.subplot(5,1,5)
        legend_str = []
        plt.plot(self.loss_history['valid_loss_discriminator'])
        legend_str.append('valid_discriminator:%f' % self.loss_history['valid_loss_discriminator'][-1])
        plt.plot(self.loss_history['valid_loss_crossentropy_encoder'])
        legend_str.append(
            'valid_loss_crossentropy_encoder:%f' % self.loss_history['valid_loss_crossentropy_encoder'][-1])
        plt.legend(legend_str,fontsize=10,loc='best',fancybox=True,framealpha=0.5)
        plt.xlabel('Epoch Number')
        plt.ylabel('adversarial loss on valid set')
        plt.tight_layout()
        plt.savefig('./learning_curve.png')


    def training(self,dataset_obj):
        self.set_up_dataset(dataset_obj)
        self.set_up_model()
        self.init_loss_history_list()
        for epoch in range(self.max_epoch):
            print('Epoch seen: {}'.format(epoch))
            print "training set size: %d" % len(self.train_X)
            print "alpha = {}".format(self.alpha.eval())
            self.cur_lr_autoencoder = self.autoencoder_with_discriminator.optimizer.lr * (1. / (
                1 + self.autoencoder_with_discriminator.optimizer.decay * self.autoencoder_with_discriminator.optimizer.iterations))
            print "current learning rate of autoencoder: {}".format(self.cur_lr_autoencoder.eval())
            self.set_up_noise_examples()
            self.compute_loss_and_plot_latent_space(epoch,'training')
            self.compute_loss_and_plot_latent_space(epoch,'valid')
            self.print_loss_history()
            self.plot_loss()
            self.training_loop(self.train_X,self.positive_examples_training,batch_size=self.batch_size)
            if epoch %self.checkpoint_epochs==0:
                self.save_models(suffix=str(epoch))
                reconstruced_seqs, output_discriminator = self.autoencoder_with_discriminator.predict(self.test_X[:self.nb_to_generate])
                self.save_generated_seqs(self.test_X[:self.nb_to_generate],
                                         max_vector=self.max_vector,
                                         min_vector=self.min_vector,
                                         suffix='original_input',epoch=epoch)
                self.save_generated_seqs(reconstruced_seqs,max_vector=self.max_vector,
                                         min_vector=self.min_vector,suffix='reconstructed',epoch=epoch)
                random_generated_seqs = self.decoder.predict(x=self.noise_vectors)
                self.save_generated_seqs(random_generated_seqs,max_vector=self.max_vector,
                                         min_vector=self.min_vector,suffix='random_generated',epoch=epoch)
                np.savez('loss_history.npz', self.loss_history)

        self.plot_loss()
        self.save_models()
        np.savez('loss_history.npz', self.loss_history)

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


    def save_generated_seqs(self,seqs,max_vector,min_vector,suffix = '',epoch=None):
        sequences = self.postprocess(seqs,max_vector,min_vector)
        for i, seq in enumerate(sequences):
            if epoch is None:
                filename = './generated_seq' + '_num_' + str(i) + suffix + '.bvh'
            else:
                filename = './generated_seq_epoch_' + str(epoch) + '_num_' + str(i) + suffix + '.bvh'
            save_seq_2_bvh(seq, sequence_name=filename, step=self.sampling_interval)

    def save_models(self,suffix=''):
        self.autoencoder_with_discriminator.save_weights('autoencoder'+suffix+'.h5')
        with open('autoencoder'+suffix+'.yaml','w') as yaml_file:
            yaml_file.write(self.autoencoder_with_discriminator.to_yaml())

        self.encoder.save_weights('encoder'+suffix+'.h5')
        with open('encoder'+suffix+'.yaml','w') as yaml_file:
            yaml_file.write(self.encoder.to_yaml())

        self.decoder.save_weights('decoder'+suffix+'.h5')
        with open('decoder'+suffix+'.yaml','w') as yaml_file:
            yaml_file.write(self.decoder.to_yaml())

        self.discriminator.save_weights('discriminator'+suffix+'.h5')
        with open('discriminator'+suffix+'.yaml','w') as yaml_file:
            yaml_file.write(self.discriminator.to_yaml())



    def batch_generator(self,iterable1,iterable2,batch_size=1,shuffle=False):
        l = len(iterable1)
        if shuffle ==True:
            indices = np.random.permutation(len(iterable1))
        else:
            indices = np.arange(0,stop=len(iterable1))
        for ndx in range(0,l,batch_size):
            cur_indices = indices[ndx:min(ndx+batch_size,l)]
            yield  iterable1[cur_indices],iterable2[cur_indices]


    def training_loop(self,dataset,positive_noise_set,batch_size):
        #batch generator
        self.data_generator = self.batch_generator(dataset,positive_noise_set,batch_size=batch_size)

        for motion_batch,prior_noise_batch in self.data_generator:
            cur_batch_size = len(motion_batch)
            latent_codes = self.encoder.predict(x=motion_batch,batch_size=cur_batch_size)
            X = np.concatenate((prior_noise_batch, latent_codes),axis=0)
            Y = [1.]*cur_batch_size+[0.]*cur_batch_size
            if self.train_disc ==True:
                self.discriminator.trainable=True
                self.discriminator.train_on_batch(x=X,y=Y)
            self.discriminator.trainable = False
            Y_hat = np.asarray([1.]*cur_batch_size,dtype=np.float32)
            self.autoencoder_with_discriminator.train_on_batch(x=motion_batch,y={'Decoder':motion_batch,\
                                                    'Discmt':Y_hat})

    def fit(self,X,y=None):
        self.set_up_dataset(self.dataset_obj)
        self.set_up_model()
        self.init_loss_history_list()
        print "training set size: %d" % len(self.train_X)
        for epoch in range(self.max_epoch):
            print('Epoch seen: {}'.format(epoch))
            self.set_up_noise_examples()
            self.training_loop(self.train_X, self.positive_examples_training, batch_size=self.batch_size)

        print "Finish this fitting process"
        return self

    def score(self,X,y=None):
        Y_hat = np.asarray([1.] * len(self.valid_X))
        loss_autoencoder, loss_mse_autoencoder, loss_crossentropy_encoder,metric_mse = \
            self.autoencoder_with_discriminator.evaluate(x=self.valid_X, y={'Decoder': self.valid_X, 'Discmt': Y_hat},
                                                         batch_size=1000, verbose=0)
        return -1. * metric_mse


if __name__ == "__main__":
    import sys
    import os
    root_path = os.getenv('Seq_AAE_V1')
    sys.path.append(root_path)
    from Seq_AAE_V1.datasets.dataset import Emilya_Dataset

    dataset_obj = Emilya_Dataset(window_width=200,shift_step=20,
                                 sampling_interval=None,
                                 with_velocity=False,
                                 number=200,nb_valid=200,nb_test=200)
    model = Sequence_Adversrial_Autoencoder_with_New_Loss(latent_dim=50,latent_activation='tanh',latent_BN=False,
                                    hidden_dim_enc_list=[100,100],activation_enc_list=['tanh','tanh'],
                                    hidden_dim_dec_list=[100,100],activation_dec_list=['tanh','tanh'],
                                    hidden_dim_dis_list=[100,40],activation_dis_list=['relu','relu'],
                                    dropout_dis_list=[0.0,0.0],batch_size=20,max_epoch=100,
                                    optimiser_autoencoder='rmsprop',optimiser_dis='sgd',
                                    lr_autoencoder=0.001,lr_dis=0.01,decay_autoencoder=0.0,decay_dis=0.0,
                                    momentum_autoencoder=0.0,momentum_dis=0.0,
                                    prior_noise_type='Gaussian',loss_weights=[1.0,0.001],train_disc=True,
                                    custom_loss=True,symetric_autoencoder=False,
                                    nb_to_generate=30)
    model.training(dataset_obj)


