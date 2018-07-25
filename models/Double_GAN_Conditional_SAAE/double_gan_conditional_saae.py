
'''
Sequential Autoencoder using the velocity along xz plane as contextual variables
'''
__author__ = 'qiwang'
import numpy as np
from keras.layers import Input,LSTM,RepeatVector,Dense,SimpleRNN,GRU,Embedding
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
from keras.utils.vis_utils import plot_model
from Seq_AAE_V1.synthesis_scripts.synthesis_utils import save_seq_2_bvh
from matplotlib.colors import Colormap,LogNorm,PowerNorm
from keras.models import model_from_yaml

class Double_GAN_Conditional_SAAE(BaseEstimator):

    def __init__(self,latent_dim=50,latent_activation='tanh',latent_BN=False,
                 hidden_dim_enc_list=[100,100],activation_enc_list=['tanh','tanh'],
                 hidden_dim_dec_list=None,activation_dec_list=None,
                 hidden_dim_dis_list=[100,40],activation_dis_list=['relu','relu'],
                 hidden_dim_classifier_list=[100, 40], activation_classifier_list=['relu', 'relu'],
                 dropout_dis_list=[0.0,0.0],
                 dropout_classifier_list=[0.0, 0.0],
                 batch_size=200,max_epoch=200,
                 optimiser_autoencoder='rmsprop',optimiser_dis='sgd',
                 optimiser_classifier= 'sgd',
                 lr_autoencoder=0.001,lr_dis=0.01,
                 decay_autoencoder=0.0,decay_dis=0.0,
                 lr_classifier = 0.01,decay_classifier = 0.0,
                 momentum_autoencoder=0.0,momentum_dis=0.0,
                 momentum_classifier = 0.0,
                 prior_noise_type='Gaussian',
                 nb_mixture = 8,
                 loss_weights=[1.0,0.0,0.0],
                 train_disc=True,
                 train_classifier = True,
                 custom_loss=False,
                 loss_weight_mse_v= 1.0,
                 checkpoint_epochs=100,symetric_autoencoder=False,
                 nb_to_generate=30,
                 condition_activity_or_emotion=1,
                 nb_label=8,
                 fully_condition=True,dataset_obj=None,embedding_dim=0,
                 is_annealing_beta = False,
                 beta_anneal_rate=0.1,
                 bias_beta = 9.,
                 reload_model_path = None,
                 reload_checkpoint_epoch=0):

        args = locals().copy()
        del args['self']
        self.__dict__.update(args)
        print args.keys()
        self.save_configuration(args)

    def set_up_model(self):
        if self.reload_model_path is not None:
            self.iterations = self.reload_checkpoint_epoch * (len(self.train_X)/self.batch_size+1)
            if self.hidden_dim_dec_list is None:
                self.hidden_dim_dec_list = self.hidden_dim_enc_list[::-1]
            if self.activation_dec_list is None:
                self.activation_dec_list = self.activation_enc_list[::-1]

            self.alpha = K.variable(self.loss_weight_mse_v, name='alpha')
            if self.is_annealing_beta is True:
                self.beta_classifier = K.variable(0., name='beta_classifier')
            else:
                self.beta_classifier = K.variable(1., name='beta_classifier')

            #load all the models
            encoder_path = self.reload_model_path +'encoder'+str(self.reload_checkpoint_epoch)+'.yaml'
            # with open(encoder_path) as f:
            #     self.encoder = model_from_yaml(f)
            self.encoder = self.Encoder()
            self.encoder.load_weights(encoder_path[:-4]+'h5')

            decoder_path = self.reload_model_path + 'decoder' + str(self.reload_checkpoint_epoch) + '.yaml'
            # with open(decoder_path) as f:
            #     self.decoder = model_from_yaml(f)
            self.decoder = self.Decoder()
            self.decoder.load_weights(decoder_path[:-4]+'h5')

            discriminator_path = self.reload_model_path + 'discriminator' + str(self.reload_checkpoint_epoch) + '.yaml'
            # with open(discriminator_path) as f:
            #     self.discriminator = model_from_yaml(f)
            self.discriminator = self.Discriminator()
            self.discriminator.load_weights(discriminator_path[:-4]+'h5')

            classifier_path = self.reload_model_path + 'classifier' + str(self.reload_checkpoint_epoch) + '.yaml'
            # with open(classifier_path) as f:
            #     self.classifier = model_from_yaml(f)
            self.classifier = self.Classifier()
            self.classifier.load_weights(classifier_path[:-4] + 'h5')

            autoencoder_path = self.reload_model_path + 'autoencoder' + str(self.reload_checkpoint_epoch) + '.yaml'
            # with open(autoencoder_path) as f:
            #     self.autoencoder_with_discriminator = model_from_yaml(f)
            self.autoencoder_with_discriminator = self.Autoencoder_with_Discmt()
            self.autoencoder_with_discriminator.load_weights(autoencoder_path[:-4] + 'h5')
        else:
            self.iterations = 0
            if self.hidden_dim_dec_list is None:
                self.hidden_dim_dec_list = self.hidden_dim_enc_list[::-1]
            if self.activation_dec_list is None:
                self.activation_dec_list = self.activation_enc_list[::-1]

            self.alpha = K.variable(self.loss_weight_mse_v, name='alpha')
            if self.is_annealing_beta is True:
                self.beta_classifier = K.variable(0.,name='beta_classifier')
            else:
                self.beta_classifier = K.variable(1., name='beta_classifier')
            if self.symetric_autoencoder==True:
                self.encoder = self.Encoder_symetric()
            else:
                self.encoder = self.Encoder()
            self.decoder = self.Decoder()
            self.discriminator = self.Discriminator()
            self.classifier = self.Classifier()
            self.autoencoder_with_discriminator = self.Autoencoder_with_Discmt()

        self.compile()
        if self.reload_model_path is not None:
            K.set_value(self.autoencoder_with_discriminator.optimizer.iterations, self.iterations)
            K.set_value(self.discriminator.optimizer.iterations, self.iterations)
            K.set_value(self.classifier.optimizer.iterations, self.iterations)

        #summary of model
        self.visualize_model([self.encoder,self.decoder,self.discriminator,self.classifier,
                              self.autoencoder_with_discriminator])

    def visualize_model(self,models):
        for model in models:
            model.summary()
            plot_model(model,
                       to_file='%s.png' % model.name,
                       show_shapes=True,
                       show_layer_names=True)

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

        if self.condition_activity_or_emotion==1:
            self.train_label_one_hot_code = self.convert_indices_2_onehot(targets=self.train_Y1,nb_labels=self.nb_label)
            self.valid_label_one_hot_code = self.convert_indices_2_onehot(targets=self.valid_Y1,nb_labels=self.nb_label)
            self.test_label_one_hot_code = self.convert_indices_2_onehot(targets=self.test_Y1,nb_labels=self.nb_label)
        elif self.condition_activity_or_emotion==2:
            self.train_label_one_hot_code = self.convert_indices_2_onehot(targets=self.train_Y2,nb_labels=self.nb_label)
            self.valid_label_one_hot_code = self.convert_indices_2_onehot(targets=self.valid_Y2,nb_labels=self.nb_label)
            self.test_label_one_hot_code = self.convert_indices_2_onehot(targets=self.test_Y2,nb_labels=self.nb_label)

        #self.test_speed = dataset_obj.test_speed
        self.max_vector = dataset_obj.max_vector[1:]
        self.min_vector = dataset_obj.min_vector[1:]

        self.postprocess = dataset_obj.postprocess
        self.sampling_interval = dataset_obj.sampling_interval
        # set up noise vector to generate seqeunces

        if self.prior_noise_type == 'Gaussian':
            mean = np.zeros(self.latent_dim)
            covariance = np.eye(N=self.latent_dim) * 1.0
            self.noise_vectors = generate_positive_samples(self.nb_to_generate/3,
                                                           mean, covariance,
                                                           'Gaussian', seed=1234)
        elif self.prior_noise_type =='Mixture':
            mean = np.zeros(self.latent_dim)
            covariance = np.eye(N=self.latent_dim) * 1.0
            self.noise_vectors = generate_positive_samples(self.nb_to_generate / 3,
                                                           mean, covariance,
                                                           type=self.prior_noise_type,
                                                           nb_mixture=self.nb_mixture,seed=1234)
        else:
            raise NotImplementedError()

    def convert_indices_2_onehot(self,targets, nb_labels):
        tmp_targets = targets.astype(int)
        ohm = np.zeros((tmp_targets.shape[0], nb_labels), dtype=np.float32)
        ohm[np.arange(tmp_targets.shape[0]), tmp_targets] = 1.0
        return ohm

    def set_up_noise_examples(self):
        if self.prior_noise_type=='Gaussian':
            mean = np.zeros(self.latent_dim)
            covariance = np.eye(N=self.latent_dim) * 1.0
            self.positive_examples_training = generate_positive_samples(len(self.train_X),mean,covariance,'Gaussian',seed=np.random.randint(0,2000))
            self.positive_examples_valid = generate_positive_samples(len(self.valid_X), mean, covariance, 'Gaussian',seed=np.random.randint(0,2000))
            self.positive_examples_test = generate_positive_samples(len(self.test_X), mean, covariance, 'Gaussian',seed=np.random.randint(0,2000))

        elif self.prior_noise_type =='Mixture':
            mean = np.zeros(self.latent_dim)
            covariance = np.eye(N=self.latent_dim) * 1.0
            self.positive_examples_training = generate_positive_samples(len(self.train_X),
                                                                        mean,
                                                                        covariance,
                                                                        type=self.prior_noise_type,
                                                                        nb_mixture=self.nb_mixture,
                                                                        seed=np.random.randint(0, 2000))
            self.positive_examples_valid = generate_positive_samples(len(self.valid_X), mean, covariance,
                                                                     type=self.prior_noise_type,
                                                                     nb_mixture=self.nb_mixture,
                                                                     seed=np.random.randint(0, 2000))
            self.positive_examples_test = generate_positive_samples(len(self.test_X), mean, covariance,
                                                                    type=self.prior_noise_type,
                                                                    nb_mixture=self.nb_mixture,
                                                                    seed=np.random.randint(0, 2000))


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

        label_input = Input(shape=(self.nb_label,), name='label_input')

        if self.embedding_dim != 0:
            label_seq = Dense(units=self.embedding_dim,activation='tanh',use_bias=False,
                                kernel_initializer='uniform')(label_input)
        else:
            label_seq = label_input
        label_seq = RepeatVector(self.max_len)(label_seq)
        decoded = merge([latent_input_seq,label_seq],mode='concat')

        for i,(dim, activation) in enumerate(zip(self.hidden_dim_dec_list,self.activation_dec_list)):
            decoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(decoded)
            if self.fully_condition:
                decoded = merge([decoded,label_seq],mode='concat')

        decoded = SimpleRNN(output_dim=self.dof,activation='sigmoid',name='decoder_output',return_sequences=True)(decoded)
        return Model(input = [latent_input,label_input], output=decoded,name='Decoder')

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

    def Classifier(self):
        input = Input(shape=(self.latent_dim,), name='classifier_input')
        for i, (dim, activation, dropout) in enumerate(zip(self.hidden_dim_classifier_list, self.activation_classifier_list, self.dropout_classifier_list)):
            if i == 0:
                discmt = Dense(dim, activation=activation)(input)
            else:
                discmt = Dropout(dropout)(discmt)
                discmt = Dense(dim, activation=activation)(discmt)

        discmt = Dense(self.nb_label, activation='softmax', name='classifier_output')(discmt)
        return Model(input=input, output=discmt, name='Classifier')

    def Autoencoder_with_Discmt(self):
        autoencoder = self.decoder([self.encoder.output,self.decoder.inputs[1]])
        self.discriminator.trainable = False
        self.classifier.trainable = False
        aux_output_discmt = self.discriminator(self.encoder.output)
        aux_output_classifier =  self.classifier(self.encoder.output)
        return Model(input=[self.encoder.input,self.decoder.inputs[1]],
                     output=[autoencoder, aux_output_discmt,aux_output_classifier], name='Autoencoder_with_Dis')


    def compile(self):
        if self.optimiser_autoencoder =='sgd':
            optimizer_ae = SGD(lr=self.lr_autoencoder,decay=self.decay_autoencoder)
        elif self.optimiser_autoencoder =='rmsprop':
            optimizer_ae = RMSprop(lr=self.lr_autoencoder,decay=self.decay_autoencoder)
        elif self.optimiser_autoencoder =='adam':
            optimizer_ae = Adam(lr=self.lr_autoencoder,decay=self.decay_autoencoder)

        if self.optimiser_dis =='sgd':
            optimizer_discmt = SGD(lr=self.lr_dis,decay=self.decay_dis)
        elif self.optimiser_dis =='adam':
            optimizer_discmt = Adam(lr=self.lr_dis,decay=self.decay_dis)
        elif self.optimiser_dis =='rmsprop':
            optimizer_discmt = RMSprop(lr=self.lr_dis, decay=self.decay_dis)

        if self.optimiser_classifier =='sgd':
            optimizer_classifier = SGD(lr=self.lr_classifier,decay=self.decay_classifier)
        elif self.optimiser_classifier =='adam':
            optimizer_classifier = Adam(lr=self.lr_classifier,decay=self.decay_classifier)
        elif self.optimiser_classifier =='rmsprop':
            optimizer_classifier = RMSprop(lr=self.lr_classifier,decay=self.decay_classifier)

        self.discriminator.trainable = True
        self.discriminator.compile(optimizer_discmt,loss='binary_crossentropy',metrics=['accuracy'])
        self.discriminator.trainable = False
        self.classifier.trainable =True
        self.classifier.compile(optimizer_classifier,loss='categorical_crossentropy',metrics=['accuracy'])
        self.classifier.trainable = False

        if self.custom_loss==True:
            self.autoencoder_with_discriminator.compile(optimizer_ae, \
                                                        loss={'Decoder': self.loss_mse_velocity_loss, \
                                                              'Discmt': 'binary_crossentropy',
                                                              'Classifier':self.loss_neg_entropy}, \
                                                        loss_weights=self.loss_weights,
                                                        metrics={'Decoder':'mse'})

        else:
            self.autoencoder_with_discriminator.compile(optimizer_ae,\
                                                        loss={'Decoder':'mse',\
                                                              'Discmt':'binary_crossentropy',
                                                              'Classifier':self.loss_neg_entropy},\
                                                        loss_weights = self.loss_weights,
                                                        metrics={'Decoder': 'mse'})

    def loss_mse_velocity_loss(self,y_true,y_pred):
        mse = K.mean(K.square(y_pred-y_true))
        mse_v = K.mean(K.square((y_pred[:,1:,:] - y_pred[:,0:-1,:])-(y_true[:,1:,:] - y_true[:,0:-1,:])))

        return mse+self.alpha*mse_v

    def loss_neg_entropy(self,y_true,y_pred):
        y_pred/= y_pred.sum(axis=-1, keepdims=True)
        # avoid numerical instability with _EPSILON clipping
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        return self.beta_classifier*K.sum(y_pred* K.log(y_pred),axis=y_pred.ndim - 1)


    def init_loss_history_list(self):

        self.loss_history = {"training_loss_discriminator": [],
                             "training_loss_mse_autoencoder": [],
                             "training_loss_crossentropy_encoder":[],
                             'training_loss_classifier':[],
                             "traning_loss_neg_entropy_encoder":[],
                             "training_accuracy_classifier":[],
                             "training_metric_mse":[],
                             "training_metric_mse_v":[],
                             "valid_loss_discriminator": [],
                             "valid_loss_mse_autoencoder": [],
                             "valid_loss_crossentropy_encoder":[],
                             'valid_loss_classifier': [],
                             "valid_loss_neg_entropy_encoder": [],
                             "valid_accuracy_classifier": [],
                             "valid_metric_mse": [],
                             "valid_metric_mse_v": [],
                             "beta_classifier":[]
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
        plt.close(fig)



    def compute_loss_and_plot_latent_space(self,epoch,whichdataset='training'):
        if whichdataset=='training':
            dataset = self.train_X
            Y1 = self.train_Y1
            Y2 = self.train_Y2
            positive_examples = self.positive_examples_training
            label_one_hot_code = self.train_label_one_hot_code
        elif whichdataset =='valid':
            dataset = self.valid_X
            Y1 = self.valid_Y1
            Y2 = self.valid_Y2
            positive_examples = self.positive_examples_valid
            label_one_hot_code = self.valid_label_one_hot_code
        elif whichdataset == 'test':
            dataset = self.test_X
            Y1 = self.test_Y1
            Y2 = self.test_Y2
            positive_examples = self.positive_examples_test
            label_one_hot_code = self.test_label_one_hot_code
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

        loss_classifier,accuracy_classifier = self.classifier.evaluate(x=latent_codes,y=label_one_hot_code,batch_size=1000,verbose=0)

        Y_hat = np.asarray([1.] * num_samples)

        total_loss, loss_autoencoder,loss_crossentropy_encoder,loss_neg_entropy_encoder, metric_mse = \
            self.autoencoder_with_discriminator.evaluate(x=[dataset,label_one_hot_code],
                                                         y={'Decoder':dataset,
                                                            'Discmt':Y_hat,
                                                            'Classifier':Y_hat},batch_size=1000,verbose=0)

        metric_mse_v = (10000*(loss_autoencoder-metric_mse)/self.alpha).eval()
        loss_neg_entropy_encoder = (loss_neg_entropy_encoder / self.beta_classifier).eval()
        if whichdataset=='training':
            self.loss_history["training_loss_discriminator"].append(loss_dis)
            self.loss_history["training_loss_mse_autoencoder"].append(loss_autoencoder)
            self.loss_history["training_loss_crossentropy_encoder"].append(loss_crossentropy_encoder)
            self.loss_history['training_loss_classifier'].append(loss_classifier)
            self.loss_history['traning_loss_neg_entropy_encoder'].append(loss_neg_entropy_encoder)
            self.loss_history['training_accuracy_classifier'].append(accuracy_classifier)
            self.loss_history["training_metric_mse"].append(metric_mse)
            self.loss_history["training_metric_mse_v"].append(metric_mse_v)

        elif whichdataset=='valid':
            self.loss_history["valid_loss_discriminator"].append(loss_dis)
            self.loss_history["valid_loss_mse_autoencoder"].append(loss_autoencoder)
            self.loss_history["valid_loss_crossentropy_encoder"].append(loss_crossentropy_encoder)
            self.loss_history['valid_loss_classifier'].append(loss_classifier)
            self.loss_history['valid_loss_neg_entropy_encoder'].append(loss_neg_entropy_encoder)
            self.loss_history['valid_accuracy_classifier'].append(accuracy_classifier)
            self.loss_history["valid_metric_mse"].append(metric_mse)
            self.loss_history["valid_metric_mse_v"].append(metric_mse_v)

        elif whichdataset=='test':
            raise NotImplementedError()


    def plot_loss(self):
        # plot mse
        fig = plt.figure(figsize=(5, 5*8))
        plt.subplot(8,1,1)
        legend_str =[]
        plt.plot(self.loss_history["training_loss_mse_autoencoder"])
        legend_str.append('training_mse'+':%f' % self.loss_history["training_loss_mse_autoencoder"][-1])
        plt.plot(self.loss_history["valid_loss_mse_autoencoder"])
        legend_str.append('valid_mse'+':%f' % self.loss_history["valid_loss_mse_autoencoder"][-1])
        plt.legend(legend_str,fontsize=10,loc='best',fancybox=True,framealpha=0.5)
        plt.xlabel('Epoch Number')
        plt.ylabel('loss (mse+alpha*mse_v)')

        plt.subplot(8, 1, 2)
        legend_str = []
        plt.plot(self.loss_history['training_metric_mse'])
        legend_str.append('training_metric_mse:%f' % self.loss_history['training_metric_mse'][-1])
        plt.plot(self.loss_history['valid_metric_mse'])
        legend_str.append(
            'valid_metric_mse:%f' % self.loss_history['valid_metric_mse'][-1])
        plt.legend(legend_str, fontsize=10, loc='best', fancybox=True, framealpha=0.5)
        plt.xlabel('Epoch Number')
        plt.ylabel('MSE')

        plt.subplot(8, 1, 3)
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
        plt.subplot(8,1,4)
        legend_str = []
        plt.plot(self.loss_history['training_loss_discriminator'])
        legend_str.append('training_discriminator:%f'%self.loss_history['training_loss_discriminator'][-1])
        plt.plot(self.loss_history['training_loss_crossentropy_encoder'])
        legend_str.append('training_loss_crossentropy_encoder:%f' % self.loss_history['training_loss_crossentropy_encoder'][-1])
        plt.legend(legend_str,fontsize=10,loc='best',fancybox=True,framealpha=0.5)
        plt.xlabel('Epoch Number')
        plt.ylabel('adversarial loss on training set')
        # plot loss of discriminator
        plt.subplot(8,1,5)
        legend_str = []
        plt.plot(self.loss_history['valid_loss_discriminator'])
        legend_str.append('valid_discriminator:%f' % self.loss_history['valid_loss_discriminator'][-1])
        plt.plot(self.loss_history['valid_loss_crossentropy_encoder'])
        legend_str.append(
            'valid_loss_crossentropy_encoder:%f' % self.loss_history['valid_loss_crossentropy_encoder'][-1])
        plt.legend(legend_str,fontsize=10,loc='best',fancybox=True,framealpha=0.5)
        plt.xlabel('Epoch Number')
        plt.ylabel('adversarial loss on valid set')

        plt.subplot(8, 1, 6)
        legend_str = []
        plt.plot(self.loss_history['training_loss_classifier'])
        legend_str.append('training_loss_classifier:%f' % self.loss_history['training_loss_classifier'][-1])
        plt.plot(self.loss_history['valid_loss_classifier'])
        legend_str.append(
            'valid_loss_classifier:%f' % self.loss_history['valid_loss_classifier'][-1])
        plt.legend(legend_str, fontsize=10, loc='best', fancybox=True, framealpha=0.5)
        plt.xlabel('Epoch Number')
        plt.ylabel('classifier loss')

        plt.subplot(8, 1, 7)
        legend_str = []
        plt.plot(self.loss_history['training_accuracy_classifier'])
        legend_str.append('training_accuracy_classifier:%f' % self.loss_history['training_accuracy_classifier'][-1])
        plt.plot(self.loss_history['valid_accuracy_classifier'])
        legend_str.append(
            'valid_accuracy_classifier:%f' % self.loss_history['valid_accuracy_classifier'][-1])
        plt.legend(legend_str, fontsize=10, loc='best', fancybox=True, framealpha=0.5)
        plt.xlabel('Epoch Number')
        plt.ylabel('classifier accuracy')

        plt.subplot(8, 1, 8)
        legend_str = []
        plt.plot(self.loss_history['traning_loss_neg_entropy_encoder'])
        legend_str.append('traning_loss_neg_entropy_encoder:%f' % self.loss_history['traning_loss_neg_entropy_encoder'][-1])
        plt.plot(self.loss_history['valid_loss_neg_entropy_encoder'])
        legend_str.append(
            'valid_loss_neg_entropy_encoder:%f' % self.loss_history['valid_loss_neg_entropy_encoder'][-1])
        plt.legend(legend_str, fontsize=10, loc='best', fancybox=True, framealpha=0.5)
        plt.xlabel('Epoch Number')
        plt.ylabel('neg entropy of encoder')

        plt.tight_layout()
        plt.savefig('./learning_curve.png')
        plt.close(fig)



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
            print "current beta_classifier value is {}".format((self.beta_classifier+0.0).eval())
            self.loss_history['beta_classifier'].append((self.beta_classifier+0.0).eval())
            self.set_up_noise_examples()
            self.compute_loss_and_plot_latent_space(epoch,'training')
            self.compute_loss_and_plot_latent_space(epoch,'valid')
            self.print_loss_history()
            self.plot_loss()

            self.training_loop(self.train_X,self.train_label_one_hot_code,self.positive_examples_training,batch_size=self.batch_size)
            if epoch %self.checkpoint_epochs==0:
                self.save_models(suffix=str(epoch))
                reconstruced_seqs, output_discriminator,output_classifier = self.autoencoder_with_discriminator.predict([self.test_X[:self.nb_to_generate],
                                                                                                       self.test_label_one_hot_code[:self.nb_to_generate]])
                self.save_generated_seqs(self.test_X[:self.nb_to_generate],
                                         max_vector=self.max_vector,
                                         min_vector=self.min_vector,
                                         suffix='original_input',epoch=epoch)
                self.save_generated_seqs(reconstruced_seqs,max_vector=self.max_vector,
                                         min_vector=self.min_vector,suffix='reconstructed',epoch=epoch)
                for ll in range(self.nb_label):
                    ll_one_hot_code= self.convert_indices_2_onehot(np.asarray([ll]*(self.nb_to_generate/3)),nb_labels=self.nb_label)
                    random_generated_seqs = self.decoder.predict(x=[self.noise_vectors,ll_one_hot_code])
                    self.save_generated_seqs(random_generated_seqs,max_vector=self.max_vector,
                                             min_vector=self.min_vector,suffix='random_generated_label'+str(ll),epoch=epoch)
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
        yaml_file.close()

        self.encoder.save_weights('encoder'+suffix+'.h5')
        with open('encoder'+suffix+'.yaml','w') as yaml_file:
            yaml_file.write(self.encoder.to_yaml())
        yaml_file.close()

        self.decoder.save_weights('decoder'+suffix+'.h5')
        with open('decoder'+suffix+'.yaml','w') as yaml_file:
            yaml_file.write(self.decoder.to_yaml())
        yaml_file.close()

        self.discriminator.save_weights('discriminator'+suffix+'.h5')
        with open('discriminator'+suffix+'.yaml','w') as yaml_file:
            yaml_file.write(self.discriminator.to_yaml())
        yaml_file.close()

        self.classifier.save_weights('classifier' + suffix + '.h5')
        with open('classifier' + suffix + '.yaml', 'w') as yaml_file:
            yaml_file.write(self.classifier.to_yaml())
        yaml_file.close()


    def batch_generator(self,iterable1,iterable2,iterable3,batch_size=1,shuffle=False):
        l = len(iterable1)
        if shuffle ==True:
            indices = np.random.permutation(len(iterable1))
        else:
            indices = np.arange(0,stop=len(iterable1))
        for ndx in range(0,l,batch_size):
            cur_indices = indices[ndx:min(ndx+batch_size,l)]
            yield  iterable1[cur_indices],iterable2[cur_indices],iterable3[cur_indices]


    def training_loop(self,dataset,labels_one_hot_codes,positive_noise_set,batch_size):
        #batch generator
        self.data_generator = self.batch_generator(dataset,labels_one_hot_codes,positive_noise_set,batch_size=batch_size)
        nb_batches = len(dataset)/batch_size +1
        for motion_batch,labels_batch,prior_noise_batch in self.data_generator:
            self.iterations = self.iterations +1
            if self.is_annealing_beta is True:
                new_beta_value = K.sigmoid(self.beta_anneal_rate*
                                           self.iterations/(nb_batches+0.0)-
                                           self.bias_beta).eval()
                K.set_value(self.beta_classifier, new_beta_value)
            #print "iteration: {}".format(self.autoencoder_with_discriminator.optimizer.iterations.eval())
            cur_batch_size = len(motion_batch)
            latent_codes = self.encoder.predict(x=motion_batch,batch_size=cur_batch_size)
            X = np.concatenate((prior_noise_batch, latent_codes),axis=0)
            Y = [1.]*cur_batch_size+[0.]*cur_batch_size
            if self.train_disc ==True:
                self.discriminator.trainable=True
                self.discriminator.train_on_batch(x=X,y=Y)
            if self.train_classifier == True:
                self.classifier.trainable=True
                self.classifier.train_on_batch(x=latent_codes,y=labels_batch)
            self.classifier.trainable = False
            self.discriminator.trainable = False
            Y_hat = np.asarray([1.]*cur_batch_size,dtype=np.float32)
            self.autoencoder_with_discriminator.train_on_batch(x=[motion_batch,labels_batch],y={'Decoder':motion_batch,\
                                                    'Discmt':Y_hat,'Classifier':Y_hat})

    def fit(self,X,y=None):
        self.set_up_dataset(self.dataset_obj)
        self.set_up_model()
        self.init_loss_history_list()
        print "training set size: %d" % len(self.train_X)
        for epoch in range(self.max_epoch):
            print('Epoch seen: {}'.format(epoch))
            self.set_up_noise_examples()
            self.training_loop(self.train_X, self.train_label_one_hot_code, self.positive_examples_training, batch_size=self.batch_size)

        print "Finish this fitting process"
        return self

    def score(self,X,y=None):
        Y_hat = np.asarray([1.] * len(self.valid_X))
        loss_autoencoder, loss_mse_autoencoder, loss_crossentropy_encoder,loss_neg_entropy_encoder, metric_mse = \
            self.autoencoder_with_discriminator.evaluate(x=[self.valid_X,self.valid_label_one_hot_code],
                                                         y={'Decoder': self.valid_X,
                                                            'Discmt': Y_hat,
                                                            'Classifier':Y_hat},
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
    model = Double_GAN_Conditional_SAAE(latent_dim=50,latent_activation='tanh',latent_BN=False,
                                    hidden_dim_enc_list=[100,100],activation_enc_list=['tanh','tanh'],
                                    hidden_dim_dec_list=[100,100],activation_dec_list=['tanh','tanh'],
                                    hidden_dim_dis_list=[100,40],activation_dis_list=['relu','relu'],
                                    dropout_dis_list=[0.0,0.0],batch_size=20,max_epoch=100,
                                    optimiser_autoencoder='rmsprop',optimiser_dis='sgd',
                                    lr_autoencoder=0.001,lr_dis=0.01,decay_autoencoder=0.0,decay_dis=0.0,
                                    momentum_autoencoder=0.0,momentum_dis=0.0,
                                    prior_noise_type='Gaussian',
                                    loss_weights=[1.0,0.001,0.001],
                                    train_disc=True,
                                    train_classifier=True,
                                    custom_loss=True,symetric_autoencoder=False,
                                    nb_to_generate=10,
                                    condition_activity_or_emotion=1,
                                    nb_label=8,fully_condition=False,
                                    is_annealing_beta=True,
                                    beta_anneal_rate=0.1,
                                    bias_beta=9.)
    model.training(dataset_obj)


