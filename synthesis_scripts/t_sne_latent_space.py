from sklearn.manifold import TSNE
from keras.layers import Input,LSTM,merge,Dense
from keras.models import Model
from Seq_AAE_V1.datasets.EmilyaDataset.src.get_dataset import get_dataset


def plot_latent_space( latent_codes, filename, Y1=None, Y2=None, dim_x=0, dim_y=1):
    if Y1 == None:
        Y1 = np.zeros(len(latent_codes), dtype=np.float32)
    if Y2 == None:
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

def encoder(max_len,dof,hidden_dim_enc_list=[100,100],activation_enc_list=['tanh','tanh'],
            latent_dim=50,speed_dim=1):

    motion_input = Input(shape=(max_len, dof), name='encoder_input')
    speed_input = Input(shape=(max_len, speed_dim), name='speed_input1')
    encoded = merge([motion_input, speed_input], mode='concat')
    for i, (dim, activation) in enumerate(zip(hidden_dim_enc_list, activation_enc_list)):
        encoded = LSTM(output_dim=dim, activation=activation, return_sequences=True)(encoded)

    encoded = LSTM(output_dim=latent_dim, activation=activation, name='encoded_layer',
                           return_sequences=False)(encoded)
    encoded = Dense(output_dim=latent_dim, activation='linear')(encoded)

    return Model(input=[motion_input, speed_input], output=encoded, name='Encoder')



Actor_name = ['Brian', 'Elie', 'Florian', 'Hu', 'Janina', 'Jessica', 'Maria', 'Muriel', 'Robert', 'Sally',
                  'Samih', 'Tatiana']

h5_handle, filesource = get_dataset(actor_list=Actor_name,window_width=200,shift_step=20,sampling_interval=None)

X = h5_handle['test_set']['X'][:]
Y1 = h5_handle['test_set']['Y1'][:]
Y2 = h5_handle['test_set']['Y2'][:]
Y3 = h5_handle['test_set']['Y3'][:]
speed_xz = h5_handle['test_set']['speed_xz']


max_len = X[0].shape[-2]
dof = X[0].shape[-1]
encoder = encoder(max_len=max_len, dof=dof,
                      hidden_dim_enc_list=[100, 100], activation_enc_list=['tanh', 'tanh'],
                      latent_dim=50, speed_dim=1)

encoder.load_weights('./encoder.h5')

latent_codes = encoder.predict(x=[X,speed_xz],batch_size=2000)
import numpy as np
latent_codes = np.asarray(latent_codes)
print "shape of latent_codes is {}".format(latent_codes.shape)
print "finishing predicting ..."
del X
tsne = TSNE(n_components=2,perplexity=30,n_iter=1000,verbose=2)
print "begin to fit model"
np.set_printoptions(suppress=True)
new_X = tsne.fit_transform(latent_codes)
print "finishing fitting"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plot_latent_space(new_X,filename='t_sne_Y1.png',Y1=Y1)

plot_latent_space(new_X,filename='t_sne_Y2.png',Y2=Y2)

plot_latent_space(new_X,filename='t_sne_Y1Y2.png',Y1=Y1, Y2=Y2)
