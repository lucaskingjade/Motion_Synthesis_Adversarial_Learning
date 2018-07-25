from keras.callbacks import Callback
import keras.backend as K
class set_alpha_rate(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        new_alpha_value = K.sigmoid(
            self.alpha_rate * (epoch * - self.begin_tune_alpha_epoch - (self.alpha_bias / self.alpha_rate))).eval()
        K.set_value(self.alpha, new_alpha_value)