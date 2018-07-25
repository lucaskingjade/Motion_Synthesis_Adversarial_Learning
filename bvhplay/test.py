#test skeleton.py
import skeleton
import numpy as np
from sklearn import preprocessing

bvh = skeleton.readbvh('./bvhFiles/Ax2CS_Brian.bvh')
bvh.read()
keyframes = bvh.keyframes
num_frames = bvh.frames
keyframes=np.asarray(keyframes,dtype=np.float32)
r,c = keyframes.shape
print 'The shape of keyframes is [{},{}]'.format(r,c)
#normalize each column to make each column have the mean of zero and variance of 1
mean_posture = np.mean(keyframes,axis=0)
deviation = np.std(keyframes,axis=0)
keyframes_norm = preprocessing.scale(keyframes,axis=0)








