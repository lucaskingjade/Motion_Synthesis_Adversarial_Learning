import matplotlib
from mpl_toolkits.mplot3d import Axes3D
print matplotlib.__version__
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111,projection='3d')
xs = range(10)
import pickle
ax.scatter(xs,xs,xs,color='y')
plt.legend(['codes_encoder'])
# #plt.show()
pickle.dump(fig, file('test.pickle', 'wb'))
# fg = pickle.load(file('test.pickle','rb'))
# print 'loading'
# import matplotlib
#
# fg.show()