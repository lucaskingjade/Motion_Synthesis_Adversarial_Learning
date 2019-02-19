import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('filename', help='specify the emotion name that you want to animate', type=str)
args = parser.parse_args()
print args.filename
fig = pickle.load(file(args.filename, 'rb'))
fig.show()
