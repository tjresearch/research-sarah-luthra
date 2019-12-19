# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
# import pandas as pd
import numpy as np
import glob
# import cv2
import os
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras import losses
from keras import backend
from keras.optimizers import *


# def load_attributes(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	#cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
x_list = [[4189880000, 104.2709122, 62.044, 0.481477902796539, 18.8], [4.38*10**9, 99.2709122, 60.05, 0.481477902796539, 18.9], [4817542204, 106.270912, 60.05, 0.481477902796539, 19],
			[8921947100, 106.2439122, 60.05, 0.481477902796539, 19.4], [9586327800, 106.2709122, 64, 0.481477902796539, 19.9], [10221705900, 106.2709122, 64.858, 0.4978081383655220, 0.1],
			[10936669900, 106.9709122, 65.617, 0.49780813836552, 20.1], [11284197000, 107.2709122, 66.273, 0.593754668022413, 21], [12282533600, 107.2709122, 66.824, 0.61270995101289, 23],
			[12282533600, 104.2702392, 66.824, 0.69125961186978, 24], [12664190300, 104.2709122, 67.978, 0.826108938299893, 25], [13243892200, 104.2709122, 68.284, 0.8931984863021, 26],
			[17601616000, 104.2926422, 68.284, 0.803010858817794, 27], [18447920000, 104.397473, 68.875, 0.952760253408986, 30], [20283780000, 111.565322875977, 69.162, 1.00248889011542, 31],
			[21386160000, 78.9535903930664, 69.432, 0.974964609846008, 33], [21990970000, 112.659187316895, 69.68, 0.975303060359352, 34], [2.22*10**10, 113.066276550293, 69.908, 1.00352083328412, 34.6],
			[22200000000, 112.550956726074, 70.124, 1.09323403753405, 40], [22300000000, 113.056793212891, 70.333, 1.09323403753405, 37.8], [22400000000, 113.822570800781, 70.542, 1.05679391254387, 40],
			[2240000000, 115.914848327637, 70.978, 1.06638943061798, 36.5], [22500000000, 115.25599670410, 71.21, 1.12614569301998, 40.6], [22400000000, 115.488296508789, 71.449, 1.14289520964458, 34.5],
			[22500000000, 114.697570800781, 71.692, 1.06748971421601, 29.6], [22700000000, 112.92505645752, 71.935, 1.04632145906758, 31.8], [22593470000, 110.210678100586, 72.175, 1.04426221422652, 34.9],
			[23438240000, 106.765419006348, 72.126, 1.07047712866198, 32.7], [24154120000, 103.112602233887, 74.644, 1.06813421711041, 29.2], [24927970000, 100.225341796875, 75.872, 0.994280785244978, 29.8]
			]
y_list = [1024, 882, 1035, 1136, 1203, 1114, 847, 871, 981, 1172, 1591, 1838, 1918, 1883, 1716, 1111, 801, 919, 3098, 3432, 3288, 3227, 3170, 4706, 6037, 7208, 8256, 11273, 14331, 18763]

x_list = np.asarray(x_list, dtype=float)
y_list = np.asarray(y_list, dtype=float)

model = Sequential()
regress = False;
model.add(Dense(5, input_shape=(5,), activation="sigmoid"))
model.add(Dense(3, activation="sigmoid"))
model.add(Dense(1, activation="linear"))



x_test = np.array([[26056950000, 96.5469436645508, 74.872, 0.998428943780015, 30.4]], dtype=float)
y_test = np.array([2212], dtype=float)

opt = RMSprop(lr=1)

model.compile(loss='mean_squared_error', optimizer=opt, metrics = ['accuracy'])
model.fit(x_list, y_list, epochs=100, batch_size=1)

print("prediction: \n\n")
print(model.predict(x_test))
# accuracy = model.evaluate(x_list, y_list)
# print('Accuracy: %.2f' % (accuracy*100))

# return model
