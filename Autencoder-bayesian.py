import pandas as pd
import numpy as np
from util import *
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import *
from keras.utils import plot_model

train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data()
print(train_x[100:105, :])
# scale data

t = MinMaxScaler()
t.fit(train_x)
X_train = t.transform(train_x)  # dimension unchanged
X_test = t.transform(train_x)
Encoder = Model
