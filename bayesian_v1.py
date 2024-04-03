#!/usr/bin/env python
# coding: utf-8


import os
from util import *
from logistic_np import *
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, InputLayer, LeakyReLU, BatchNormalization, Input
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils import plot_model
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from mlxtend.evaluate import proportion_difference

# In[2]:


# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[4]:


# Preprocess data
X_train, X_test = normalize_all_pixel(X_train, X_test)
X_train = reshape2D(X_train)
X_test = reshape2D(X_test)


# In[5]:


# Define encoder and decoder dimensions
encoder_dim = X_train.shape[1] // 2


# In[6]:


# Define Encoder model
Encoder = Sequential(
    [
        InputLayer(input_shape=X_train.shape[1:]),
        Dense(encoder_dim),
        BatchNormalization(),
        LeakyReLU(),
        Dense(encoder_dim),
        Dense(encoder_dim // 2),
    ],
    name="Encoder",
)
Encoder.summary()


# In[7]:


# Define Decoder model
Decoder = Sequential(
    [
        InputLayer(input_shape=(encoder_dim // 2,)),
        Dense(encoder_dim),
        BatchNormalization(),
        LeakyReLU(),
        Dense(encoder_dim),
        Dense(X_train.shape[1]),
    ],
    name="Decoder",
)
Decoder.summary()


# In[8]:


# Combine Encoder and Decoder to form Autoencoder model
inp = Input(X_train.shape[1:])
Encoderoutput = Encoder(inp)
Decoderoutput = Decoder(Encoderoutput)
AutoEncodermodel = Model(inputs=inp, outputs=Decoderoutput)
AutoEncodermodel.summary()


# In[13]:


# Compile Autoencoder model
AutoEncodermodel.compile(optimizer="adam", loss="mse")


# In[14]:


# Train Autoencoder model
callback = EarlyStopping(monitor="val_loss", patience=3)
hist = AutoEncodermodel.fit(
    X_train, X_train, epochs=100, validation_data=(X_test, X_test), callbacks=[callback]
)


# In[15]:


# Plot loss curves
plt.plot(range(1, len(hist.history["loss"]) + 1), hist.history["loss"], label="train")
plt.plot(
    range(1, len(hist.history["loss"]) + 1), hist.history["val_loss"], label="test"
)
plt.legend()
plt.show()


# In[16]:


# Save Encoder model
encoder = Model(inputs=inp, outputs=Encoderoutput)
encoder.compile()
if os.path.exists("encoder_v1.keras"):
    os.remove("encoder_v1.keras")  # Deletes the existing file
encoder.save("encoder_v1.keras")


# In[24]:


# Visualize Encoder model architecture
plot_model(encoder, "encoder_compress_v1.png", show_shapes=True)


# In[17]:


# Feature extraction using Encoder
feature_extract = load_model("encoder_v1.keras")
X_train_encoded = feature_extract.predict(X_train)
X_test_encoded = feature_extract.predict(X_test)


# In[26]:


gaussian = GaussianNB()
for i in range(100):
    gaussian.fit(X_train, y_train)
    print("Epoch: ", i)

y_pred = gaussian.predict(X_test)

print("Accuracy: %.2f%%" % (100 * accuracy_score(y_test, y_pred)))


# In[27]:


# Gaussian Naive Bayes classifier
gaussian_v2 = GaussianNB()
for i in range(300):
    gaussian_v2.fit(X_train_encoded, y_train)
    print("Epoch: ", i)


# In[28]:


# Evaluate Gaussian Naive Bayes classifier
y_pred_v2 = gaussian_v2.predict(X_test_encoded)
accuracy_v2 = accuracy_score(y_test, y_pred_v2)
print("Accuracy: %.2f%%" % (100 * accuracy_v2))


# In[29]:


# Logistic Regression classifier on original data
model = LogisticRegression(multi_class="ovr")
model.fit(X_train, y_train)
yhat = model.predict(X_test)
acc_original = accuracy_score(y_test, yhat)
print(f"Accuracy score (Original data): {acc_original}")


# In[30]:


# Logistic Regression classifier on encoded data
model = LogisticRegression(multi_class="ovr")
model.fit(X_train_encoded, y_train)
yhat = model.predict(X_test_encoded)
acc_encoded = accuracy_score(y_test, yhat)
print(f"Accuracy score (Encoded data): {acc_encoded}")


# In[31]:


from sklearn.metrics import precision_score, f1_score

# Calculate Precision and F1 score for Gaussian Naive Bayes
precision_nb = precision_score(y_test, y_pred_v2, average="weighted")
f1_nb = f1_score(y_test, y_pred_v2, average="weighted")

print("Gaussian Naive Bayes:")
print(f"Precision: {precision_nb:.4f}")
print(f"F1 Score: {f1_nb:.4f}")

# Calculate Precision and F1 score for Logistic Regression
precision_lr = precision_score(y_test, yhat, average="weighted")
f1_lr = f1_score(y_test, yhat, average="weighted")

print("\nLogistic Regression:")
print(f"Precision: {precision_lr:.4f}")
print(f"F1 Score: {f1_lr:.4f}")
