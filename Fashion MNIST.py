#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mnist_reader
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras


# In[2]:


X_train, y_train = mnist_reader.load_mnist('data', kind='train')
X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')


# In[4]:


X_train = (X_train.reshape(-1,28,28))
X_test = (X_test.reshape(-1,28,28))
plt.imshow(X_train[0])
plt.colorbar()


# In[5]:


def get_model(x):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(x, activation='relu'),
        keras.layers.Dense(10)
    ])
    return model


# In[6]:


models = []
models_trained = []
sizes = [10,20,50,100,128,256,512,784]
for i in sizes:
    models.append(get_model(i))

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt =tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999)

for i in range(len(models)):
    models[i].compile(opt, loss,validation_split=0.2, metrics=['accuracy'])
    models_trained.append(models[i].fit(X_train, y_train,validation_split=0.2, epochs=5))


# In[7]:


for i in range(len(models)):
    plt.plot(models_trained[i].history['val_accuracy'])
plt.ylabel('val_accuaracy')
plt.xlabel('epochs')
plt.legend(sizes)
plt.title("choosing hidden layer size")


# In[8]:


for i in range(len(models)):
    print(i, max(models_trained[i].history['val_accuracy']))


# In[9]:


models = []
models_trained = []
rates = [0.0001,0.0002,0.0005, 0.001, 0.002, 0.005]
for i in range(len(rates)):
    models.append(get_model(sizes[5]))
    
for i in range(len(models)):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt =tf.keras.optimizers.Adam(
    learning_rate=rates[i], beta_1=0.9, beta_2=0.999)
    models[i].compile(opt, loss, metrics=['accuracy'])
    models_trained.append(models[i].fit(X_train, y_train, validation_split=0.2, epochs=10))


# In[10]:


for i in range(len(models)):
    plt.plot(models_trained[i].history['val_accuracy'])
plt.ylabel('accuaracy')
plt.xlabel('epochs')
plt.legend(rates)
plt.title("choosing learning rate")


# In[11]:


for i in range(len(models)):
    print(i, max(models_trained[i].history['val_accuracy']))


# In[17]:


model = get_model(sizes[5])

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt =tf.keras.optimizers.Adam(
learning_rate=rates[1], beta_1=0.9, beta_2=0.999)
model.compile(opt, loss, metrics=['accuracy'])
models_trained.append(model.fit(X_train, y_train, epochs=20))


# In[18]:


predictions = model.predict(X_test)
acc = 0
for i in range(len(y_test)):
    if np.argmax(predictions[i]) == y_test[i]:
        acc += 1
print("accuracy on test data:",acc/len(y_test))


# In[19]:


model.summary()


# In[22]:


model.save('my_model') 


# In[ ]:




