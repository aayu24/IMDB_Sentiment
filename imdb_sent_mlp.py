
# coding: utf-8

# In[1]:


import numpy as np
from keras.datasets import imdb
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
seed = 7
np.random.seed(seed)


# In[2]:


top_words = 5000 #Vocab size for embedding
(X_train, y_train), (X_test,y_test) = imdb.load_data(num_words=top_words)
max_words = 500 #upper limit for review length
X_train = sequence.pad_sequences(X_train,maxlen=max_words)
X_test = sequence.pad_sequences(X_test,maxlen=max_words)


# In[3]:


#MLP model
model = Sequential()
model.add(Embedding(top_words,32,input_length=max_words))
model.add(Flatten())
model.add(Dense(250,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[4]:


#fit model
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=2, batch_size=128,verbose=2)
scores = model.evaluate(X_test,y_test,verbose=0)
print("Accuracy: %2f%%" % (scores[1]*100))
print(scores)

