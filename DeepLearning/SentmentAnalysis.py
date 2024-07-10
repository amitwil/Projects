import nltk

from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

import re
import pandas as pd

# #Data preprocessing
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

df = pd.read_csv(r'C:\Users\amitw\Downloads\reviews.csv', encoding='ISO-8859-1')

df.isnull().sum()
print(df.shape)

X = df.drop('Sentiment',axis = 1)
y = df['Sentiment']

df.reset_index(inplace = True) # dropped null values so need to reset index 

## using Stemming
corpus = []
for i in range(0,len(X)):
    rvw = re.sub('[^a-zA-Z]', ' ',X['Text'][i])
    rvw = rvw.lower()
    rvw = rvw.split()
    rvw = [ps.stem(words) for words in rvw if not words in stopwords.words('english')]
    rvw = [lemmatizer.lemmatize(words) for words in rvw if words not in set(stopwords.words('english'))]
     
    rvw = ' '.join(rvw)
    corpus.append(rvw)
print('DONE')

## onehot
from tensorflow.keras.preprocessing.text import one_hot
voc_size=10000
# one hot encoding
ohe_rpr = [one_hot(words,voc_size) for words in corpus]

from tensorflow.keras.preprocessing.sequence import pad_sequences
#Embedding Representaion
sent_length = 20
emb = pad_sequences(ohe_rpr,padding = 'pre',maxlen = sent_length)
# emb = emb.reshape(emb.shape[0], emb.shape[1], 1)

from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, Dropout

model = Sequential([
    Input(shape=(20,)),
    Embedding(input_dim=voc_size, output_dim=40),  # Example input dimensions
    Dropout(0.5),
    # Dense(10, activation='relu'),
    LSTM(400, return_sequences=True),
    Dropout(0.5),
    LSTM(300),
    # Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.summary()

import numpy as np
x_fin = np.array(emb)
y_fin = np.array(y)

from sklearn.model_selection import train_test_split
xtr,xtst,yrt,ytst = train_test_split(x_fin,y_fin,test_size=0.33,random_state=42)

history=model.fit(xtr,yrt,validation_data=(xtst,ytst),epochs = 10,batch_size = 32)

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

history.history.keys()
history.history.items()
import matplotlib.pyplot as plt
plot_history(history)

yp = model.predict(xtst)
pp = np.where(yp>=0.5,1,0)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
confusion_matrix(ytst,pp)

accc = accuracy_score(ytst,pp)
print('Accuracy',accc)