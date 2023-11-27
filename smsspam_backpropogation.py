import pandas as pd
from sklearn.model_selection import train_test_split
from BackPropogation import  BackPropogation
import tensorflow as tf
import pickle

dataset = pd.read_csv(r'C:\Users\SRUTHI\OneDrive\Desktop\Github Folder\DL-Algorithms\SMSSpamCollection.txt',sep='\t',names=['label','message'])
dataset['label'] = dataset['label'].map( {'spam': 1, 'ham': 0} )
X = dataset['message'].values
y = dataset['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tokeniser = tf.keras.preprocessing.text.Tokenizer()
tokeniser.fit_on_texts(X_train)
encoded_train = tokeniser.texts_to_sequences(X_train)
encoded_test = tokeniser.texts_to_sequences(X_test)
print(encoded_train[0:2])

max_length = 10
padded_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_train, maxlen=max_length, padding='post')
padded_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')

backpropogation = BackPropogation(epochs=10)

backpropogation.fit(padded_train, y_train)
pred = backpropogation.predict(padded_test)

with open("backpropogation_smsspam.pkl",'wb') as model_file:
    pickle.dump(backpropogation, model_file)