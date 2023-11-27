import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, Adam
import pickle
import tensorflow as tf

dataset = pd.read_csv(r"DL-ALGORITHMS\SMSSpamCollection.txt",sep='\t',names=['label','message'])

dataset['label'] = dataset['label'].map( {'spam': 1, 'ham': 0} )
X = dataset['message'].values
y = dataset['label'].values

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tokeniser = tf.keras.preprocessing.text.Tokenizer()
tokeniser.fit_on_texts(X_train)
encoded_train = tokeniser.texts_to_sequences(X_train)
encoded_test = tokeniser.texts_to_sequences(X_test)
max_length = 10
padded_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_train, maxlen=max_length, padding='post')
padded_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')

n_features = padded_train.shape[1]
output_class = 1

#Modelling a sample DNN
model = Sequential()
model.add(Dense(64, activation='relu',input_shape=(n_features,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(output_class,activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


with open("spam_dnn_model.pkl", 'wb') as model_file:
    pickle.dump(model, model_file)