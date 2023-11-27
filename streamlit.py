
import streamlit as st
from model_define import BackPropogation, Perceptron
import pickle 
from PIL import Image 
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import argmax
from sklearn.model_selection import train_test_split

class Basic_functions:
    def upload_image():
        input_data = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
        if input_data is not None:
            file_bytes = np.asarray(bytearray(input_data.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            return opencv_image


    def open_model(model_name):
        load_model = open(model_name, 'rb') 
        model = pickle.load(load_model)
        return model
    


    def pred(input_data,model,model_name):

        if model_name == 'CNN_tumor.pkl':

            img=Image.fromarray(input_data)
            img=img.resize((128,128))
            img=np.array(img)
            input_img = np.expand_dims(img, axis=0)
            res = model.predict(input_img)
            if res:
                st.write("Tumor Detected")
            else:
                st.write("No Tumor")




        if model_name == 'RNN_smsspam1.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 10
            encoded_test = tokeniser.texts_to_sequences(input_data)
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = (model.predict(input_data) > 0.5).astype("int32")
            if preds[0] == [0]:
                st.write('The given message is ham')

            elif preds[0]== [1]:
                st.write('The given message is spam')


        if model_name == 'simple_lstm.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 50
            encoded_test = tokeniser.texts_to_sequences([input_data])
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = (model.predict(input_data) > 0.5).astype("int32")
            if preds[0] == [0]:
                st.write('The given message is ham')

            elif preds[0]== [1]:
                st.write('The given message is spam') 



        if model_name == 'spam_dnn_model.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 10
            encoded_test = tokeniser.texts_to_sequences([input_data])
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = (model.predict(input_data) > 0.5).astype("int32")
            if argmax(preds)==0:
                st.write('The given message is ham')

            elif argmax(preds)==1: 
                st.write('The given message is spam')   



        if model_name == 'backpropogation.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 10
            encoded_test = tokeniser.texts_to_sequences([input_data])
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = model.predict([input_data])
            if preds==[0]:
                st.write('The given message is ham')

            elif preds==[1]: 
                st.write('The given message is spam')



        if model_name == 'perceptron_spam_model.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 10
            encoded_test = tokeniser.texts_to_sequences([input_data])
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = model.predict([input_data])
            if preds==[0]:
                st.write('The given message is ham')

            elif preds==[1]: 
                st.write('The given message is spam') 

def main():

    st.title("Deep Learning MODEL")     

    option = st.selectbox("Choose an Option",('Tumor prediction','Sentiment analysis'),index=None,placeholder="Select a method",)

    st.write('You selected:', option)

    if option == None:
        pass

    elif option == 'Tumor prediction':

        
        st.write('Upload tumor data')
        model_name = 'CNN_tumor.pkl'
        
        input_data = Basic_functions.upload_image()
        if input_data is not None:
            st.image(input_data, channels="BGR")

        but = st.button("Predict", type="primary")    
        if but:
            model = Basic_functions.open_model('CNN_tumor.pkl')
            Basic_functions.pred(input_data,model,model_name)       
        
    elif option == 'Sentiment analysis':
        out =  st.radio(
            "Select your prediction",
            key="visibility",
            options=["Recurrent Neural Network", "LSTM", "DEEP NEURAL NETWORK", "Back propagation", "Perceptron"],)  

    

        if out == "Recurrent Neural Network":
            model_name = 'RNN_smsspam1.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name)


        elif out == "LSTM":
            model_name = 'simple_lstm.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name) 

        elif out == "DEEP NEURAL NETWORK": 
            model_name = 'spam_dnn_model.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name)


        elif out == "Back propagation":
            model_name = 'backpropogation_smsspam.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name)




        elif out == "Perceptron": 
            model_name = 'perceptron_smsspam.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name)



if __name__ == "__main__":
    main()           


