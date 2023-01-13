import streamlit as st
import requests as re
import librosa, librosa.display
import numpy as np
from Model import model, vgg
import matplotlib as plt
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import noisereduce as nr

st.set_option('deprecation.showPyplotGlobalUse', False)

def get_features(inp):
    X, sr = librosa.load(inp, res_type='kaiser_fast')
    feature = []
    S = librosa.feature.melspectrogram(y=X[:50000], sr=16000, n_mels=13)
    spec = librosa.power_to_db(S, ref=np.max)
    feature.append(spec)
    inputs = np.stack(tuple(feature))
    return inputs

# def imagePreprocess(inp):
#     data = image(inp)
#     test_data = ImageDataGenerator(rescale=1./255)
#     resized_img = tf.image.resize(
#         data,
#         (224,224)
#     )
    # test_images = test_data.flow_from_directory(
    #     data,
    #     target_size=(224,224),
    #     color_mode='rgb',
    #     class_mode="binary",
    # )
    # return resized_img

def image(inp):
    # fig, ax = plt.subplot()
    byteCode, sr = librosa.load(inp, res_type='kaiser_fast',sr=16000)
    reduced_sig = nr.reduce_noise(byteCode,sr)
    melspec = librosa.feature.melspectrogram(reduced_sig)
    log_mel_spec = librosa.amplitude_to_db(melspec)
    img = librosa.display.specshow(log_mel_spec)
    st.pyplot()
    return img

def main():
    st.title("Gender recognition using speech signals")
    choice = st.sidebar.selectbox("Models",["Vanilla CNN","VGG"])
    file_uploader = st.file_uploader(label="", type=".wav")
    # if file_uploader is not None:
    #     st.write(file_uploader)
        # audio_bytes = file_uploader.read()
    if choice=="Vanilla CNN":
        inp= get_features(file_uploader)
        if st.button("Classify"):
            p = model.predict(inp)
            if p>=0.497:
                st.write("Male")
            else:
                st.write("Female")
            st.audio(file_uploader,format='audio/ogg')
    elif choice=="VGG":
        # inps = imagePreprocess(file_uploader)
        if st.button("Classify"):
            st.write("Spectrogram image")
            image(file_uploader)
            # p = vgg.predict(inps)
            # if p>=0.497:
            #     st.write("Male")
            # else:
            #     st.write("Female")
        # response = re.get(f"http://127.0.0.1:8000/predict?_file={str(file_uploader)}").text
        # # response = response.json()
        # st.write(response)

    # if choice=="Vanilla CNN"

if __name__ == '__main__':
    main()