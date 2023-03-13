import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow as tf
import plotly.express as px

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    layout="wide"

)
#load the model
@st.cache_resource
def load_model_(model_path):
    model = load_model(model_path, compile=False)
    return model

#Prepare the image uploaded 
def prepare_img(filepath):
    img_ = cv2.imread(filepath)
    resize = tf.image.resize(img_, (48,48))
    return np.expand_dims(resize/255, 0)

labels = np.array(("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"))
labels_dict={}
for i,label,in enumerate(labels):
    labels_dict[i] = label

#color_dict={0:(0,0,255),1:(0,255,0)}
st.title("Emotion Classification")
st.header("Hi their ! You can detect the emotion of a given image")

model = load_model_("model.h5")
model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=["accuracy"])
image_streamlit  = st.file_uploader("Upload your image",type=["jpg","png","jpeg"])
col1, col2= st.columns((1,1),gap="medium")

if image_streamlit != None:
    with col1:
        st.image(image_streamlit)
    with open("files/image.jpg",mode = "wb") as f: 
        f.write(image_streamlit.getbuffer())      
    st.success("Saved File")
    image = cv2.imread("files/image.jpg")
    #Setup the file path and the algo path
    alg = "haarcascade_frontalface_default.xml"
    haar_cascade = cv2.CascadeClassifier(alg)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # loop through all the faces in the image and get the crop face of each person then predict it take the crop face then put the predict label
    for i,(x, y, w, h) in enumerate(faces):
        face_crop = image[y:y+h, x:x+w]
        pict_name = 'cut_face.jpg'
        cv2.imwrite(pict_name, face_crop)
        pred = model.predict(prepare_img(pict_name))
        class_ = np.argsort(pred,axis=1,)[0]
        print(labels[class_[0]],pred[0][class_[0]])
        print(labels[class_[1]],pred[0][class_[1]])
        print(labels[class_[2]],pred[0][class_[2]])
       
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 3)
        cv2.rectangle(image, (x, y - 24), (x + w, y), (0,0,0) ,-3)
        cv2.putText(image, f'{labels[np.argmax(pred,axis=1)[0]]}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,255,0), 2)
    cv2.imwrite("imageDetect.jpg", image)
    with col2:
        st.image("imageDetect.jpg")
    st.markdown("<h5>The Prodiction Probabilities:</h5>",unsafe_allow_html=True)
    fig_ = px.bar(x=[labels[i] for i in class_],
              y=[pred[0][i] for i in class_])
    st.plotly_chart(fig_,use_container_width=True)
# TO DO: ADD camera future