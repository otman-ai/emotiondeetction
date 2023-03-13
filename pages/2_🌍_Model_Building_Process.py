import streamlit as st
st.set_page_config(page_title="Model building Process",layout="wide",
page_icon="🌍")
st.title("Emotion Classification")

st.info("[Emotion classification](https://en.wikipedia.org/wiki/Emotion_recognition) Emotion recognition is the process of identifying human emotion. People vary widely in their accuracy at recognizing the emotions of others. Use of technology to help people with emotion recognition is a relatively nascent research area")
st.info("**What is computer vision?**:\n well according to [wikipedia](https://en.wikipedia.org/wiki/Computer_vision) Computer vision tasks include methods for acquiring, processing, analyzing and understanding digital images, and extraction of high-dimensional data from the real world in order to produce numerical or symbolic information")

st.markdown("<h2>Model Building Processe</h2>",unsafe_allow_html=True)

st.write("Build ML model is one of the most hard and fun things to do . I went throught several steps to make this model alive .")
st.warning("This Steps can fit to any ML project.")

st.markdown("<h4>1.Data Collection</h2>",unsafe_allow_html=True)
st.write("Using pre-collected data from [Kaggle](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer).The dataset contain 35,685 examples of 48x48 pixel gray scale images of faces divided into train and test dataset. Images are categorized based on the emotion shown in the facial expressions (happiness, neutral, sadness, anger, surprise, disgust, fear).")

st.markdown("<h4>2.Build the model </h2>",unsafe_allow_html=True)
st.write("Ml is all about experiment ,Model tweaking, regularization, and hyperparameter tuning this is where we iteratively go from a ~good enough~ model to our best effort.In this step I have build three differents models **(differnets layers structure , number and the change on some prametres)**")

st.markdown("<h4>3.Evaluate the model performance</h2>",unsafe_allow_html=True)
st.write("In each model we evaluate the model performance to see how it behave against unseen data,we Uses some metric like *F1 score* , *Recall*, *Precision*")

st.markdown("<h4>4.Download the models on our local directory for Deployment uses <h4>",unsafe_allow_html=True)
st.write("Save the models on our local directory")

st.markdown("<h4>5.Test the model on random Image from the web</h4>",unsafe_allow_html=True)
st.write("Pick up any random image from the web and we test to see on real world the performance of our models")

st.markdown("<h4>6.Deploy our model</h4>",unsafe_allow_html=True)
st.write("We deploy the models using library for data science deployment in python called [Streamlit](https://streamlit.io/),it makes easy to build and deploy web app ")
# TO DO