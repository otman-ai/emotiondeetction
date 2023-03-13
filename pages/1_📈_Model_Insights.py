import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly import graph_objs as go
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix , classification_report 
st.set_page_config(page_title="Models inshight", page_icon="📈",layout="wide")

def load_model_insight_(df):
    data = pd.read_csv(df)
    return data

st.title("Models Insight")
labels = np.array(("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"))
model_matrix = pd.read_csv("model_matrix.csv")
model_matrix.columns=["matrix","Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral","accuracy","macro avg","weighted avg"]

st.markdown("<h4>Model</h4>",unsafe_allow_html=True)
model_matrix
st.markdown("<h5>Visualization the matrix :</h5>",unsafe_allow_html=True)
fig = px.bar(model_matrix.loc[model_matrix["matrix"] != "support"],y=labels,x="matrix",
              barmode="group",width=900,height=350,title="The matrix of all the classes")

st.plotly_chart(fig,use_container_width=False)
st.markdown("<h5>Confustion matrix</h5>",unsafe_allow_html=True)
cm_data = np.array([[77, 63, 54, 71, 78, 70, 78],
       [65, 62, 45, 67, 77, 91, 84],
       [71, 65, 58, 78, 70, 76, 73],
       [95, 64, 50, 76, 71, 73, 62],
       [80, 67, 57, 91, 73, 60, 63],
       [68, 56, 47, 79, 67, 90, 84],
       [88, 54, 51, 81, 81, 72, 64]])
cm = pd.DataFrame(cm_data, columns=labels, index = labels)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
fig = px.imshow(cm,text_auto=True,aspect="auto")
st.plotly_chart(fig)
st.markdown("<h4>Tensorboard</h4>",unsafe_allow_html=True)
st.write("You can check the model training performance [here](https://tensorboard.dev/experiment/yxPyb407RQG2W2RxvEwBgQ/)")
