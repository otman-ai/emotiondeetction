import streamlit as st
st.set_page_config(page_title="About the us", page_icon="📊",layout="wide")

st.title("About us")

st.write("I am just man who want to create safe intelligent systems  that can make our life easier and more productive.")
st.write("The project repo in my [github]() ") # TO DO : put the ulr of github 
col1,col2,col3 ,_,_,_,_,_,_,_,_,_,_,_,_,_,_= st.columns((1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1))
with col1:
    st.markdown("[<div><img src='https://cdn-icons-png.flaticon.com/512/174/174855.png' width=30 /></div>](https://www.instagram.com/otman_heddouch/)", unsafe_allow_html=True)
with col2:
    st.markdown("[<div><img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width=30 /></div>](https://www.linkedin.com/in/otman-heddouch-723714215/)", unsafe_allow_html=True)
with col3:
    st.markdown("[<div><img src='https://cdn-icons-png.flaticon.com/512/25/25231.png' width=30 /></div>](https://github.com/otman-ai)", unsafe_allow_html=True)
