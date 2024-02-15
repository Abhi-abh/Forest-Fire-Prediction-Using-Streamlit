import pandas as pd
import numpy as np
import streamlit as st
import pickle
import sklearn
from PIL import Image

model = pickle.load(open('model.sav', 'rb'))

st.title('Forest fire Prediction')
st.sidebar.header('Forest Fire Data')

image1 = Image.open('forest fire.jpg')
image2 = Image.open('forest-pictures.jpg')

# FUNCTION
def user_report():
  
  FFMC = st.sidebar.slider('FFMC', 0.0,100.0 )
  ISI = st.sidebar.slider('ISI', 0.0,100.0 )
  FWI = st.sidebar.slider('FWI', 0.0,100.0)
  


  user_report_data = {
      
      'FFMC': FFMC,
      'ISI':ISI,
      'FWI':FWI,
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.header('Forest Fire Data')
st.write(user_data)

clas = model.predict(user_data)
if clas != 1:
  result="Not Fire"
  result1=st.image(image2, '')
else:
  result="Fire"
  result1=st.image(image1, '')
st.subheader('Fire Prediction')
st.subheader(result)
