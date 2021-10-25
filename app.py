# Import necessaty Libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import base64

from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

st.title('Disaster Message Classifier')
img = Image.open("/Users/suelemlee/Desktop/portfolio/ppt/58274551_303.jpeg")
st.image(img)

# Load your data to Classify Multiouput predictions category by it's appropriate column
@st.cache
def load_data():
    df = pd.read_csv('./datasets/df_clean.csv')
    df.drop(columns = ['Unnamed: 0', 'content_length', 'content_word_count', 'genre', 'related', 'PII'], inplace = True)
    return df

def tokenize_correct_spelling(text):
    
    textBlb = TextBlob(text)            # Making our first textblob
    textCorrected = textBlb.correct()   # Correcting the text
    
    tokens = word_tokenize(str(textCorrected))
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()      
        clean_tokens.append(clean_tok)

    return clean_tokens

# Functions of Output the category names
def get_category_names(df):
    return list(df.columns[1:])

def get_predicted_category_names(category_predicted):
    return [category_names[i] for i in range(len(category_predicted)) if category_predicted[i] == 1]

def list_to_string(s): 
    return (" ".join(s))

# Add description to what the app does
st.subheader('About this project')
st.write('''
    Social media is being explored as tool for disaster management by developers, 
    researchers, government agencies and businesses. \n
    The disaster-affected area requires both cautionary and disciplinary measures. 
    The need for decision-making system during emergencies and in real time poses problems 
    classifying emergencies. \n
    **Can we explore Social Media tools during times of crisis to help people make more informed and better decisions?**
''')

# Load your favorite pickle model
with open('models/multiout_adaboost.pkl', 'rb') as pickle_in:
    pipe = pickle.load(pickle_in)

category_names = list(load_data().columns[1:])
your_text = st.text_input('Tell us what is happening:', max_chars=500)

# Predict Classification 
category_predicted = pipe.predict([your_text])[0]
result = list_to_string(get_predicted_category_names(category_predicted))

fire = 'fire'
important_categories = ['request', 'search_and_rescue', 'death', 'weather_related', 'floods', 'fire', 'earthquake']

# Load video of Fire location

file_ = open("/Users/suelemlee/Desktop/portfolio/imgs/n_audio_fire.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

# Classification of the message provided
if st.button("Classify Message"):
    if fire in result:

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="fire gif">',
            unsafe_allow_html=True,
        )
        st.subheader("Please confirm your location in affected area")
        page_names = ['Yes! Im here! I need help', 'No, go help someone else.']

        page = st.radio('Emergency status', page_names, index=0)

        if page == 'Yes! Im here! I need help':
            st.subheader("We are on our way!")

        else:
            st.subheader("Thank you, stay safe.")

    else:
        st.write(f'Your message Classifies as -  {result.title()} disaster.')
