import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from preprocess import preprocessing
from streamlit_lottie import st_lottie
import json
##Helper functions


def load_pickle_files(file_name):
    with open(file_name,"rb") as f:
        obj = pickle.load(f)

    return obj


def get_lottie_files(file_path):
    with open(file_path,"r") as f:
        return json.load(f)


def lottie_emoji(lottie_file):
    st_lottie(
        lottie_file
    )

##get emoji for emotion


def get_emoji(emotion):
    if emotion == "joy":
        lottie_file = get_lottie_files("lottie_animations/joy_emoji.json")
        lottie_emoji(lottie_file)
    elif emotion == "sadness":
        lottie_file = get_lottie_files("lottie_animations/sad_emoji.json")
        lottie_emoji(lottie_file)
    elif emotion == "anger":
        lottie_file = get_lottie_files("lottie_animations/anger_emoji.json")
        lottie_emoji(lottie_file)
    elif emotion == "fear":
        lottie_file = get_lottie_files("lottie_animations/fear_emoji.json")
        lottie_emoji(lottie_file)
    else:
        lottie_file = get_lottie_files("lottie_animations/love_emoji.json")
        lottie_emoji(lottie_file)



#############

st.title("Don't Know how you feel?")

input_text = st.text_input(label="Tell us....",placeholder="write here")

##load model
classifier_name = st.sidebar.selectbox("Select classifier",
                             ("Logistic Regression","KNN","Decision Tree","SVM"))


def get_classifier(classifier_name):
    if classifier_name == "Logistic Regression":
        model = load_pickle_files("logistic_reg.pkl")
    elif classifier_name == "KNN":
        model = load_pickle_files("knn.pkl")
    elif classifier_name == "Decision Tree":
        model = load_pickle_files("dec_tree.pkl")
    else:
        model = load_pickle_files("svm.pkl")

    return model


model = get_classifier(classifier_name)

#### Predict the emotion


def map_val_to_class(val):

    val_to_class = {0: 'joy', 1: 'sadness', 2: 'anger', 3: 'fear', 4: 'love'}
    return val_to_class[val]

####predict

#convert the input text to vector


input_text = preprocessing(input_text)

tf_idf_obj = load_pickle_files("Tf_idf_obj.pkl")
text_vector = tf_idf_obj.transform([input_text]) #shape 1*vocab_Size

svd_obj = load_pickle_files("svd_obj.pkl")
text_vector = svd_obj.transform(text_vector)

label = model.predict(text_vector)
emotion = map_val_to_class(label[0])
st.write("Detected emotion",emotion)
get_emoji(emotion)

