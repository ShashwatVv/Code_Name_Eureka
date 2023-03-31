import matplotlib.pyplot as plt
import streamlit as st
from sklearn import  datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

st.title("Classify your choice of dataset on your choice of model")
st.write("### Try out different classifiers")

dataset_name = st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine Dataset"))
classifier_name = st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest"))


def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    Y = data.target

    return X,Y


X,Y = get_dataset(dataset_name)
st.write("Shape of dataset",X.shape)
st.write("Number of Classes",len(np.unique(Y)))

#plot
pca = PCA(n_components=2)
x_components = pca.fit_transform(X)
x1 = x_components[:,0]
x2 = x_components[:,1]
fig,ax = plt.subplots()

st.write('<p style="color:orange;">Variance plot of the dataset <br ></p>',unsafe_allow_html=True)

plt.scatter(x1,x2,c=Y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)



def add_parameter_ui(clf_name):

    params = dict()

    test_size = st.sidebar.selectbox("Train_test_split",("80-20","70-30","60-40","50-50"))

    if test_size =="80-20":
        params["test_size"]= 0.2
    elif test_size == "70-30":
        params["test_size"] = 0.3
    elif test_size == "60-40":
        params["test_size"] = 0.4
    else:
        params["test_size"] = 0.5

    if clf_name == "KNN":
        K = st.sidebar.slider("K",1,15)#parameters - name,start_value,end_value
        params["K"] = K
    elif clf_name == "SVM":
        C= st.sidebar.slider("C",0.01,10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    return params


parameters = add_parameter_ui(classifier_name)


def get_classifier(clf_name,parameter):
    if clf_name=="KNN":
        model = KNeighborsClassifier(n_neighbors=parameter["K"])
    elif clf_name=="SVM":
        model = SVC(C=parameter["C"])
    else:
        model = RandomForestClassifier(n_estimators=parameter["n_estimators"],max_depth=parameter["max_depth"])

    return model


def report_to_df(report):
    report = [x.split(' ') for x in report.split('\n')]
    header = ['Class Name']+[x for x in report[0] if x!='']
    values = []
    for row in report[1:-5]:
        row = [value for value in row if value!='']
        if row!=[]:
            values.append(row)
    df = pd.DataFrame(data = values, columns = header)
    return df


model = get_classifier(classifier_name,parameters)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=parameters["test_size"],random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

perf_report = classification_report(y_true=y_test,y_pred=y_pred)
perf_report = report_to_df(perf_report)
st.write('<p style="color:blue;">Classifier</p>',unsafe_allow_html=True)
st.write(classifier_name)
st.write('<p style="color:blue;">Results on test set </p>',unsafe_allow_html=True)
st.dataframe(perf_report)


