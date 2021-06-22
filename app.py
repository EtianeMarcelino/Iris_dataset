import pandas as pd
import streamlit as st
import seaborn as sn
from sklearn import preprocessing
from sklearn import tree
from pycaret.classification import *
import matplotlib.pyplot as plt

# loading the trained model.
model = load_model('model/modelo-final-dt')

# carregando uma amostra dos dados.
#dataset = pd.read_csv('data/dataset.csv') 
#classifier = pickle.load(pickle_in)
dataset = sn.load_dataset('iris')


# título
st.title("Dataset flores de Iris")

# subtítulo
st.markdown("Prever tipo de flores de ires.")


st.sidebar.header('Definir parametros')
st.sidebar.subheader("Defina os atributos da flor")


def user_input_features():
    # mapeando dados do usuário para cada atributo
    sepal_length = st.sidebar.slider('sepal_length', 4.3, 7.9, value=dataset["sepal_length"].mean())
    sepal_width = st.sidebar.slider('sepal_width',2.0, 4.4, value=dataset["sepal_width"].mean())
    petal_length = st.sidebar.slider('petal_length',1.0, 6.9, value=dataset["petal_length"].mean())
    petal_width = st.sidebar.slider('petal_width',0.1, 2.5, value=dataset["petal_width"].mean())
    data = {
        'sepal_length' : sepal_length,
        'sepal_width' : sepal_width,
        'petal_length' : petal_length,
        'petal_width' : petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


st.subheader('Valor definido pelo utilizador')
st.write(df)

st.subheader('Classes de Iris e respectivos index')
st.write(dataset['species'].unique())

st.subheader('**Previsão**')

#realiza a predição
result = predict_model(model, data=df)["Label"]
result = int(round(result[0],2))
def classe (num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'versicolor'
    elif num == 2:
        return 'virginica'

st.write(classe(result))
