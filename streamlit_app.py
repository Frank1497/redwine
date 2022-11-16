import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

#Intialise Model
model = pickle.load(open('randfc.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


st.title('Machine Learning App')
st.header('Red Wine Dataset :wine_glass:')

feature = st.sidebar.selectbox('Select Feature', ('Description', 'Data Visualisation', 'Predict with User Input'))
@st.cache()
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return  df
df = read_csv('winequality-red.csv')
# data = df.drop('quality', axis=1)
st.dataframe(df)

if feature == 'Description':
    st.subheader('The goal of the model is for the user to input values of their wine and the model will predict quality of wine that it will produce')

    st.write(' Attribute Information:')
    st.write('For more information, read[Cortez et al., 2009].\n')
    st.write('Input variables(based on physicochemical tests):\n')
    st.write('1 - fixed acidity\n')
    st.write('2 - volatile acidity\n')
    st.write('3 - citric acid\n')
    st.write('4 - residual sugar\n')
    st.write('5 - chlorides\n')
    st.write('6 - free sulfur dioxide\n')
    st.write('7 - total sulfur dioxide\n')
    st.write('8 - density\n')
    st.write('9 - pH\n')
    st.write('10 - sulphates\n')
    st.write('11 - alcohol\n')
    st.write('Output variable(based on sensory data):\n')
    st.write('12 - quality\n')

if feature == 'Data Visualisation':
    st.header('Data Visiualisation')
    def chart(y):
        x = df['quality']
        chart = plt.figure(figsize=(10, 4))
        sns.boxplot(x=x, y=y)
        return chart

    st.subheader('Quality vs Alcohol')
    st.pyplot(chart(df.alcohol))

    st.subheader('Quality vs Citric Acid')
    st.pyplot(chart(df['citric acid']))

    st.subheader('Quality vs Sulphates')
    st.pyplot(chart(df['sulphates']))

    st.write('It can be seen that as more alcohol, citric acid and sulphates is used the better the quality of wine')


    st.subheader('Quality vs Volatile Acidity')
    st.pyplot(chart(df['volatile acidity']))

    st.subheader('Quality vs Density')
    st.pyplot(chart(df['density']))

    st.subheader('Quality vs pH')
    st.pyplot(chart(df['pH']))

    st.write('It can be seen that as more denser, more volatile acidity  and higher pH is used the lower the quality of wine')

if feature == 'Predict with User Input':
    col = df.columns
    data = []
    p1 = st.number_input(f'{col[0]}')
    p2 = st.number_input(f'{col[1]}')
    p3 = st.number_input(f'{col[2]}')
    p4 = st.number_input(f'{col[3]}')
    p5 = st.number_input(f'{col[4]}')
    p6 = st.number_input(f'{col[5]}')
    p7 = st.number_input(f'{col[6]}')
    p8 = st.number_input(f'{col[7]}')
    p9 = st.number_input(f'{col[8]}')
    p10 = st.number_input(f'{col[9]}')
    p11 = st.number_input(f'{col[10]}')

    if st.button('Predict'):
        data.append([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11])
        final_input = scaler.transform(np.array(data).reshape(1, -1))
        output = model.predict(final_input)[0]
        quality = {0: 'poor', 1: 'good', 2: 'excellent'}
        st.write(f'The wine that will be produced will be of {quality[output]} quality')