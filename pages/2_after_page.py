import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title = 'After',
    page_icon = ''
)

df = pd.read_csv('./Datasets/packaging.csv')
df = df.apply(lambda x: x.fillna(x.mode()[0]))

st.title('Data after cleaning')

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader('Info')
    c1 = st.container(border = True)
    # c1.write(df.info())
    "Total no of null/missing values: " ,df.isna().sum()

    st.subheader('C3')
    st.text('This is Pie chart')
    c3 = st.container(border = True) 
    fig = px.pie(df, values = 'Sales', names = 'Visual Imagery')
    c3.plotly_chart(fig, use_container_width = True)

with col2:
    st.subheader('C2')
    st.text('This is Histogram chart')
    c2 = st.container(border = True)
    fig = px.histogram(data_frame = df, x = 'Color', color = 'Material')
    c2.plotly_chart(fig, use_container_width = True)

    st.subheader('C4')
    st.text('This is Treemap chart')
    c4 = st.container(border = True)
    fig = px.treemap(df, path = ['Color', 'Material'], values = 'Sales', color = 'Number of Outlets')
    c4.plotly_chart(fig, use_container_width = True)

st.divider()

if st.button(label = 'Model Selection'):
    'Go to Model Selection'
    switch_page('model selection')
