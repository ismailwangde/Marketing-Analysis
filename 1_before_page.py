import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title = 'Before',
    page_icon = ''
)

# uploaded_file = st.session_state['file']

# if uploaded_file is not None:
#     try:
#         st.text('It does get till here')
#         df = pd.read_csv(uploaded_file)
#         st.table(df)
        
#     except FileNotFoundError:
#         st.error("Uploaded file not found. Please re-upload the data.")
# else:
#     st.info("No file uploaded yet. Please upload a CSV file in the previous page.")

df = pd.read_csv('./Datasets/packaging.csv')

st.title('Data before cleaning')

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
    st.text('This is Histogram')
    c2 = st.container(border = True)
    fig = px.histogram(data_frame = df, x = 'Color', color = 'Material')
    c2.plotly_chart(fig, use_container_width = True)

    st.subheader('C4')
    st.text('This is Treemap chart')
    c4 = st.container(border = True)
    fig = px.treemap(df, path = ['Color', 'Material'], values = 'Sales', color = 'Number of Outlets')
    c4.plotly_chart(fig, use_container_width = True)

st.divider()

if st.button(label = 'Clean the data'):
    switch_page('after page')


# st.session_state


