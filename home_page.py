import streamlit as st
# from streamlit_option_menu import option_menu
# import pages
import pandas as pd
from streamlit_extras.switch_page_button import switch_page

st.title("Role of Packaging Design Analysis")
st.divider()

st.session_state['current_page'] = 'home_page'

col1, col2 = st.columns([3, 5]) 

with col1:
    st.subheader("Project Details")
    st.write("Packaging contributes to marketing efforts by differentiating products from competitors. Today, simple wrap sheets and storage boxes have become strategic communication tools. Marketers use custom packages with unique colors, shapes, sizes, fonts, and graphic designs to engage customers")
    st.write('')
    st.write("- Customer preference analysis was conducted to understand preferences regarding color, material, typography, branding elements, packaging functionality, visual imagery, labeling information, economic status, and metropolitan setting ")
    st.write('')
    st.write("- Logistic regression was used to predict whether a customer resides in an urban or rural setting based on the provided features")
    st.write('')
    st.write("- Market basket analysis was performed to uncover patterns in customer purchasing behavior, identifying frequently co-purchased items")
    st.write('')
    st.write("- Recency, frequency, and monetary (RFM) analysis was conducted to segment customers based on their purchasing behavior, aiming to identify high-value customers and customers at a risk of churning")
    st.write('')
    st.write("- Pairwise conjoint analysis was carried out to determine the relative importance of different product attributes in influencing customer preferences and purchase decisions")
 
with col2:
    uploaded_file = st.file_uploader("Upload your data file", type="csv")

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        df = pd.read_csv(uploaded_file)
        st.table(df.head(5))

        if uploaded_file not in st.session_state:
            st.session_state['file'] = uploaded_file

        if st.button("Submit"):
           st.session_state['current_page'] = 'page_1'
           switch_page('before page')

    else:
        st.info("Please upload a CSV file.")

st.divider()

# st.session_state
