import streamlit as st
import pandas as pd
from streamlit_extras.switch_page_button import switch_page

df = pd.read_csv('./Datasets/packaging.csv')
df = df.apply(lambda x: x.fillna(x.mode()[0]))

st.set_page_config(
    page_title='Model Inferencing',
    page_icon=''
)

st.title('Model Inferencing')

# st.button('Back')
# if st.button:
#     switch_page('model selection')

rfm_graph_1 = st.session_state.get('rfm_graph_1')
rfm_graph_2 = st.session_state.get('rfm_graph_2')
rfm_graph_3 = st.session_state.get('rfm_graph_3')
rfm_graph_4 = st.session_state.get('rfm_graph_4')
mba_graph = st.session_state.get('mba_graph')
str1 = st.session_state.get('str1')
str2 = st.session_state.get('str2')
str3 = st.session_state.get('str3')
str4 = st.session_state.get('str4')

def inference_for_RFM():
    st.write("Inference for RFM model: ")
    st.plotly_chart(rfm_graph_1)
    st.markdown('')
    st.markdown('''
        **Understanding RFM Scores:**

        Interpreting the individual RFM scores (R ecency, Frequency, Monetary value) for each customer segment.

        * **Recency (R):**
            * High R: Recent purchase (valuable customer). 
            * Low R: Long time since last purchase (potential churn risk).
        * **Frequency (F):**
            * High F: Purchases frequently (loyal customer).
            * Low F: Purchases infrequently (may need re-engagement).
        * **Monetary (M):**
            * High M: Spends a lot per purchase (valuable customer).
            * Low M: Spends less per purchase (may require upselling).
    ''')
    
    st.plotly_chart(rfm_graph_2)
    st.markdown('')

    st.markdown('''
        For example, Stella Johnson is a valuable customer, her last purchase was 2 days ago while Alexander Young is a potential churn risk, his last purchase was 730 days ago. The company may want to upsell their services/products to Alexander, and give promotional deals or coupons to Stella to reward her for her loyalty. 

        **Deriving Customer Insights from Segments:**

        After segmenting customers based on RFM scores, analyze each segment to draw inferences:

        * **Champions (High R, High F, High M):** Ideal customers, prioritize retention strategies (loyalty programs, exclusive offers).
        * **Loyal Customers (High R, High F, Medium M):** Valuable, nurture loyalty (personalized recommendations, birthday discounts).
        * **Potential Loyals (High R, Medium F, Medium M):** Encourage repeat purchases (targeted promotions, bundled products).
        * **At-Risk Customers (Low R, Medium F, Medium M):** Re-engagement needed (win-back campaigns, special offers).
        * **Does Not Repeat (Low R, Low F, Low M):** Low value, consider acquisition cost vs. retention effort.
    ''')
    st.markdown('')


    st.plotly_chart(rfm_graph_3)
    st.markdown('')
    st.markdown('''
    **Tailoring Inferences to Business Goals:**

    * **Increase Revenue:** Focus on segments with high monetary value (Champions, Loyal Customers) and encourage upselling/cross-selling.
    * **Reduce Churn:** Target at-risk and does-not-repeat segments with win-back campaigns and personalized offers.
    * **Improve Customer Lifetime Value:** Implement strategies for all segments to encourage repeat purchases and loyalty.
    ''')
    st.plotly_chart(rfm_graph_4)


def inference_for_pairwise_conjoint():
    st.write("Inference for pairwise conjoint analysis model: ")
    st.markdown('''
    - Color Preferences:
        Black and Red tend to be paired more favorably than Yellow, Green, or Orange.
        Indigo seems to be the least preferred color overall.
        White is generally well-liked in combination with other colors.
    - Material Preferences:
        Cardboard is preferred when paired with Glass, Paper, or Plastic.
        Glass and Metal are seen favorably together.
        Paper and Plastic tend to be a good combination.
        Metal is not typically preferred with Cardboard or Paper.
    - Font Preferences:
        Arial is generally preferred over Courier New and Times New Roman.
        Verdana is a strong contender for Arial.
    - Product Packaging Preferences:
        Resealable and Portable are often desired attributes.
        Easy to Open is valued, but not necessarily with Resealable or Portable.
        Recyclable is not a top priority but might be relevant for some products.
    - Product Information Preferences:
        Nutritional Value is most important and often paired with Allergen Information, Storage Solutions, or Recipes.
        Certifications are of lesser importance but might be relevant with Allergen Information.
        Ingredients are assumed to be included and not explicitly mentioned in preferences.
    - Marketing Preferences:
        Icons are preferred over Images or Graphics.
        A Logo & Slogan combination is more favorable than just a Logo.
        Brand Name is not a key factor in these pairings.
    ''')

def inference_for_log_regression():
    st.write("Inference for logistic regression model: ")
    st.markdown('''
        Given a new product or packaging design with its corresponding attribute values, the trained logistic regression model will predict the likelihood of it belonging to the "Rural" or "Urban" category.
        The new data point would need to be preprocessed similarly to the training data (x). This includes encoding categorical variables and potentially scaling numerical features.
        Once preprocessed, the model will classify into a class ("Rural" or "Urban") for the new product.
                    
        Evaluation Metrics:
        - Confusion Matrix:
            This table summarizes the number of correct and incorrect predictions made by the model.
            In the presented confusion matrix:                
            292 \t 223 \n
            243 \t 242 \n
        
        Analyzing the confusion matrix helps identify potential biases or errors in the model's predictions.
        
        - Accuracy Score:
            This metric represents the overall proportion of correct predictions made by the model.
            The model achieves an accuracy of slightly above 53%.
                        
            Precision = TP / (TP + FP) = 292 / (292 + 223) = 0.568
            This indicates that out of the instances predicted as Rural, a little over half (56.8%) were actually correct.
            
            Recall = TP / (TP + FN) = 292 / (292 + 243) = 0.544
            It captures a little over half (54.4%) of the actual Rural instances and misses a substantial portion (45.6%) as false negatives (predicting Urban when it's actually Rural).
                    
        - F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
                = 2 * (0.568 * 0.544) / (0.568 + 0.544)
                = 0.555
            Reflects a somewhat balanced performance between precision and recall, but with a slight bias towards precision. This suggests the model prioritizes avoiding false positives over capturing all true positives.
                    
        **Interpretation of the Results**:
        Given an accuracy score of 53.4%, the model's performance is not very strong. It performs only slightly better than random guessing (which would achieve 50% accuracy in a two-class problem).
        Further model tuning might be necessary to improve the model's ability to distinguish between Rural and Urban categories based on the provided attributes.  
    ''')

def inference_for_market_basket():
    st.write("Inference for market basket model goes here.")
    st.plotly_chart(mba_graph)
    st.markdown('''
    - Food and Kitchen Utilities:
        - There is a negative association between Food and Kitchen Utilities. This implies that when Food is present, Kitchen Utilities are less likely to be present, and vice versa.
        - The low Leverage indicates that the occurrence of Food and Kitchen Utilities together is less frequent than expected if they were independent.
        - The Conviction value suggests that the occurrence of Kitchen Utilities is slightly less likely given the presence of Food compared to its absence.
        Zang's Metric indicates a weak association between Food and Kitchen Utilities.
    - Kitchen Utilities and Food:
        - This rule essentially mirrors the first rule with the same interpretation but from the opposite direction.
    - Drinks and Food:
        - There is a negative association between Drinks and Food. This suggests that when Food is present, Drinks are less likely to be present, and vice versa.
        - The negative Leverage value indicates that the co-occurrence of Drinks and Food is less frequent than expected if they were independent.
        - The Conviction value implies that the occurrence of Food is slightly less likely given the presence of Drinks compared to its absence.
        - Zang's Metric indicates a weak association between Drinks and Food.
    - Food and Drinks:
        - This rule essentially mirrors the third rule with the same interpretation but from the opposite direction.
    - Decor and Drinks:
        - There is a negative association between Decor and Drinks. This suggests that when Decor is present, Drinks are less likely to be present, and vice versa.
        - The negative Leverage value indicates that the co-occurrence of Decor and Drinks is less frequent than expected if they were independent.
        - The Conviction value implies that the occurrence of Drinks is slightly less likely given the presence of Decor compared to its absence.
        - Zang's Metric indicates a weak association between Decor and Drinks
    ''')

def inference_for_customer_preference():
    st.write("Inference for Customer Preference: ")
    st.write(str1)
    st.write(str2)
    st.write(str3)
    st.write(str4)

inference_functions = {
    'RFM': inference_for_RFM,
    'pairwise': inference_for_pairwise_conjoint,
    'log_reg': inference_for_log_regression,
    'market_basket': inference_for_market_basket,
    'customer_preference': inference_for_customer_preference
}
current_model = st.session_state.get('current_model')

if current_model in inference_functions:
    inference_functions[current_model]()
else:
    st.write("No model selected or inference not available.")





