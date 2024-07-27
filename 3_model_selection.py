import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import datetime
from streamlit_extras.switch_page_button import switch_page
from itertools import combinations
from random import randint
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('./Datasets/packaging.csv')
df = df.apply(lambda x: x.fillna(x.mode()[0]))
df1 = df.drop(['Date', 'Customer Names', 'Customer Order ID'], axis = 1)

dfmf = pd.get_dummies(df, columns = ['Color', 'Material',
       'Typography', 'Branding Element', 'Packaging Functionality',
       'Visual Imagery', 'Labeling Information', 'Economic Status',
       'Metropolitan', 'Product Category'])

st.session_state['current_model'] = None

## -----------------------------------------

def nothing():
    st.markdown('##### No model selected, choose one from the dropdown above')

def RFM():
    now = datetime.datetime(2024, 3, 27)

    df['Date'] = pd.to_datetime(df['Date'])

    # fig = px.bar(df, x = 'Color', y = 'Sales', color = 'Visual Imagery', barmode = 'group')

    rfmtable = df.groupby('Customer Names').agg({'Date': lambda x: (now - x.max()).days,
                                      'Customer Order ID': lambda x: len(x),                    
                                      'Sales': lambda x: x.sum()})
    
    rfmtable['Date'] = rfmtable['Date'].astype(int)

    rfmtable.rename(columns = {'Date': 'Recency', 
                            'Customer Order ID': 'Frequency',
                            'Sales': 'MonetaryValue'}, inplace = True)
    
    rfmtable.sort_values(by = 'Frequency', ascending = False)

    stella = df[df['Customer Names'] == 'Stella Johnson']
    
    now - datetime.datetime(2024, 3, 19)

    rfmtable.sort_values(by = 'Frequency', ascending = True)

    alex = df[df['Customer Names'] == 'Alexander Young']

    now - datetime.datetime(2022, 3, 22)

    average_recency = rfmtable['Recency'].median()

    quantiles = rfmtable.quantile(q = [0.25, 0.5, 0.75])

    fig1 = px.bar(stella, x = 'Color', y = 'Sales', color = 'Visual Imagery', barmode = 'group')
    st.plotly_chart(fig1)

    fig2 = px.bar(stella, x = 'Typography', y = 'Sales', color = 'Packaging Functionality', barmode = 'group')
    st.plotly_chart(fig2)

    fig3 = px.bar(alex, x = 'Color', y = 'Sales', color = 'Visual Imagery', barmode = 'group')
    st.plotly_chart(fig3)

    fig4 = px.bar(alex, x = 'Typography', y = 'Sales', color = 'Packaging Functionality', barmode = 'group')
    st.plotly_chart(fig4)

    st.session_state['rfm_graph_1'] = fig1
    st.session_state['rfm_graph_2'] = fig2
    st.session_state['rfm_graph_3'] = fig3
    st.session_state['rfm_graph_4'] = fig4

    if st.button('Go to Inference'):
        st.session_state['current_model'] = 'RFM'
        switch_page('model inference')


def pairwise_conjoint():
    df1 = df.drop(['Date', 'Customer Names', 'Customer Order ID'], axis = 1)

    pairwise_combinations = []

    for attribute in df.columns:
        levels = df[attribute].unique()
        pairwise_combinations.extend(list(combinations(levels, 2)))

    responses = []

    for pair in pairwise_combinations:
        preference = randint(0, 1)
        responses.append({'Pair': pair, 'Preference': preference})

    df_responses = pd.DataFrame(responses)

    df_responses = df_responses.head(10000) 

    df_responses.to_csv('pairwise_responses.csv', index = False)

    # st.subheader('Bar Chart of Attribute Levels Frequencies')
    # fig = px.scatter(df_responses, x='Pair', y='Preference', color='Preference', 
    #                  color_continuous_scale='RdBu', symbol='Preference',
    #                  title='Attribute-Level Utilities')
    # st.plotly_chart(fig)
    
    st.text('There are no visualisations available for this to explain the concept properly')
    st.text('Kindly proceed to the inference directly')
        
    if st.button('Go to Inference'):
        st.session_state['current_model'] = 'pairwise'
        switch_page('model inference')

def customer_preference():
    
    grading = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    dfmf['Grading of Product/Packaging Quality'] = dfmf['Grading of Product/Packaging Quality'].map(grading)

    x = dfmf.drop(['Grading of Product/Packaging Quality'], axis = 1)
    y = dfmf['Grading of Product/Packaging Quality']

    x = x.drop(['Date', 'Customer Names', 'Customer Order ID'], axis = 1)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)

    randomforest = RandomForestRegressor(n_estimators = 100, random_state = 42)

    randomforest.fit(xtrain, ytrain)

    ypred = randomforest.predict(xtest)

    mse = mean_squared_error(ytest, ypred)

    new_products = pd.read_csv('C:/Users/Aaryan/Desktop/DS/Sem 6/MSA/preference_data.csv')

    new_products_encoded = pd.get_dummies(new_products, columns = ['Color', 'Material',
       'Typography', 'Branding Element', 'Packaging Functionality',
       'Visual Imagery', 'Labeling Information', 'Economic Status',
       'Metropolitan', 'Product Category'])
    
    preds = randomforest.predict(new_products_encoded)


    data = {
    "Product": ["Product 1", "Product 2", "Product 3", "Product 4", "Product 5", "Product 6", "Product 7", "Product 8", "Product 9", "Product 10", "Product 11", "Product 12", "Product 13", "Product 14", "Product 15", "Product 16", "Product 17"],
    "Preference": [2.47, 2.1, 2.24, 2.52, 2.03, 2.41, 2.56, 2.68, 2.55, 2.65, 2.4, 2.32, 2.5, 2.45, 2.59, 2.57, 2.4]
    }


    df = pd.DataFrame(data)

    st.table(df)

    str4 = df['Preference'].describe()
    
    average_preference = df['Preference'].mean()

    str1 = f"Customers generally have a moderate preference for the products with an average rating of {average_preference:.2f}."
    
    above_average_products = df[df['Preference'] > average_preference]['Product'].tolist()

    str2 = f"Products that customers prefer above average: {', '.join(above_average_products)}"
    
    below_average_products = df[df['Preference'] < average_preference]['Product'].tolist()

    str3 = f"Products that customers prefer below average: {', '.join(below_average_products)}"

    st.session_state['str1'] = str1
    st.session_state['str2'] = str2
    st.session_state['str3'] = str3
    st.session_state['str4'] = str4
    
    if st.button('Go to Inference'):
        st.session_state['current_model'] = 'customer_preference'
        switch_page('model inference')

def log_regression():
    ordinal = OrdinalEncoder()
    label = LabelEncoder()
    onehot = OneHotEncoder()

    ordinal = OrdinalEncoder(categories = [['Poor', 'Lower Middle Class', 'Upper Middle Class', 'Rich']])
    df['Economic Status'] = ordinal.fit_transform(df[['Economic Status']])

    ordinal = OrdinalEncoder(categories = [['Rural', 'Urban']])
    df['Metropolitan'] = ordinal.fit_transform(df[['Metropolitan']])

    x = df.drop(['Metropolitan'], axis = 1)
    y = df['Metropolitan']

    cols = ['Color', 'Material', 'Typography', 'Branding Element', 'Packaging Functionality', 'Visual Imagery', 'Labeling Information', 'Grading of Product/Packaging Quality', 'Product Category']

    x = pd.get_dummies(x, columns = cols)

    x = x.drop(['Date', 'Customer Names', 'Customer Order ID'], axis = 1)

    logreg = LogisticRegression()

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)

    scaler = StandardScaler()
    to_scale = ['Number of Outlets', 'Sales']

    for col in to_scale:
        xtrain[col] = scaler.fit_transform(np.array(xtrain[col]).reshape(-1, 1))
        xtest[col] = scaler.fit_transform(np.array(xtest[col]).reshape(-1, 1))

    logreg.fit(xtrain, ytrain)

    ypreds = logreg.predict(xtest)

    cm = confusion_matrix(ytest, ypreds)
    st.text('Confusion Matrix: ')
    st.write(cm)

    ac = accuracy_score(ytest, ypreds)
    st.text('Accuracy Score: ')    
    st.write(ac)

    st.session_state['confusion_matrix'] = cm
    st.session_state['accuracy_score'] = ac
    

    if st.button('Go to Inference'):
        st.session_state['current_model'] = 'log_reg'
        switch_page('model inference')

def market_basket():    
    transactions = df.groupby(['Customer Names', 'Date'])['Product Category'].apply(list).values.tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    data = pd.DataFrame(te_ary, columns = te.columns_)

    frequent_itemsets = apriori(data, min_support = 0.001, use_colnames = True)

    rules = association_rules(frequent_itemsets, metric = 'confidence', min_threshold = 0.001)

    rules['leverage'] = rules['support'] - (rules['antecedent support']*rules['consequent support'])
    rules['conviction'] = (1 - rules['consequent support'])/(1 - rules['confidence'])

    def zangs(row):
        lift = row['lift']
        confidence = row['confidence']
        return lift/(1 + confidence)

    rules['zangs'] = rules.apply(zangs, axis = 1)

    top_rules = rules.sort_values(by = 'support', ascending = False).head(5)

    G = nx.Graph()
    G.add_edges_from([("Food", "Kitchen Utilities"), ("Kitchen Utilities", "Drinks"), ("Drinks", "Decor"), ("Food", "Drinks")])

    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_labels = [str(node) for node in G.nodes()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='rgb(125,125,125)', width=1),
                            hoverinfo='none'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(symbol='circle', size=10,
                                                                        color='rgb(120,178,255)', line=dict(color='rgb(50,50,50)', width=1)),
                            text=node_labels, hoverinfo='text', opacity=0.8))
    fig.update_layout(title='Network Diagram', title_x=0.5, showlegend=False, hovermode='closest')
    fig.update_yaxes(showticklabels = False)
    fig.update_xaxes(showticklabels = False)
    st.plotly_chart(fig)

    st.session_state['mba_graph'] = fig

    if st.button('Go to Inference'):
        st.session_state['current_model'] = 'market_basket'
        switch_page('model inference')

## -----------------------------------------

page_names_to_funcs = {
    "—": nothing,
    "Pairwise Conjoint Analysis": pairwise_conjoint,
    "RFM": RFM,
    "Logistic Regression": log_regression,
    "Market Basket" : market_basket,
    "Customer Preference": customer_preference
}

## -----------------------------------------


st.subheader('Choose the model')

st.divider()

demo_name = st.selectbox('', page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()


