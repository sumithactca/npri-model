import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
df_ML = pd.read_csv("ML_data.csv")

# Set page title and favicon
st.set_page_config(page_title="NPRI ML Model", page_icon="📊")

# Set up the Streamlit app with a custom background color and text colors
st.markdown(
    """
    <style>
    body {
        background-color: #d4f4dd; /* Light green */
        color: #333333; /* Dark blue */
    }
    .feature-box {
        background-color: #c7ecee; /* Light blue */
        padding: 10px ;
        border-radius: 10px ;
        box-shadow: 0px 0px 10px 0px rgba(0,0,0.1,0.9);
        margin-right: 10px;
        margin-bottom: 10px;
        display: inline-block;
        width: 200px; /* Adjust the width of the feature boxes */
    }
    .feature-box:nth-child(even) {
        background-color: #a9cce3; /* Light blue for alternate boxes */
    }
    .title-text {
        font-size: 18px;
        color: #004c6d; /* Dark blue */
        font-weight: bold; /* Make title text bold */
    }
    .subheader-text {
        font-size: 16px;
        color: #666666; /* Dark blue */
    }
    .selected-feature {
        font-weight: bold; /* Make selected feature text bold */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.title("NPRI ML Model")

# Sidebar
st.sidebar.header('Select Features')

# Select box for categorical features
st.subheader('Selected Features')
def user_input_features():
    company_name = st.sidebar.selectbox('Company Name', df_ML['Company name'].unique())
    province = df_ML[df_ML['Company name'] == company_name]['Province'].iloc[0]
    substance_name = st.sidebar.selectbox('Substance Name', df_ML['Substance name'].unique())
    number_of_employees = st.sidebar.number_input('Number of Employees', min_value=1, max_value=5000, value=10, step=1)
    growth_input = st.sidebar.radio('Growth', ('up', 'down'))
    Price_input = st.sidebar.radio('Price', ('up', 'down'))
    
    user_input_data = {'Company Name': company_name,
                       'Province': province,
                       'Substance Name': substance_name,
                       'Number of Employees': number_of_employees,
                       'Growth': growth_input,
                       'Price': Price_input}

    features = pd.DataFrame(user_input_data, index=[0])

    return features

df_user_input = user_input_features()

st.write(df_user_input.style.set_properties(**{'font-weight': 'bold'}))

# Placeholder for displaying predicted quantity for 2023
st.markdown("<hr>", unsafe_allow_html=True)
st.write("<p class='title-text'><b>Predicted Quantity:</b></p>", unsafe_allow_html=True)

# Filter data for the selected company name and substance name for the year 2022
filtered_data = df_ML[(df_ML['Company name'] == df_user_input['Company Name'].iloc[0]) &
                      (df_ML['Substance name'] == df_user_input['Substance Name'].iloc[0]) &
                      (df_ML['NPRI_Report_ReportYear'] == 2022)]

if not filtered_data.empty:
    # Fetch the current year quantity for 2022
    current_year_quantity = filtered_data['Current year quantity'].iloc[0]

    # Encode growth
    growth_mapping = {'up': 1, 'down': 0}
    growth_encoded = growth_mapping[df_user_input['Growth'].iloc[0]]

    # Encode price
    price_mapping = {'up': 1, 'down': 0}
    price_encoded = price_mapping[df_user_input['Price'].iloc[0]]

    # Assign the fetched quantities to the corresponding placeholders
    input_data = {
        'Last year quantity': [filtered_data['Current year quantity'].iloc[0]],
        'Two years ago quantity': [filtered_data['Last year quantity'].iloc[0]],
        'Three years ago quantity': [filtered_data['Two years ago quantity'].iloc[0]],
        'Number of employees': [df_user_input['Number of Employees'].iloc[0]],
        'Growth': [growth_encoded],
        'Price': [price_encoded],
    }

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(df_ML[['Last year quantity', 'Two years ago quantity', 'Three years ago quantity',
                     'Number of employees', 'Growth', 'Price']], df_ML['Current year quantity'])

    # Make prediction for 2023
    prediction = model.predict(pd.DataFrame(input_data))

    # Display the predicted quantity for 2023
    st.write(f"<div class='feature-box'><b>{prediction[0]}</b></div>", unsafe_allow_html=True)
else:
    st.write("<p class='subheader-text'>No data available for the selected company name and substance name for the year 2022.</p>", unsafe_allow_html=True)
