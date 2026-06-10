import streamlit as st
import pickle
import os
import pickle
import numpy as np


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)


# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    h1 {
        text-align:center;
        color:#00FFAA;
        font-size:50px;
    }

    .stButton button {
        width:100%;
        height:50px;
        background-color:#00FFAA;
        color:black;
        font-size:20px;
        border-radius:15px;
    }

    .prediction {
        background-color:#123524;
        padding:25px;
        border-radius:20px;
        text-align:center;
        font-size:30px;
        color:#00FFAA;
        margin-top:20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# LOAD MODEL
# -----------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "car_price_model.pkl"
)


try:

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        model = pickle.load(file)


except Exception as e:

    st.error(
        f"Model loading failed: {e}"
    )

    st.stop()


# -----------------------------
# SIDEBAR
# -----------------------------


st.sidebar.title(
    "🚗 Car Price ML App"
)


st.sidebar.info(
    """
    Machine Learning Project

    Algorithm:
    🌲 Random Forest Regressor

    Features:
    - Present Price
    - Kilometers Driven
    - Fuel Type
    - Seller Type
    - Transmission
    - Owners
    - Car Age
    """
)


st.sidebar.success(
    "Developed using Python & Scikit-Learn"
)



# -----------------------------
# TITLE
# -----------------------------


st.title(
    "🚗 Car Price Prediction System"
)


st.write(
    "### Enter car details to estimate the selling price"
)



# -----------------------------
# INPUT SECTION
# -----------------------------


col1, col2 = st.columns(2)



with col1:


    present_price = st.number_input(
        "💰 Present Price (Lakhs)",
        min_value=0.0,
        max_value=200.0,
        value=5.0
    )


    kms = st.number_input(
        "🚘 Kilometers Driven",
        min_value=0,
        max_value=500000,
        value=30000
    )


    fuel = st.selectbox(
        "⛽ Fuel Type",
        [
            "Petrol",
            "Diesel",
            "CNG"
        ]
    )


    owner = st.selectbox(
        "👤 Previous Owners",
        [
            0,
            1,
            2,
            3
        ]
    )



with col2:


    seller = st.selectbox(
        "🏪 Seller Type",
        [
            "Dealer",
            "Individual"
        ]
    )


    transmission = st.selectbox(
        "⚙ Transmission",
        [
            "Manual",
            "Automatic"
        ]
    )


    age = st.slider(
        "📅 Car Age",
        0,
        30,
        5
    )



# -----------------------------
# ENCODING
# -----------------------------


fuel_dict = {

    "CNG":0,
    "Diesel":1,
    "Petrol":2

}


seller_dict = {

    "Dealer":0,
    "Individual":1

}


trans_dict = {

    "Automatic":0,
    "Manual":1

}



# -----------------------------
# PREDICTION
# -----------------------------


if st.button(
    "🔮 Predict Car Price"
):


    data = np.array(
        [[

        present_price,
        kms,
        fuel_dict[fuel],
        seller_dict[seller],
        trans_dict[transmission],
        owner,
        age

        ]]
    )



    prediction = model.predict(
        data
    )


    price = round(
        prediction[0],
        2
    )



    st.markdown(
        f"""

        <div class="prediction">

        Estimated Selling Price

        <br><br>

        ₹ {price} Lakhs 🚗

        </div>

        """,
        unsafe_allow_html=True
    )



    st.subheader(
        "📊 Selected Car Details"
    )



    st.write(
        {
            "Present Price":str(present_price)+" Lakhs",

            "Driven KM":kms,

            "Fuel":fuel,

            "Seller":seller,

            "Transmission":transmission,

            "Owners":owner,

            "Age":str(age)+" years"
        }
    )



# -----------------------------
# FOOTER
# -----------------------------


st.markdown(
    """

    ---
    ⭐ Machine Learning Regression Project  
    Build by Aman
    """
)
