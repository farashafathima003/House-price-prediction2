
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="HomeValue | Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef5ff, #f8fafc);
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #102a43, #176b87);
    padding: 35px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 10px;
}

.hero p {
    color: #e0f2fe;
    font-size: 17px;
}

.section-title {
    color: #123047;
    font-size: 24px;
    font-weight: 700;
    margin: 20px 0 12px 0;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: white;
    border-radius: 16px;
    border: 1px solid #dbeafe;
    padding: 18px;
}

div[data-testid="stWidgetLabel"] p {
    color: #23445b;
    font-weight: 600;
}

.stNumberInput input {
    border-radius: 10px;
}

div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #087e8b, #12a4a6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 13px;
    font-weight: 700;
    transition: 0.2s;
}

div.stButton > button:hover {
    background: #075985;
    color: white;
    border: none;
}

.price-card {
    background: linear-gradient(120deg, #d1fae5, #ecfdf5);
    border: 1px solid #a7f3d0;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
}

.price-label {
    color: #065f46;
    font-size: 16px;
    font-weight: 600;
}

.price-value {
    color: #047857;
    font-size: 34px;
    font-weight: 800;
    margin-top: 8px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px 0 5px 0;
    font-size: 13px;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🏡 HomeValue")
st.sidebar.write("House Price Prediction System")

st.sidebar.divider()

st.sidebar.subheader("About the Project")
st.sidebar.write(
    "This application uses Machine Learning "
    "to estimate house prices from property details."
)

st.sidebar.subheader("Technologies Used")
st.sidebar.write("🐍 Python")
st.sidebar.write("📊 Pandas")
st.sidebar.write("🤖 Scikit-learn")
st.sidebar.write("🎨 Streamlit")

st.sidebar.divider()
st.sidebar.caption("Academic ML Project")
# Store prediction history
if "history" not in st.session_state:
    st.session_state.history = []
# Header
st.markdown("""
<div class="hero">
    <h1>🏡 HomeValue</h1>
    <p>Smart House Price Prediction using Machine Learning</p>
    <p>Discover an estimated property value using your house details.</p>
</div>
""", unsafe_allow_html=True)

# Main form
st.markdown(
    '<div class="section-title">🏠 Enter Property Details</div>',
    unsafe_allow_html=True
)

with st.form("house_form"):
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (square feet)",
            min_value=100,
            max_value=10000,
            value=1200,
            step=50
        )

        bedrooms = st.selectbox(
            "Number of bedrooms",
            [1, 2, 3, 4, 5, 6, 7, 8],
            index=2
        )

        bathrooms = st.selectbox(
            "Number of bathrooms",
            [1, 2, 3, 4, 5, 6],
            index=1
        )

    with col2:
        age = st.number_input(
            "House age (years)",
            min_value=0,
            max_value=200,
            value=5
        )

        location = st.selectbox(
            "Property location",
            ["Kozhikode", "Kochi", "Malappuram", "Kannur"]
        )

        st.info("Enter the property details carefully.")

    submitted = st.form_submit_button("✨ Predict House Price")

if submitted:
    house = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age": age,
        "location": location
    }])

    price = model.predict(house)[0]
    st.session_state.history.append({
    "Location": location,
    "Area (sq. ft.)": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Age (years)": age,
    "Predicted Price (₹)": round(price, 2)
})

    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Estimated House Price</div>
        <div class="price-value">₹ {price:,.0f}</div>
        <p>📍 {location} &nbsp; | &nbsp; {area:,} sq. ft.</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Note: This estimate uses a small sample dataset "
        "for educational purposes. It is not a verified "
        "real-world property valuation."
    )

# Footer
st.markdown("""
<div class="footer">
    HomeValue | House Price Prediction ML Project
    <br>Powered by Python, Scikit-learn and Streamlit
</div>
""", unsafe_allow_html=True)

# Property banner image
st.image(
    "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9"
    "?auto=format&fit=crop&w=1200&q=80",
    caption="Find the estimated value of your property",
    use_container_width=True
)
# Prediction History
st.divider()

st.subheader("📊 Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    if st.button("Clear Prediction History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No predictions yet. Predict a house price to see your history.")
