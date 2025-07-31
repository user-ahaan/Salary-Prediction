import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Salary Prediction App", page_icon="💼", layout="centered")

st.markdown(
    """
    <style>
    [data-testid="stHeader"] {display: none;}
    footer {display: none;}
    #MainMenu {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>

    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1707902665498-a202981fb5ac?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }

    .block-container {
        background: rgba(0, 0, 0, 0.60);
        border-radius: 18px;
        max-width: 600px;
        margin-top: 10px;
        margin-bottom: 10px;
        padding: 48px 36px 32px 36px;
    }

    .result-box {
        background: rgba(0, 230, 77, 0.70);
        color: #fff;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1.5rem;
    }
    
    label, .stSelectbox label, .stSlider label, .stRadio label {
        font-weight: bold !important;
        color: #fff !important;
    }

    .stTextInput > div > input,
    .stSelectbox > div > div,
    .stSlider > div > div,
    .stRadio > div > div,
    .stNumberInput > div > input,
    .stButton > button {
        background: rgba(0, 0, 0, 0.20) !important;
        color: #fff !important;
        border-radius: 5px !important;
        border: none !important;
    }

    .stButton {
        display: flex;
        justify-content: center;
        margin-top: 1.5em;
    }

    ::placeholder {
        color: #ddd !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

data = joblib.load("model.pkl")
rf = data['Random Forest']
gb = data['Gradient Boosting']
voting = data['Voting Ensemble']
scaler = data['scaler']
le_gender = data['le_gender']
le_job = data['le_job']
le_edu = data['le_edu']

st.title("💼 Salary Prediction App")
st.write("Enter your details below to predict your salary.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age", 18, 70, 30)
    gender = st.selectbox("🧑 Gender", le_gender.classes_.tolist())
    education = st.selectbox("🎓 Education Level", le_edu.classes_.tolist())
with col2:
    job = st.selectbox("🏢 Job Title", le_job.classes_.tolist())
    experience = st.slider("📈 Years of Experience", 0.0, 40.0, 5.0, step=0.5)
    model_choice = st.selectbox(
    "🧠 Choose Model",
    ("Random Forest", "Gradient Boosting", "Voting Ensemble")
    )

if st.button("🔍 Predict Salary"):
    try:
        gender_enc = le_gender.transform([gender])[0]
        edu_enc = le_edu.transform([education])[0]
        job_enc = le_job.transform([job])[0]
    except ValueError as e:
        st.error(f"Unknown category: {e}")
        st.stop()

    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender_enc,
        'Education Level': edu_enc,
        'Job Title': job_enc,
        'Years of Experience': experience
    }])

    input_df[['Years of Experience', 'Age']] = scaler.transform(input_df[['Years of Experience', 'Age']])

    model = {
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "Voting Ensemble": voting
    }[model_choice]

    pred = model.predict(input_df)[0]

    st.markdown(
        f"""
        <div class="result-box">
            💰 Predicted Salary: ₹ {pred:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )
