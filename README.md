# Salary-Prediction

<br>

# Project Description
The Salary Prediction App is a web-based tool that leverages ensemble machine learning to estimate a user’s expected salary based on key demographic and professional attributes: age, gender, education level, job title, and years of experience. Users choose among three regression models—Random Forest, Gradient Boosting, or a Voting Ensemble—to generate a prediction. A polished, responsive UI built with Streamlit overlays a full-screen background image, providing an intuitive form and visually striking result card.

<br>

# Implementation Details

Data & Models: Trained on a cleaned Kaggle salary dataset using scikit-learn’s RandomForestRegressor and GradientBoostingRegressor. A VotingRegressor combines both. All models use 200 trees/estimators.

Preprocessing: Categorical features (Gender, Job Title, Education Level) were encoded via LabelEncoder; numerical features (Age, Years of Experience) were scaled using StandardScaler. These transformers, along with trained models, are serialized together in a single model.pkl via joblib.

Web Application: The front end uses Streamlit for rapid development and deployment. Custom CSS targets Streamlit’s stAppViewContainer and .block-container to apply a full-screen background image and a semi-transparent dark overlay behind the content. Widgets for input (sliders, selectboxes) render inside this overlay. The predict button is centered via flexbox styling. User inputs are encoded, scaled, and passed to the selected model for real-time inference.

Architecture: Everything runs in a single Python process. On startup, model.pkl is loaded, and the app listens for interactions. On “Predict Salary,” a pandas DataFrame is constructed for the user input, preprocessed, and fed into the chosen model. Prediction is displayed in a styled result box.

<br>

# Challenges & Solutions

Styling Streamlit Containers <br>
­Issue: Streamlit doesn’t nest widgets inside custom HTML "<div>" blocks, so manual wrappers didn’t “cover” content. <br>
Solution: Switched to targeting Streamlit’s native .block-container class in CSS, applying a transparent black background to encapsulate all widgets automatically. <br>

Hiding Native UI Elements <br>
­Issue: The hamburger menu and footer changed identifiers in newer Streamlit versions (IDs like #MainMenu no longer existed). <br>
Solution: Used resilient selectors—[data-testid="stHeader"] and footer with display: none—to reliably remove those elements across versions. <br>

Local Background Images <br>
­Issue: Local file paths for background images did not render in deployed environments. <br>
Solution: Switched to a hosted Unsplash URL for the background image, ensuring consistent rendering without path issues. <br>

Textual vs. Encoded Inputs <br>
­Issue: Users naturally enter job titles and education as text, but the model expects integer-encoded values. <br>
Solution: Saved each LabelEncoder to model.pkl. At prediction time, raw text inputs are transformed with the same encoders, preserving consistency and preventing mismatches. <br>

Responsive Layout <br>
­Issue: Centering the predict button and ensuring the content fits without scrolling proved tricky. <br>
Solution: Employed CSS flexbox (.stButton { display:flex; justify-content:center; }) and, optionally, disabled scrolling via overflow: hidden on html, body, .stAppViewContainer when the content height remained within viewport bounds.
