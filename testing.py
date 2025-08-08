import pandas as pd
import joblib

model_data = joblib.load('model.pkl')

rf = model_data['Random Forest']
gb = model_data['Gradient Boosting'] 
voting = model_data['Voting Ensemble']
scaler = model_data['scaler']
le_gender = model_data['le_gender']
le_job = model_data['le_job']
le_edu = model_data['le_edu']

def predict_salary(age, gender, education_level, job_title, years_experience, model_choice='Random Forest'):    
    user_input = {
        'Age': age,
        'Gender': gender,
        'Education Level': education_level,
        'Job Title': job_title,
        'Years of Experience': years_experience
    }
    
    try:
        gender_encoded = le_gender.transform([user_input['Gender']])[0]
        job_encoded = le_job.transform([user_input['Job Title']])[0]
        edu_encoded = le_edu.transform([user_input['Education Level']])[0]
    except ValueError as e:
        print(f"Error: Unknown category value. {e}")
        print("Make sure your input values match the categories from the training data.")
        return
    
    user_df = pd.DataFrame([{
        'Age': user_input['Age'],
        'Gender': gender_encoded,
        'Education Level': edu_encoded,
        'Job Title': job_encoded,
        'Years of Experience': user_input['Years of Experience']
    }])
    
    features_to_scale = ['Years of Experience', 'Age']
    user_df[features_to_scale] = scaler.transform(user_df[features_to_scale])
    
    if model_choice == 'Random Forest':
        model = rf
    elif model_choice == 'Gradient Boosting':
        model = gb
    elif model_choice == 'Voting Ensemble':
        model = voting
    else:
        print("Invalid model choice. Using Random Forest.")
        model = rf
    
    salary_pred = model.predict(user_df)[0]
    
    return salary_pred

try:
    print("#             Welcome to the Salary Prediction App!             #")
    print("Please enter your details below (Note: Inputs are case-sensitive)")

    age = int(input("Enter age: "))
    gender = input("Enter gender (Male/Female/Other): ")
    education = input("Enter education level (Bachelor's/Master's/PhD/High School): ")
    job = input("Enter job title (Software Engineer, Data Scientist, Senior Manager, Director, Product Manager, IT Support, etc.): ")
    experience = float(input("Enter years of experience: "))
    model_type = input("Choose model (Random Forest/Gradient Boosting/Voting Ensemble): ")
    
    result = predict_salary(age, gender, education, job, experience, model_type)
    
    if result:
        print(f"Predicted Salary (Monthly): {result:.2f}")
        
except ValueError:
    print("Please enter valid inputs!")

    
