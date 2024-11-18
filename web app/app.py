from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
import ollama

# Load models
student_model = tf.keras.models.load_model('models/nn_student.h5')
professional_model = tf.keras.models.load_model('models/nn_professional.h5')

# Features for each model with descriptions
student_features = {
    'Gender': ['Male', 'Female'],
    'Age': 'Enter your age (e.g., 25)',
    'Academic Pressure': 'Rate your academic pressure from 1 (low) to 5 (high)',
    'Study Satisfaction': 'Rate your study satisfaction from 1 (low) to 5 (high)',
    'Sleep Duration': ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'],
    'Dietary Habits': ['Healthy', 'Moderate', 'Unhealthy'],
    'Have you ever had suicidal thoughts ?': ['Yes', 'No'],
    'Study Hours': 'Enter your daily study hours (e.g., 4)',
    'Financial Stress': 'Rate your financial stress from 1 (low) to 5 (high)',
    'Family History of Mental Illness': ['Yes', 'No']
}

professional_features = {
    'Gender': ['Male', 'Female'],
    'Age': 'Enter your age (e.g., 40)',
    'Work Pressure': 'Rate your work pressure from 1 (low) to 5 (high)',
    'Job Satisfaction': 'Rate your job satisfaction from 1 (low) to 5 (high)',
    'Sleep Duration': ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'],
    'Dietary Habits': ['Healthy', 'Moderate', 'Unhealthy'],
    'Have you ever had suicidal thoughts ?': ['Yes', 'No'],
    'Work Hours': 'Enter your daily work hours (e.g., 8)',
    'Financial Stress': 'Rate your financial stress from 1 (low) to 5 (high)',
    'Family History of Mental Illness': ['Yes', 'No']
}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form/<user_type>')
def form(user_type):
    if user_type == 'student':
        features = student_features
    elif user_type == 'professional':
        features = professional_features
    else:
        return "Invalid user type", 400
    return render_template('form.html', user_type=user_type, features=features)

@app.route('/predict/<user_type>', methods=['POST'])
def predict(user_type):
    if user_type == 'student':
        model = student_model
        features = list(student_features.keys())
    elif user_type == 'professional':
        model = professional_model
        features = list(professional_features.keys())
    else:
        return "Invalid user type", 400

    # Collect input data
    input_data = [request.form[feature] for feature in features]

    # Convert to DataFrame
    df_input = pd.DataFrame([input_data], columns=features)

    # Preprocess data
    try:
        for col in df_input.columns:
            if col in ['Gender', 'Dietary Habits', 'Sleep Duration', 'Have you ever had suicidal thoughts ?',
                       'Family History of Mental Illness']:
                # Encode categorical variables
                df_input[col] = df_input[col].astype('category').cat.codes
            else:
                # Ensure numerical variables are floats
                df_input[col] = df_input[col].astype(float)
    except Exception as e:
        return render_template('result.html', user_type=user_type, result=f'Preprocessing Error: {str(e)}')

    # Predict
    try:
        probabilities = model.predict(df_input)
        probability = float(probabilities[0]) * 100  # Convert to percentage
        result = "Depression" if probability > 50 else "No Depression"
    except Exception as e:
        return render_template('result.html', user_type=user_type, result=f'Model Prediction Error: {str(e)}')

    # Generate feedback using Ollama
    feedback = generate_feedback(user_type, df_input.iloc[0].to_dict(), result, probability)

    return render_template('result.html', user_type=user_type, result=result, probability=probability, feedback=feedback)


    # Generate feedback using Ollama
    feedback = generate_feedback(user_type, df_input.iloc[0].to_dict(), result, probability)

    return render_template('result.html', user_type=user_type, result=result, probability=probability, feedback=feedback)

def generate_feedback(user_type, user_input, prediction, probability):
    """
    Use Ollama to generate feedback based on the user's input, prediction, and probability.
    """
    input_summary = ', '.join([f"{k}: {v}" for k, v in user_input.items()])
    message_content = (
        f"User type: {user_type}\n"
        f"Prediction: {prediction} (Probability: {probability:.2%})\n"
        f"User input details: {input_summary}\n"
        "Provide practical and concise feedback based on the prediction and user input. "
        "Avoid lengthy explanations, avoid adding external resources or contacts, and ensure the feedback is "
        "empathetic but to the point. Focus on actionable suggestions. Do not use bold or special formatting. "
        "Include simple emojis to make the response more friendly."
    )
    response = ollama.chat(model='llama3.2:latest', messages=[{'role': 'user', 'content': message_content}])
    return response['message']['content']


if __name__ == '__main__':
    app.run(debug=True)
