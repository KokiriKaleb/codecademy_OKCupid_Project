from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack

app = Flask(__name__)

# Load the saved model, vectorizer, and feature names
model = joblib.load('ridge_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
feature_names = joblib.load('feature_names.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    form_data = request.form.to_dict()
    input_df = pd.DataFrame([form_data])

    # --- Feature Engineering for the input ---
    input_df['all_essays'] = input_df['essay0']

    # Create a DataFrame with all the model's feature columns, initialized to 0
    processed_input = pd.DataFrame(columns=feature_names, index=[0])
    processed_input.fillna(0, inplace=True)

    # Fill in all known values from the form and add reasonable defaults
    # Numerical features
    processed_input['height'] = float(input_df.loc[0, 'height'])
    processed_input['income'] = int(input_df.loc[0, 'income'])
    processed_input['essay_length'] = len(input_df.loc[0, 'all_essays'])
    processed_input['word_count'] = len(input_df.loc[0, 'all_essays'].split())
    # Hardcode reasonable defaults for other numerical features
    processed_input['last_online_year'] = 2012 
    processed_input['last_online_month'] = 6
    processed_input['last_online_dayofweek'] = 3

    # One-hot encode the user's input
    edu_feature = 'education_' + input_df.loc[0, 'education']
    if edu_feature in processed_input.columns:
        processed_input[edu_feature] = 1

    offspring_feature = 'offspring_cleaned_' + input_df.loc[0, 'offspring']
    if offspring_feature in processed_input.columns:
        processed_input[offspring_feature] = 1
        
    # TF-IDF transformation for the essay
    essay_tfidf = tfidf_vectorizer.transform(input_df['all_essays'])

    # Combine all features
    final_features = hstack([processed_input.astype(float).values, essay_tfidf])

    # Make the prediction
    prediction = model.predict(final_features)
    predicted_age = int(round(prediction[0]))
    
    # Return the result
    return render_template('index.html', prediction_text=f'Predicted Age: {predicted_age}')

if __name__ == "__main__":
    app.run(debug=True)