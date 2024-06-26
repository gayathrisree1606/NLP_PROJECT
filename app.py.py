import streamlit as st
import joblib

# Load the trained model for suicide ideation
model = joblib.load("classification_suicide.pkl")

# Define the label mapping for binary classification
labels = {0: 'not suicidal', 1: 'suicidal'}

# Set the title of the Streamlit app
st.title("Suicide Ideation detection from Text")

# Create a text area for user input
user_input = st.text_area("Enter your text here:")

# Add a predict button
if st.button("Predict"):
    # Predict the suicide ideation label for the given text
    predicted_label = model.predict([user_input])[0]
    
    # Map the predicted label to a more readable format
    predicted_suicide_ideation_label = labels[predicted_label]
    
    # Display the prediction
    st.info(f"The model predicts that the person has {predicted_suicide_ideation_label} ideation.")

# Optional: Add a footer or additional information
st.write("This tool is for educational purposes and should not be used as a substitute for professional mental health advice.")
