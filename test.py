import streamlit as st
import pandas as pd
import pickle

df=pd.read_csv("personality_datasert.csv")
df.head()

with open("personality_model1.pkl", 'rb')as file:
    model=pickle.load(file)

st.title("Personality Prediction (Introvert vs Extrovert)")
st.write("This app uses a Logistic Regression model to predict personality type.")

st.write("### Manual Input for Prediction")

time_spent = st.number_input("Time spent alone (hours)", min_value=0, max_value=24, value=4)
stage_fear = st.selectbox("Do you have stage fear?", ["Yes", "No"])
social_events = st.number_input("Number of social events attended", min_value=0, value=2)
going_out = st.number_input("Frequency of going outside", min_value=0, value=5)
drained = st.selectbox("Do you feel drained after socializing?", ["Yes", "No"])
friends = st.number_input("Friends circle size", min_value=0, value=10)
posts = st.number_input("Social media post frequency", min_value=0, value=3)


# Convert categorical inputs to match dataset format
input_data = pd.DataFrame([{
    "Time_spent_Alone": time_spent,
    "Stage_fear": stage_fear,
    "Social_event_attendance": social_events,
    "Going_outside": going_out,
    "Drained_after_socializing": drained,
    "Friends_circle_size": friends,
    "Post_frequency": posts
}])

if st.button("Predict Personality"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Personality: **{prediction}**")