import streamlit as st
import pickle
import pandas as pd

teams = [
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Chennai Super Kings",
    "Gujarat Titans",
    "Lucknow Super Giants",
    "Kolkata Knight Riders",
    "Punjab Kings",
    "Mumbai Indians",
]

cities = [
    "Ahmedabad",
    "Kolkata",
    "Mumbai",
    "Navi Mumbai",
    "Pune",
    "Dubai",
    "Sharjah",
    "Abu Dhabi",
    "Delhi",
    "Chennai",
    "Hyderabad",
    "Visakhapatnam",
    "Bengaluru",
    "Jaipur",
    "Bangalore",
    "Raipur",
    "Ranchi",
    "Cuttack",
    "Johannesburg",
    "Centurion",
    "Durban",
    "Bloemfontein",
    "Port Elizabeth",
    "Kimberley",
    "East London",
    "Cape Town",
]
st.title("IPL Win Predictor")

pipe = pickle.load(
    open(
        "C:/Users/91956/Dropbox/PC/Desktop/python/python/MachineLearning/ML_projects/pipe.pkl", "rb"
    )
)

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select Batting Team", teams)
with col2:
    bowling_team = st.selectbox("Select Bowling Team", teams)

selected_city = st.selectbox("Select_City", cities)
target = st.number_input("Target")

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input("Score")
with col4:
    overs = st.number_input("Overs Completed")
with col5:
    wickets = st.number_input("Wickets")

runs_left = 0
balls_left = 0
wickets = 0
crr = 0
rrr = 0

if st.button("Predict Probability"):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = runs_left * 6 / balls_left


input_df = pd.DataFrame(
    {
        "BattingTeam": [batting_team],
        "BowlingTeam": [bowling_team],
        "City": [selected_city],
        "runs_left": [runs_left],
        "balls_left": [balls_left],
        "wickets": [wickets],
        "total_run_x": [target],
        "crr": [crr],
        "rrr": [rrr],
    }
)
result = pipe.predict_proba(input_df)
loss = result[0][0]
win = result[0][1]
st.header(batting_team + "-" + str(round(win * 100))+"%")
st.header(bowling_team + "-" + str(round(loss * 100))+"%")
