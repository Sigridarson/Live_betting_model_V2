
import streamlit as st
import pandas as pd

def live_bet_prediction(ht_stats, clf_model, reg_model):
    input_df = pd.DataFrame([ht_stats])
    outcome_probs = clf_model.predict_proba(input_df)[0]
    outcome_classes = clf_model.classes_
    predicted_outcome = outcome_classes[outcome_probs.argmax()]
    confidence = round(outcome_probs.max(), 2)
    predicted_2nd_half_goals = reg_model.predict(input_df)[0]
    first_half_goals = ht_stats['HTHG'] + ht_stats['HTAG']
    predicted_total_goals = predicted_2nd_half_goals + first_half_goals
    ou_predictions = {
        'Over/Under 2.5': 'Over 2.5' if predicted_total_goals > 2.5 else 'Under 2.5',
        'Over/Under 3.5': 'Over 3.5' if predicted_total_goals > 3.5 else 'Under 3.5',
        'Over/Under 4.5': 'Over 4.5' if predicted_total_goals > 4.5 else 'Under 4.5',
    }
    results = {
        'Predicted Outcome (1/X/2)': predicted_outcome,
        'Confidence': confidence,
        'Predicted 2nd Half Goals': round(predicted_2nd_half_goals, 2),
        'Predicted Total Goals': round(predicted_total_goals, 2),
    }
    results.update(ou_predictions)
    return results

def main_ui(clf_model=None, reg_model=None):
    st.title("⚽ Live Match Betting Predictor")
    st.header("Enter Half-Time Stats")
    ht_stats = {
        'HTHG': st.number_input("Half-Time Home Goals", min_value=0, step=1),
        'HTAG': st.number_input("Half-Time Away Goals", min_value=0, step=1),
        'HS': st.number_input("Home Shots", min_value=0, step=1),
        'AS': st.number_input("Away Shots", min_value=0, step=1),
        'HST': st.number_input("Home Shots on Target", min_value=0, step=1),
        'AST': st.number_input("Away Shots on Target", min_value=0, step=1),
        'HC': st.number_input("Home Corners", min_value=0, step=1),
        'AC': st.number_input("Away Corners", min_value=0, step=1),
        'HF': st.number_input("Home Fouls", min_value=0, step=1),
        'AF': st.number_input("Away Fouls", min_value=0, step=1),
        'HY': st.number_input("Home Yellow Cards", min_value=0, step=1),
        'AY': st.number_input("Away Yellow Cards", min_value=0, step=1),
        'HR': st.number_input("Home Red Cards", min_value=0, step=1),
        'AR': st.number_input("Away Red Cards", min_value=0, step=1),
    }
    if st.button("Predict Outcome & Markets"):
        if clf_model is None or reg_model is None:
            st.error("This demo version does not include trained models.")
        else:
            prediction = live_bet_prediction(ht_stats, clf_model, reg_model)
            st.subheader("📊 Prediction Results")
            for key, value in prediction.items():
                st.write(f"**{key}:** {value}")

if __name__ == "__main__":
    main_ui()
