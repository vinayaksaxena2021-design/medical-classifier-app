import streamlit as st
from transformers import pipeline
import pandas as pd
st.set_page_config(page_title="AI Symptom Classifier (LLM Edition)", layout="wide")
st.title("🩺 AI Symptom Classifier — LLM Edition")
st.markdown("""Welcome! Select your symptoms and indicate their severity.  
The app will use a language model to predict the most likely medical conditions.""")
st.divider()
# Load LLM pipeline
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2", max_new_tokens=100)
model = load_model()
# Define condition info
condition_info = {
    "Flu": {"description": "A contagious respiratory illness caused by influenza viruses, causing fever, cough, sore throat, and fatigue.",
            "treatment": "Rest, fluids, antiviral medications if prescribed.",
            "advice": "See a doctor if fever is high or symptoms worsen."},
    "Migraine": {"description": "A neurological condition characterized by intense, throbbing headaches, often with nausea or sensitivity to light.",
                 "treatment": "Pain relief medications, rest in a quiet dark room, hydration.",
                 "advice": "Consult a doctor if migraines are frequent or severe."},
    "Pneumonia": {"description": "Infection that inflames the air sacs in one or both lungs, causing cough, fever, and difficulty breathing.",
                  "treatment": "Antibiotics if bacterial, rest, fluids.",
                  "advice": "Seek immediate medical attention if breathing is difficult."},
    "Common Cold": {"description": "A viral infection of the upper respiratory tract causing sneezing, runny nose, and mild cough.",
                    "treatment": "Rest, fluids, over-the-counter remedies.",
                    "advice": "See a doctor if symptoms persist or worsen."},
    "COVID-19": {"description": "A viral respiratory infection caused by SARS-CoV-2.",
                 "treatment": "Rest, fluids, symptom management, isolation.",
                 "advice": "Seek medical care if severe symptoms occur."},
    "Asthma": {"description": "A chronic condition causing episodes of wheezing, shortness of breath, and coughing.",
               "treatment": "Inhalers, avoiding triggers, medications.",
               "advice": "Consult a doctor for proper management."},
    "Allergies": {"description": "Immune system reaction to substances like pollen, dust, or certain foods.",
                  "treatment": "Antihistamines, avoiding triggers.",
                  "advice": "Consult a doctor if reactions are severe or frequent."},
}
# Symptom input
symptoms = ["cough", "fever", "headache", "fatigue", "joint pain", 
            "shortness of breath", "sneezing", "nausea", "chest pain"]
st.markdown("### Select your symptoms and their severity")
selected_symptoms = []
severity_scores = []
for s in symptoms:
    checked = st.checkbox(s)
    if checked:
        severity = st.slider(f"Severity of {s}", 1, 5, 3)
        selected_symptoms.append(s)
        severity_scores.append(severity)
# Run LLM prediction
if st.button("Predict Condition"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        symptom_text = ", ".join([f"{s} (severity {sev})" for s, sev in zip(selected_symptoms, severity_scores)])
        prompt = f"The patient has {symptom_text}. Based on this, list the top 3 most likely medical conditions."
        with st.spinner("Analyzing symptoms using LLM..."):
            result = model(prompt)[0]["generated_text"]
        st.markdown("### 🧠 LLM Prediction Output")
        st.write(result)
        import re
        found_conditions = re.findall(r"(Flu|Migraine|Pneumonia|Common Cold|COVID-19|Asthma|Allergies)", result, re.IGNORECASE)
        found_conditions = list(dict.fromkeys([c.capitalize() for c in found_conditions]))[:3]
        if found_conditions:
            st.success(f"**Top Predicted Condition:** {found_conditions[0]}")
            top_conditions = found_conditions[:3]
            df = pd.DataFrame({"Condition": top_conditions})
            df.index = df.index + 1
            st.table(df)
            for cond in top_conditions:
                info = condition_info.get(cond, None)
                if info:
                    with st.expander(f"🩺 {cond} — Details"):
                        st.markdown(f"**Description:** {info['description']}")
                        st.markdown(f"**Treatment:** {info['treatment']}")
                        st.markdown(f"**Doctor Advice:** {info['advice']}")
        else:
            st.info("The model did not produce a recognizable condition name. Try adding more symptoms or adjusting severity.")
if st.button("Reset"):
    st.experimental_rerun()
