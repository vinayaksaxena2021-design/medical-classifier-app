import streamlit as st
from transformers import pipeline
import pandas as pd
st.set_page_config(page_title="AI Symptom Classifier", layout="wide")
st.title("🩺 AI Symptom Classifier")
st.markdown("""Welcome! Select your symptoms and indicate their severity to get predicted medical conditions.""")
#Conditions that the app can "diagnose"
condition_info = {
"Flu":{"description":"A contagious respiratory illness caused by influenza viruses, causing fever, cough, sore throat, and fatigue.","treatment":"Rest, fluids, antiviral medications if prescribed.","advice":"See a doctor if fever is high or symptoms worsen."},
"Migraine":{"description":"A neurological condition characterized by intense, throbbing headaches, often with nausea or sensitivity to light.","treatment":"Pain relief medications, rest in a quiet dark room, hydration.","advice":"Consult a doctor if migraines are frequent or severe."},
"Diabetes":{"description":"A chronic condition where blood sugar levels are abnormally high due to insulin issues.","treatment":"Blood sugar monitoring, medication or insulin, healthy diet, exercise.","advice":"Always consult a doctor for proper management."},
"Pneumonia":{"description":"Infection that inflames the air sacs in one or both lungs, causing cough, fever, and difficulty breathing.","treatment":"Antibiotics if bacterial, rest, fluids.","advice":"Seek immediate medical attention, especially if breathing is difficult."},
"Hypertension":{"description":"High blood pressure, often with no symptoms but can increase risk of heart disease and stroke.","treatment":"Lifestyle changes, medication if prescribed.","advice":"Consult a doctor for diagnosis and monitoring."},
"Arthritis":{"description":"Inflammation of joints causing pain, stiffness, and swelling, common in older adults.","treatment":"Pain relievers, anti-inflammatory medications, physical therapy.","advice":"See a doctor for proper evaluation and treatment plan."},
"Heart Attack":{"description":"A medical emergency where blood flow to the heart is blocked, causing chest pain and other symptoms.","treatment":"Immediate emergency care, medications, possible surgery.","advice":"Call emergency services immediately if suspected."},
"Common Cold":{"description":"A viral infection of the upper respiratory tract causing sneezing, runny nose, and mild cough.","treatment":"Rest, fluids, over-the-counter remedies.","advice":"See a doctor if symptoms persist or worsen."},
"Allergies":{"description":"Immune system reactions to substances like pollen, dust, or certain foods, causing sneezing, itching, or rashes.","treatment":"Antihistamines, avoiding triggers.","advice":"Consult a doctor if reactions are severe or frequent."},
"Bronchitis":{"description":"Inflammation of the bronchial tubes, causing coughing, mucus production, and discomfort in the chest.","treatment":"Rest, fluids, possibly antibiotics if bacterial.","advice":"See a doctor if cough persists or breathing is difficult."},
"Asthma":{"description":"A chronic respiratory condition that causes episodes of wheezing, shortness of breath, and coughing.","treatment":"Inhalers, avoiding triggers, medications.","advice":"Consult a doctor for proper management."},
"COVID-19":{"description":"A viral respiratory infection caused by SARS-CoV-2, symptoms include fever, cough, and loss of taste/smell.","treatment":"Rest, fluids, medications for symptoms, isolation.","advice":"Seek medical care if breathing difficulty or severe symptoms occur."},
"Anxiety":{"description":"A mental health condition involving excessive worry, nervousness, and fear.","treatment":"Therapy, stress management techniques, medication if prescribed.","advice":"Consult a mental health professional for proper care."},
"Depression":{"description":"A mood disorder causing persistent sadness, loss of interest, and fatigue.","treatment":"Therapy, support, medication if prescribed.","advice":"See a mental health professional for assessment and treatment."}
}
#Mapping the potential symptoms to conditions
symptom_map = {
"cough":["flu","pneumonia","bronchitis","common cold","COVID-19","asthma"],
"fever":["flu","pneumonia","COVID-19"],
"headache":["migraine","flu","COVID-19"],
"fatigue":["flu","diabetes","depression","COVID-19"],
"joint pain":["arthritis","flu"],
"shortness of breath":["asthma","pneumonia","heart attack","COVID-19"],
"sneezing":["common cold","allergies"],
"nausea":["migraine","flu","COVID-19"],
"chest pain":["heart attack","pneumonia","bronchitis"]
}
#Creates a checkbox for conditions along with a severity slider to assign weights to the selected symptoms
st.markdown("### Select your symptoms and their severity")
selected_symptoms = []
severity_scores = []
for symptom in symptom_map.keys():
    checked = st.checkbox(symptom)
    if checked:
        severity = st.slider(f"Severity of {symptom}", 1, 5, 3)
        selected_symptoms.append(symptom)
        severity_scores.append(severity)
#Running the selected symptoms through the LLM to generate confidence probabilites for each condition
if st.button("Predict Condition"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        candidate_conditions = {}
        for symptom, severity in zip(selected_symptoms, severity_scores):
            for condition in symptom_map[symptom]:
                candidate_conditions[condition] = candidate_conditions.get(condition,0) + severity
        sorted_conditions = sorted(candidate_conditions.items(), key=lambda x: x[1], reverse=True)
        top_label = sorted_conditions[0][0]
        confidence = round(sorted_conditions[0][1]/sum(severity_scores)*100,2)
        info = condition_info.get(top_label,{"description":"No description available","treatment":"No treatment information available","advice":"Consult a doctor"})
        st.success(f"**Top Predicted Condition:** {top_label} ({confidence}% confidence)")
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**Recommended Treatment:** {info['treatment']}")
        st.markdown(f"**Doctor Advice:** {info['advice']}")
        top3 = sorted_conditions[:3]
        df = pd.DataFrame({
            "Condition":[c[0] for c in top3],
            "Score":[c[1] for c in top3]
        })
        #Displaying top 3 conditions
        df.index = df.index+1
        st.markdown("### Top 3 Predicted Conditions")
        st.table(df)
