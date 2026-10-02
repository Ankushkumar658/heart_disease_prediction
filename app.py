import streamlit as st
import pandas as pd
import joblib


# load saved model,scaler,and excepted columns
model=joblib.load("KNN_heart.pkl")
scaler=joblib.load("scaler.pkl")
expected_columns=joblib.load("columns.pkl")

st.title("🫀Heart Diseases Risk prediction ")
st.markdown("Enter Patient medical parameters below to evaluate heart diseases risk.")

col1,col2,col3= st.columns(3)

with col1:
    age=st.slider("Age",18,100,40)
    sex=st.selectbox("SEX",['M','F'])
    chest_pain=st.selectbox("Chest pain Type",["ATA","NAP","TA","ASY"])

with col2:    
    resting_bp=st.number_input("Resting Blood pressure(mm Hg)",80,200,120)
    cholesterol=st.number_input("Cholesterol (mg/dl)",100,600,200)
    fasting_bs=st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])
    resting_ecg=st.selectbox("Resting ECG",["Normal","ST","LVH"])

with col3:    
    max_hr=st.slider("Max Heart Rate",60,220,150)
    exercise_angina=st.selectbox("Exercise-Induced Angina",["Y","N"])
    oldpeak=st.slider("Oldpeak (ST Depression)",0.0,6.0,1.0)
    st_slope=st.selectbox("ST Slope",["Up","Flat","Down"])


if st.button ("🔍 Analyze Heart Risk",type="primary",use_container_width=True):
    raw_input={
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_'+chest_pain:1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_'+ st_slope: 1
    } 

    input_df=pd.DataFrame([raw_input])  

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0

    input_df=input_df[expected_columns]  

    scaled_input=scaler.transform(input_df)
    prediction=model.predict(scaled_input)[0]     


    if prediction==1:
        st.error("⚠️ High risk of a Heart Diseases")  
    else:
        st.success("✅Low Risk of Heart Diseases")





        
    
        
        



