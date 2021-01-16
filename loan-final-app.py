import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(layout="wide")
df=pd.read_csv("loan-cleaned.csv")
st.write("""
# Loan Prediction App
This app predicts the **If a customer is eligible for loan or not**!""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
Enter customer properties
""")

def user_input_features():
        Gender = st.sidebar.selectbox("Enter Sex",('Male','Female'))
        Married = st.sidebar.selectbox("Marital Status",('Married','Unmarried'))
        Dependents = st.sidebar.selectbox("Enter number of dependents",[0,1,2,3])
        Education = st.sidebar.selectbox('Education',('Graduate','Non-Graduate'))
        Self_Employed = st.sidebar.selectbox('Self Employed?', ("Yes","No"))
        ApplicantIncome =st.sidebar.number_input("Enter Applicant's Income")
        CoapplicantIncome = st.sidebar.number_input('Co Applicants Income')
        LoanAmount = st.sidebar.number_input('Loan Amount')
        Loan_Amount_Term=st.sidebar.selectbox('Loan Amount Term',sorted(df.Loan_Amount_Term.unique()))
        Credit_History=st.sidebar.selectbox('Credit history',sorted(df.Credit_History.unique()))
        Property_Area=st.sidebar.selectbox('Property Area',('Urban','Semiurban'))
        data = {    
                'Gender':Gender,
                'Married':Married,
                'Dependents': Dependents,
                'Education': Education,
                'Self_Employed': Self_Employed,
                'ApplicantIncome': ApplicantIncome,
                'CoapplicantIncome': CoapplicantIncome,
                'LoanAmount':LoanAmount,
                'Loan_Amount_Term':Loan_Amount_Term,
                'Credit_History':Credit_History,
                'Property_Area':Property_Area
                
                }
        features = pd.DataFrame(data, index=[0])

        return features
df_raw=df.copy()
df = df.drop(columns=['Loan_Status'])
input_df = user_input_features()

X=np.where(input_df["Gender"]=='Male',1,0)
input_df['Gender']=X

X=np.where(input_df["Married"]=='Married',1,0)
input_df['Married']=X

X=np.where(input_df["Education"]=='Graduate',1,0)
input_df['Education']=X

X=np.where(input_df["Self_Employed"]=='Yes',1,0)
input_df['Self_Employed']=X


df = pd.concat([input_df,df],axis=0)

encode = ['Property_Area']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

df = df.drop(columns=['Property_Area_Rural'])
df = df[:1]
st.subheader('**Applied customer characteristics**')
st.table(input_df)

load_clf = pickle.load(open('loan-pkl-file.pkl', 'rb'))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('*----------Prediction----------*')
st.subheader('Applied customer falls under the category stated below:')
eligibility = np.array(['No','Yes'])
st.write(eligibility[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
