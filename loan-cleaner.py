import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Load The Data set
df = pd.read_csv("loan_data_set.csv", sep = ',')

categorical_columns = ['Gender',
                       'Married',
                       'Education',
                       'Self_Employed',
                       'Property_Area',
                       'Loan_Status',
                       ]

for column in categorical_columns:
    df[column] = df[column].astype('category')
    
df_copy = df.copy()
df = df.dropna()

# Categorical Variable Handling
X=np.where(df["Gender"]=='Male',1,0)
df['Gender']=X

X=np.where(df["Married"]=='Yes',1,0)
df['Married']=X

X=np.where(df["Education"]=='Graduate',1,0)
df['Education']=X

X=np.where(df["Self_Employed"]=='Yes',1,0)
df['Self_Employed']=X

# Dummy variable and Label_Encoding
#df = pd.get_dummies(df, prefix=["Property_Area"], columns=["Property_Area"])
#df["Property_Area_Urban"] = df['Property_Area_Urban'].astype(int)
#df["Property_Area_Semiurban"] = df['Property_Area_Semiurban'].astype(int)

# Avoiding Dummy Variable Trap
#df = df.drop(['Property_Area_Rural'], axis = 1) 
#df["Property_Area_Rural"] = df['Property_Area_Rural'].astype(int)  ###### ----(Convert Objet to int32)

# Drop Unnecessary features
df = df.drop(['Loan_ID'], axis = 1) 

# Replace Columns Values(3+) with 3
df["Dependents"] = df["Dependents"].replace('3+', '3')
df["Dependents"] = df['Dependents'].astype(int)  ###### ----(Convert Objet to int32)

df.to_csv("loan-cleaned.csv")




