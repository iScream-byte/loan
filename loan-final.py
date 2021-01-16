import numpy as np
import pandas as pd

# Load The Data set
df = pd.read_csv("loan-cleaned.csv", sep = ',')


column=['Gender',
        'Married',
        'Dependents',
        'Education',
        'Self_Employed',
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Credit_History',
        'Property_Area_Urban',
        'Property_Area_Semiurban']



df = pd.get_dummies(df, prefix=["Property_Area"], columns=["Property_Area"])
df["Property_Area_Urban"] = df['Property_Area_Urban'].astype(int)
df["Property_Area_Semiurban"] = df['Property_Area_Semiurban'].astype(int)

# Avoiding Dummy Variable Trap
df = df.drop(['Property_Area_Rural'], axis = 1) 

x=df[column]
y=df.iloc[:,10:11].values
y=np.where(y=='Y',1,0)


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(bootstrap = True,
                                       max_depth = 9,
                                       max_features = 2,
                                       min_samples_leaf = 3,
                                       min_samples_split = 8,
                                       n_estimators = 20)

random_forest.fit(x,y)

import pickle
pickle.dump(random_forest,open("loan-pkl-file.pkl","wb"))


