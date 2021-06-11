import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,make_scorer
import pickle

df = pd.read_csv('datasets_13996_18858_WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.drop('customerID',axis=1,inplace=True)
df.loc[(df['TotalCharges'] == ' '),'TotalCharges'] = 0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df.loc[(df['Churn'] == 'No'),'Churn'] = 0
df.loc[(df['Churn'] == 'Yes'),'Churn'] = 1
df['Churn'] = pd.to_numeric(df['Churn'])
df.replace('','_',regex=True,inplace=True)
df_encoded = pd.get_dummies(df,columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'], drop_first=True)


#Determining numerical variable
num = df_encoded[["tenure","MonthlyCharges","TotalCharges","Churn"]]
#Filtering using Pearson Correlation
plt.figure(figsize=(12,10))
cor = num.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Correlation with output variable
cor_target = abs(cor["Churn"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
relevant_features

print(df[["tenure","MonthlyCharges"]].corr())
print(df[["tenure","TotalCharges"]].corr())
print(df[["MonthlyCharges","TotalCharges"]].corr())

#Determining categorical variable
cat = df_encoded[['Churn', 'gender_Male',
       'SeniorCitizen_1', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
       'MultipleLines_No phone service', 'MultipleLines_Yes',
       'InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
       'OnlineBackup_No internet service', 'OnlineBackup_Yes',
       'DeviceProtection_No internet service', 'DeviceProtection_Yes',
       'TechSupport_No internet service', 'TechSupport_Yes',
       'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No internet service', 'StreamingMovies_Yes',
       'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']] 
#those are only some examples of the actual variables
kendalls = cat.corr(method='kendall')
relevant_features = kendalls[kendalls>0.1]
# kendalls
relevant_features

df_encoded[['SeniorCitizen_1','PaperlessBilling_Yes','PaymentMethod_Electronic check','InternetService_Fiber optic','Churn']].corr()

X = df_encoded[['tenure', 'MonthlyCharges','PaperlessBilling_Yes','PaymentMethod_Electronic check']]
y = df_encoded["Churn"]
X.columns = X.columns.str.replace(' ','_')
X.head()

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42,stratify=y)
#Feature scaling to scale the data or commonly known as normalization
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = xgb.XGBClassifier(objective='binary:logistic',
                            seed = 42,
                            gamma=0.25,
                            learn_rate=0.1,
                            max_depth = 4,
                            reg_lambda=10,
                            scale_pos_weight=3,
                            subsample=0.9,
                            colsample_bytree=0.5)
model.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_rounds=10,
            eval_metric= 'aucpr',
            eval_set=[(X_test,y_test)])

plot_confusion_matrix(model,
                        X_test,
                        y_test,
                        values_format='d',
                        display_labels=['Did not leave','left'])

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

y_pre = model.predict_proba(X_test)[:,1]
print( 'AUC value:', roc_auc_score(y_test,y_pre))

Pkl_Filename = "Customer_churn_classifier.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)