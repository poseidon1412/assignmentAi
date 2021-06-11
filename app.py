import streamlit as stl
import pandas as pd
import numpy as np
import xgboost
import pickle

pickle_in = open('Customer_churn_classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


stl.header('Customer Churn Classification')
stl.header('Developed and deployed by MUTEMA TAKUDZWA ')
stl.write('Please insert values to get Customer Churn prediction')
Tenure = stl.slider('Tenure',0.0,100.0)
monthlyCharges = stl.slider('Monthly Charges',0.0,100.0)
billingType = stl.text_input("Paperless Billing", "Type Yes or No")
paymentMethod = stl.text_input("Electronic Check Payment System", "Type Yes or No")




def prediction(Tenure, monthlyCharges, billingType, paymentMethod):  


	if billingType == 'Yes':
		billing = 1
	else:
		billing = 0
	if paymentMethod == 'Yes':
		payment = 1
	else:
		payment = 0			
	
	arr = np.array([[Tenure, monthlyCharges,billing, payment]])	
	
	prediction = classifier.predict(arr)
	if prediction == 1:
		prediction = stl.write('Customer will Churn or leave')
	
	if prediction == 0:
		prediction = stl.write('Customer will not Churn or leave')
		
	return prediction

if stl.button("Predict"):
	result = prediction(Tenure, monthlyCharges, billingType, paymentMethod)
	stl.success('THE END')









