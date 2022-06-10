import pandas as pd
import numpy as np
result_csv=pd.read_csv("processedv1.csv")
data=pd.DataFrame(result_csv,columns=['Case_ID','Slide_ID','Tumor_Histological_Type','Gender','Tumor_Site'])
#Gender:  transform female->F   male->M
data['Gender']=data['Gender'].str.replace('female','F').replace('male','M')

#Tumor_site
data=data[data['Tumor_Site'].str.contains('metastatic')==False]
data['Tumor_Site']='Primary'


#Tumor_Histological_Type
data=data[data['Tumor_Histological_Type'].str.contains('cell renal cell carcinoma')==True]
data['Tumor_Histological_Type']='CCRCC'
print(data)
data.to_csv("processedv2_1.csv",index=False,columns=['Case_ID','Slide_ID','Tumor_Histological_Type','Gender','Tumor_Site'])









