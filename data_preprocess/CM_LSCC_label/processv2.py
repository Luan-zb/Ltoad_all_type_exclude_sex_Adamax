import pandas as pd
import numpy as np
result_csv=pd.read_csv("processedv1.csv")
data=pd.DataFrame(result_csv,columns=['Case_ID','Slide_ID','Topographic_Site','Gender','Tumor_Site'])
#Gender:  transform female->F   male->M
data['Gender']=data['Gender'].str.replace('female','F').replace('male','M')
data.to_csv("processedv2.csv",index=False,columns=['Case_ID','Slide_ID','Topographic_Site','Gender','Tumor_Site'])
#Tumor_site
data=data[data['Tumor_Site'].str.contains('metastatic')==False]
print(data)
data['Tumor_Site']='Primary'
#transform columns names
data2=data.rename(columns={'Case_ID':'case_id','Slide_ID':'slide_id','Topographic_Site':'label','Gender':'sex','Tumor_Site':'site'})
print(data2)
data2.to_csv("processedv3.csv",index=False,columns=['case_id','slide_id','label','sex','site'])


#noted:   iloc->整数标签   loc->标签





# data.iloc[data[:,3] == 'female'] = 'F'
# data.iloc[data[:,3] == 'male'] = 'M'
# print(data)
# data1=data[data.Gender.isin(['female','male'])]
# print(data1)
# data2=data1[data1.Specimen_Type.isin(['tumor_tissue'])]
# print(data2.shape[0])
# print(data2)
# data2.to_csv("processedv1.csv",index=False,columns=['Case_ID','Slide_ID','Topographic_Site','Gender','Tumor_Site'])
#
# #data2.to_csv("all_processed.csv",index=False)
# data2=data2.replace('female','F')
# data2=data2.replace('male','M')
# #data2.Gender=data2.Gender.map(lambda x: x[:1].upper())
# # data2.iloc[data2[:,'Gender'] == 'female'] = 'F'
# # data2.iloc[data2[:,'Gender'] == 'male'] = 'M'
# data2.to_csv("all_processed.csv",index=False,columns=['Case_ID','Slide_ID','Topographic_Site','Gender','Tumor_Site'])
# #data2.to_csv("all_processed.csv",index=False,columns=['slide_id','case_id','label','sex','site'])