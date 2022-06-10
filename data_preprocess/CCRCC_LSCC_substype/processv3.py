import pandas as pd
import numpy as np
result_csv_1=pd.read_csv("processedv2_1.csv")
data1=pd.DataFrame(result_csv_1,columns=['Case_ID','Slide_ID','Tumor_Histological_Type','Gender','Tumor_Site'])

result_csv_2=pd.read_csv("processedv2_2.csv")
data2=pd.DataFrame(result_csv_2,columns=['Case_ID','Slide_ID','Tumor_Histological_Type','Gender','Tumor_Site'])

result=result_csv_1.append(result_csv_2)

print(result.shape)

#transform columns names
data3=result.rename(columns={'Case_ID':'case_id','Slide_ID':'slide_id','Tumor_Histological_Type':'label','Gender':'sex','Tumor_Site':'site'})
print(data3)
data3.to_csv("processedv3.csv",index=False,columns=['case_id','slide_id','label','sex','site'])