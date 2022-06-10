import pandas as pd
import numpy as np
result_csv=pd.read_csv("CM_LSCC.csv")
data=pd.DataFrame(result_csv,columns=['slide_id','case_id','label','sex','site'])
print(data.sex.dtype)
data1=data[data.sex.isin(['F','M'])]
#print(data1)
data2=data1[data1.site.isin(['Primary','Metastatic'])]
print(data2.shape[0])
print(data2)
data2.to_csv("CM_LSCC_processed.csv",index=False)

