import pandas as pd
import numpy as np
result_csv=pd.read_csv("processedv1.csv")
data=pd.DataFrame(result_csv,columns=['Case_ID','Slide_ID','Topographic_Site'])


#transform columns names
data2=data.rename(columns={'Case_ID':'case_id','Slide_ID':'slide_id','Topographic_Site':'label'})
print(data2)
data2.to_csv("processedv2.csv",index=False,columns=['case_id','slide_id','label'])


