import pandas as pd
import numpy as np
result_csv=pd.read_csv("cohort.csv")
data=pd.DataFrame(result_csv,columns=["_id","Case_ID","Specimen_ID","Slide_ID","Radiology","Tumor","Topographic_Site","Specimen_Type","Weight","Tumor_Site","Tumor_Histological_Type","Tumor_Segment_Acceptable","Tumor_Percent_Tumor_Nuclei","Tumor_Percent_Total_Cellularity","Tumor_Percent_Necrosis","Normal_Free_of_Tumor","Progression_or_Recurrence","Genomics","Proteomics","Genomics_Available","GDC_Link","Proteomics_Available","PDC_Link","Gender","Age_at_Diagnosis","Ethnicity","Race","Vital_Status","Patholgy","HasRadiology","Pathology"])
print(data)
data1=data[data.Gender.isin(['female','male'])]
print(data1)
data2=data1[data1.Specimen_Type.isin(['tumor_tissue'])]
print(data2.shape[0])
print(data2)
data2.to_csv("processedv1.csv",index=False,columns=['Case_ID','Slide_ID','Topographic_Site','Gender','Tumor_Site'])



