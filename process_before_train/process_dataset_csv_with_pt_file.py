import pandas as pd
import os
#save the all_pt_file
pt_file_path="/home/daichuangchuang/Nature/Lclam/DATA_ROOT_DIR/DATA_DIR/pt_files"
files=os.listdir(pt_file_path)
# print(files)
all_slide_id=[]
for file in files:
    slide_id=file.split(".")[0]
    #print(slide_id)
    all_slide_id.append(slide_id)
#print(all_slide_id)


result_csv=pd.read_csv("../dataset_csv/cohort_all_type.csv")
#print(m)
data=pd.DataFrame(result_csv,columns=["case_id","slide_id","label"])
data=data[data["slide_id"].isin(all_slide_id)]
data.to_csv("../dataset_csv/cohort_all_type_processed.csv",index=False,columns=['case_id','slide_id','label'])


