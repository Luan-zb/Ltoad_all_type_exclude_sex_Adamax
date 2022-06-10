import os
print(os.getcwd())
os.chdir("/home/daichuangchuang/Nature/Lclam")
#segmentation and patch
os.system("python create_patches_fp.py --source DATA_DIRECTORY_one_wsi --save_dir RESULTS_DIRECTORY_one_wsi --patch_size 256 --seg --patch --stitch")
#feature extraction
os.system("CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir RESULTS_DIRECTORY_one_wsi/  --data_slide_dir DATA_DIRECTORY_one_wsi  --csv_path RESULTS_DIRECTORY_one_wsi/process_list_autogen.csv --feat_dir FEATURES_DIRECTORY_one_wsi --batch_size 512 --slide_ext .svs")

os.chdir("/home/daichuangchuang/Nature/Ltoad_one_wsi")
#predict one wsi on model
os.system("CUDA_VISIBLE_DEVICES=0 python model_predict_one_wsi.py --drop_out --k 1 --models_exp_code dummy_mtl_sex_s1 --save_exp_code dummy_mtl_sex_s1_eval --task study_v2_mtl_sex  --results_dir results --data_root_dir /home/daichuangchuang/Nature/CLAM/DATA_ROOT_DIR")
