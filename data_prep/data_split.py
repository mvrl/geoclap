from config import cfg
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


meta_df = pd.read_csv(os.path.join(cfg.data_path,"final_metadata_with_captions.csv"))
##Split IDs
train_df, validate_df, test_df = np.split(meta_df.sample(frac=1, random_state=42), [int(.70*len(meta_df)), int(.8*len(meta_df))])
print("sample count:train/val/test for ratio 70:10:20",(len(train_df),len(validate_df)),len(test_df)) #(35554, 5079) 10159
splits_path = cfg.data_path
train_df.to_csv(os.path.join(splits_path,'train_df.csv'))
validate_df.to_csv(os.path.join(splits_path,'validate_df.csv'))
test_df.to_csv(os.path.join(splits_path,'test_df.csv'))