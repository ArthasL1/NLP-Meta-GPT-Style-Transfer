import pandas as pd
import sys
import os

training_size = float(sys.argv[1])
validation_size = float(sys.argv[2])
test_size_meta_training = float(sys.argv[3])
test_size_meta_testing = float(sys.argv[4])
data_folder_name = sys.argv[5]
base_path = sys.argv[6]

Meta_training = ['TFU', 'TPA', 'ATP', 'ARR', 'PPR']
Meta_testing = ['TPR', 'PTA', 'SBR']

# Create the directory if it doesn't exist
dir_path = base_path+data_folder_name
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(data_folder_name + " Directory Created!")
else:
    raise Exception("The directory %s already exists!" % dir_path)

meta_training_dir = os.path.join(dir_path, 'Meta_training')
meta_testing_dir = os.path.join(dir_path, 'Meta_testing')

# Create the directories
os.makedirs(meta_training_dir, exist_ok=True)
os.makedirs(meta_testing_dir, exist_ok=True)

for i in Meta_training:
    sub_class_dir = os.path.join(meta_training_dir, i)
    os.makedirs(sub_class_dir, exist_ok=True)
for i in Meta_testing:
    sub_class_dir = os.path.join(meta_testing_dir, i)
    os.makedirs(sub_class_dir, exist_ok=True)
# #Load the datasets
print("*******************************************************************")
for i in Meta_training:
    # Read csv from raw data
    test_df = pd.read_csv(base_path+'Raw_Data/'+i+'/test.tsv', sep='\t', header=None)
    train_df = pd.read_csv(base_path+'Raw_Data/'+i+'/train.tsv', sep='\t', header=None)
    valid_df = pd.read_csv(base_path+'Raw_Data/'+i+'/valid.tsv', sep='\t', header=None)

    # Combine the datasets
    combined_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    # Shuffle the combined dataset
    combined_df_shuffled = combined_df.sample(frac=1).reset_index(drop=True)
    print("The Meta-training Dataset "+i+" has been shuffled!")

    # 计算每个子集的大小
    total_rows = combined_df_shuffled.shape[0]
    train_end = int(total_rows * training_size)
    valid_end = train_end + int(total_rows * validation_size)
    test_end = valid_end + int(total_rows * test_size_meta_training)

    # Split the data into training, validation, and test sets
    train_set = combined_df_shuffled.iloc[:train_end]
    valid_set = combined_df_shuffled.iloc[train_end:valid_end]
    test_set = combined_df_shuffled.iloc[valid_end:test_end]
    remain = combined_df_shuffled.iloc[test_end:]
    print("Number of samples for Meta_training "+i+" in the training set: ", len(train_set))
    print("Number of samples for Meta_training "+i+" in the validation set: ", len(valid_set))
    print("Number of samples for Meta_training "+i+" in the test set: ", len(test_set))
    print("Number of samples for Meta_training "+i+" in the remaining set: ", len(remain))
    print("*******************************************************************")

    # Save the splits to new .tsv files
    train_set.to_csv(base_path+data_folder_name+'/Meta_training/'+i+'/train.tsv', sep='\t', index=False, header=False)
    valid_set.to_csv(base_path+data_folder_name+'/Meta_training/'+i+'/valid.tsv', sep='\t', index=False, header=False)
    test_set.to_csv(base_path+data_folder_name+'/Meta_training/'+i+'/test.tsv', sep='\t', index=False, header=False)
    remain.to_csv(base_path+data_folder_name+'/Meta_training/'+i+'/remain.tsv', sep='\t', index=False, header=False)

for i in Meta_testing:
    # Read csv from raw data
    test_df = pd.read_csv(base_path+'Raw_Data/'+i+'/test.tsv', sep='\t', header=None)
    train_df = pd.read_csv(base_path+'Raw_Data/'+i+'/train.tsv', sep='\t', header=None)
    valid_df = pd.read_csv(base_path+'Raw_Data/'+i+'/valid.tsv', sep='\t', header=None)

    # Combine the datasets
    combined_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    # Shuffle the combined dataset
    combined_df_shuffled = combined_df.sample(frac=1).reset_index(drop=True)
    print("The Meta-testing Dataset "+i+" has been shuffled!")

    total_rows = combined_df_shuffled.shape[0]
    test_end = int(total_rows * test_size_meta_testing)

    # Split the data into training, validation, and test sets
    test_set = combined_df_shuffled.iloc[:test_end]
    remain = combined_df_shuffled.iloc[test_end:]
    print("Number of samples for Meta_testing"+i+" in the test set: ", len(test_set))
    print("Number of samples for Meta_testing"+i+" in the remaining set: ", len(remain))
    print("*******************************************************************")

    # Save the splits to new .tsv files
    test_set.to_csv(base_path+data_folder_name+'/Meta_testing/'+i+'/test.tsv', sep='\t', index=False, header=False)
    remain.to_csv(base_path+data_folder_name+'/Meta_testing/'+i+'/remain.tsv', sep='\t', index=False, header=False)

