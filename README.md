# DriverDNN: classification-of-cancer-driver-gene
Goal of this project is classification of specific cancer driver gene.
In our study, we classify 14 cancers driver gene by deep neural network.
# make input data
if you want to make input data, you can use data_preprocessor.py. 
Sample data for making input data are in sample_data folder. 
Before you run data_preprocessor.py, you have to decompress humannet_ens_PPI.zip in for_ref folder. 
Cancer_input_data.csv(ex: BRCA_input_data.csv) is input data file that made by data_preprocessor.py.
## format of files that needed for making input data
### mutation data
In mutation data, maf format file is needed for making input data

### gene expression data
In gene expression data, format like sample data in sample_data folder is needed for making input data

## argument
### 1. cancer_name 
cancer_name argument is meaning TCGA cancer name.(One of the following: BRCA, PAAD, PRAD)
### 2. out_dir
directory for save data file
### 3. exp_data_dir
directory of gene expresssion data
### 4. muta_data_dir
directory of gene mutation data(maf file)
### example
    python3 data_preprocessor.py -cancer_name BRCA -out_dir /home/bml_pjh/ -exp_data_dir /mnt/disk1/driver_gene/data/bf_preprocess/BRCA/gene_exp/BRCA-gene-exp.tsv -muta_data_dir /mnt/disk1/driver_gene/data/bf_preprocess/BRCA/gene_muta/BRCA.varscan.maf

# training by DriverDNN
if you want to train your data, you can use DriverDNN_training.py
## argument
### 1.cancer_name
cancer_name argument is meaning TCGA cancer name.(One of the following: BRCA, PAAD, PRAD)

### 2.data_directory 
when you have one dataset and you want to seperate the dataset by train and test,don't use test_data_directory argument, and use only data_directory argument. 
we divide train,test dataset by 3 fold stratified cross validation. So we make three dataset. by using 3 fold stratified cross validation, you can get three model by one dataset.
### example 
    python3 DriverDNN_training.py -cancer_name BRCA -data_directory sample_data/TCGA_sample_BRCA_input.csv -lr 0.001 -epoch 10 -batch 64 -early_stop 5 -result_folder_name result 

### 3.lr
lr is meaning learning rate of model training.

### 4.epoch
epoch is meaning epoch of model training.

### 5.batch
batch argument is meaning batch size of model training.

### 6.early_stop
early_stop argument is number of epochs with no improvement after which training will be stopped.

### 7.result_folder_name
result_folder_name argument is name of folder that save result and model.

### 8.test_data_directory
when you have train dataset and test dataset separately, you can use data_directory argument for train dataset and use test_data_directory for test dataset.
### example 
    python3 DriverDNN_training.py -cancer_name BRCA -data_directory sample_data/TCGA_sample_BRCA_input.csv -lr 0.001 -epoch 10 -batch 64 -early_stop 5 -result_folder_name result -test_data_directory sample_data/CPTAC_BRCA_input.csv

# predict by DriverDNN model which is already trained
if you want to predict your data by our trained model, you can use DriverDNN_predict.py
### example 
    python3 DriverDNN_predict.py -cancer_name BRCA -test_data_directory sample_data/TCGA_sample_BRCA_input.csv -result_folder_name result_1

## argument
### 1.cancer_name
cancer_name argument is meaning TCGA cancer name.

### 2.test_data_directory
directory of data that you wanna predict.

### 3.result_folder_name
folder name that you wanna save result file.
