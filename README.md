# DriverMONI: classification-of-cancer-driver-gene
Goal of this project is classification of specific cancer driver gene.
In our study, we classify 14 cancers driver gene by deep neural network.
# make input data
if you want to make input data, you can use data_preprocessor.py. 
Sample data for making input data are in sample_data folder. 
Before you run data_preprocessor.py, you have to decompress humannet_v2_and_v3.zip in for_ref folder. 
Cancer_input_data.csv(ex: BRCA_input_data.csv) is input data file that made by data_preprocessor.py.
## format of files that needed for making input data
### mutation data
In mutation data, maf format file is needed for making input data

### gene expression data
In gene expression data, format like sample data in sample_data folder is needed for making input data

## argument
### 1. cancer_name 
cancer_name argument is meaning TCGA cancer name.(One of the following: BRCA, PAAD, PRAD)
### 2. exp_data_dir
directory of gene expresssion data
### 3. muta_data_dir
directory of gene mutation data(maf file)
### example
    python3 data_preprocessor.py -cancer_name BRCA -exp_data_dir /mnt/disk1/driver_gene/data/bf_preprocess/BRCA/gene_exp/BRCA-gene-exp.tsv -muta_data_dir /mnt/disk1/driver_gene/data/bf_preprocess/BRCA/gene_muta/BRCA.varscan.maf

# training by DriverMONI
if you want to train your data, you can use CV_multimodal_compare.py
## argument
### 1.cancer_name
cancer_name argument is meaning TCGA cancer name.(One of the following: BRCA, PAAD, PRAD)

### example 
    python3 CV_multimodal_compare.py -cancer_name BRCA 


