import pandas as pd
import numpy as np
from pyensembl import EnsemblRelease
import re
import os
from multiprocessing import Process
import time
import sys
import json
import argparse
import pickle

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cancer_name', '--dis_name', help="TCGA cancer name", required=True)
    parser.add_argument('-exp_data_dir', '--exp_dir', help="gene expression data directory",default='', required=False)
    parser.add_argument('-muta_data_dir', '--muta_dir', help="maf file directory",default='', required=False)

    args = vars(parser.parse_args())
    return args

class make_input():
    def __init__(self):
        self.data=EnsemblRelease(102)

    def step_2(self,inputs):
        def cut_threshold(data,threshold):
            idxs=[idx for idx in range(data.shape[0]) if data.loc[idx,'LLS']>threshold]
            data=data.loc[idxs,:]
            return data
        def make_inter_dict(data):
            data2=pd.concat((data.loc[:,'entrez_id_1'].drop_duplicates(),data.loc[:,'entrez_id_2'].drop_duplicates()),axis=0)
            data2=data2.drop_duplicates()
            dict_={}
            for dis in list(data2):
                dict_[dis]=[]
            for gene1,gene2 in np.array(data.iloc[:,[1,2]]):
                dict_[gene1].append(gene2)
                dict_[gene2].append(gene1)
            return dict_
        dis_name=inputs['dis_name']# get disease name


        data_exp=pd.read_csv(inputs['exp_dir'],sep='\t')
        gene_exp=[]

        for i in range(data_exp.shape[0]):
            if '.' in data_exp.iloc[i,0]:
                gene_exp.append(data_exp.iloc[i,0].split('.')[0])
            else:
                gene_exp.append(data_exp.iloc[i,0])
        data_exp['Ensembl_ID']=gene_exp


        data_exp=data_exp.set_index('Ensembl_ID')



        sample_id=list(data_exp.columns)
        bad_sample_id=[]
        good_sample_id=[]
        for i in range(len(sample_id)):

            if int(sample_id[i][-3:-1])>=10:
                good_sample_id.append(sample_id[i])
            else:
                bad_sample_id.append(sample_id[i])
                
        good_samples=data_exp.loc[:,good_sample_id]
        good_sample_sum=good_samples.sum(axis=1)
        good_sample_sum=good_sample_sum/len(good_sample_id)

        bad_samples=data_exp.loc[:,bad_sample_id]
        bad_sample_sum=bad_samples.sum(axis=1)
        bad_sample_sum=bad_sample_sum/len(bad_sample_id)
        
        deg__=bad_sample_sum-good_sample_sum
        if deg__.isna().sum()>=1:
            print('nan value in deg feature')
            quit()










        data=pd.read_csv(inputs['step_1_data'])
        ens_sym={}
        for i in range(data.shape[0]):
            ens_sym[data.loc[i,'gene_ens']]=data.loc[i,'gene_symbol']
        idx_=0
        col_l=list(data.columns)
        for col in list(data.columns):
            if col=='gene_ens':
                col_l[idx_]='gene'
            idx_+=1
        data.columns=col_l
        data.drop(['gene_symbol'],axis=1,inplace=True)
        

        data.index=list(range(data.shape[0]))
        protein=pd.read_csv('./for_ref/hsa_id_conv.txt',sep='\t')
        protein_filter=protein[protein['gene_biotype']=='protein_coding']
        pro_idx=[]
        iter_protein_filter=1
        total_iter=data.shape[0]
        for i in range(data.shape[0]):
            if data.loc[i,'gene'] in list(protein_filter.loc[:,'ensembl_gene_id']):
                pro_idx.append(i)
            if (iter_protein_filter/total_iter)*100 != 100:
                per=str((iter_protein_filter/total_iter)*100)
                if  per.index('.')==1:
                    print(per[:3] + '% filtering protein coding gene processed..',end='\r',flush=True)
                else:
                    print(per[:4] + '% filtering protein coding gene processed..',end='\r',flush=True)
            else:
                print(str((iter_protein_filter/total_iter)*100)[:5] + '% filtering protein coding gene processed..')
            iter_protein_filter+=1
        data=data.loc[pro_idx,:]
        data.index=list(range(data.shape[0]))






        ss=data.groupby('gene')
        total_gene=len(list(set(list(data.loc[:,'gene']))))
        last_data=[]
        for_col=0
        esang=0
        iter=1
        for i,k in enumerate(ss):

            data_=k[1].iloc[:,2:]
            if for_col==0:# for make new columns
                cols=list(data_.columns)
                cols.insert(0,'gene')
                cols.append('muta_count')
            

                for_col+=1
            sum_=data_.sum(axis=0)

   

            div=sum_.sum(axis=0)
            if div>0:
                sum_=sum_/div
            else:
                sum_=sum_/1


            if sum_.sum(axis=0)<0.9 or sum_.sum(axis=0)>1.1 :
                esang+=1
            

            
            sum_=list(sum_)
            sum_.insert(0,k[0])

            filter_k=k[1][(k[1]['stop_gained']>=1) | (k[1]['missense_variant']>=1) | (k[1]['frameshift_variant']>=1) | (k[1]['splice_region_variant']>=1)]                                          
            sum_.append(filter_k.shape[0])

            last_data.append(sum_)
            per=str((iter/total_gene)*100)
            if (iter/total_gene)*100 != 100:
                if  per.index('.')==1:

                    print(per[:3] + '% mutation feature processed..',end='\r',flush=True)
                else:
                    print(per[:4] + '% mutation feature processed..',end='\r',flush=True)
        
            else:
                print(per[:5] + '% mutation feature processed..')

            iter+=1
        last_data=pd.DataFrame(last_data)
        last_data.columns=cols
        deg_lists=[]
        for deg_gene in list(last_data.loc[:,'gene']):
            if deg_gene in list(deg__.index):
                deg_lists.append(deg__.loc[deg_gene,])
            else:
                deg_lists.append(np.nan)
        last_data['deg']=deg_lists
        #last_data['deg']=list(deg__.loc[list(last_data.loc[:,'gene']),])
        last_data.dropna(subset=['deg'],inplace=True)
        print(last_data.isna().sum())
        
        last_data_protein=last_data
        last_data_protein.index=list(range(last_data_protein.shape[0]))

            



        dis_pathway={}
        dis_pathway['BRCA']=['hsa05224','hsa04151','hsa04310','hsa04330','hsa04915','hsa04115','hsa03440', 'hsa04151','hsa04110', 'hsa04010']
        dis_pathway['PRAD']=['hsa05215','hsa00140','hsa04010','hsa04060','hsa04110','hsa04115','hsa04151','hsa04210','hsa05202']
        dis_pathway['BLCA']=['hsa05219','hsa04010','hsa04012','hsa04110','hsa04115','hsa04370','hsa04520']
        dis_pathway['PAAD']=['hsa05212','hsa04010','hsa04012','hsa04110','hsa04115','hsa04151','hsa04210','hsa04350','hsa04370','hsa04630']
        dis_pathway['THCA']=['hsa05216','hsa03320','hsa04010','hsa04115','hsa04310','hsa04520']
        #dis_pathway['LAML']=['hsa05221','hsa04010','hsa04110','hsa04150','hsa04151','hsa04210','hsa04630','hsa04640'] 
        dis_pathway['SKCM']=['hsa05218','hsa04010','hsa04110','hsa04115','hsa04151','hsa04520','hsa04916']
        dis_pathway['UCEC']=['hsa05213','hsa04010','hsa04012','hsa04060','hsa04110','hsa04115','hsa04151','hsa04310','hsa04520']
        dis_pathway['LAML']=['hsa05221','hsa04010','hsa04110','hsa04150','hsa04151','hsa04210','hsa04630','hsa04640']
        dis_pathway['COAD']=['hsa05210','hsa04010','hsa04012','hsa04110','hsa04115','hsa04150','hsa04151','hsa04210','hsa04310','hsa04350']
        dis_pathway['LGG']=['hsa05214','hsa04010','hsa04012','hsa04020','hsa04060','hsa04110','hsa04115','hsa04150']
        dis_pathway['GBM']=['hsa05214','hsa04010','hsa04012','hsa04020','hsa04060','hsa04110','hsa04115','hsa04150']
        dis_pathway['UCEC']=['hsa05213','hsa04010','hsa04012','hsa04060','hsa04110','hsa04115','hsa04151','hsa04310','hsa04520']
        dis_pathway['SARC']=['hsa05200','hsa03320','hsa04010','hsa04020','hsa04024','hsa04060','hsa04066','hsa04110','hsa04115','hsa04150','hsa04151','hsa04210','hsa04310','hsa04330','hsa04340','hsa04350','hsa04370','hsa04510','hsa04512','hsa04520','hsa04630','hsa04915']
        dis_pathway['READ']=['hsa05210','has04010','hsa04012','hsa04110','hsa04115','hsa04150','hsa04151','hsa04210','hsa04310','hsa04350']
        dis_pathway['LIHC']=['hsa05225','hsa04010','hsa04020','hsa04110','hsa04115','hsa04151','hsa04310','hsa04350','hsa04932','hsa04936','hsa05160','hsa05161']
        dis_pathway['KIRC']=['hsa05211','hsa00020','hsa04010','hsa04066','hsa04120','hsa04350','hsa04370']
        dis_pathway['KIRP']=['hsa05211','hsa00020','hsa04010','hsa04066','hsa04120','hsa04350','hsa04370']
        dis_pathway['LUSC']=['hsa05223','hsa04010','hsa04012','hsa04014','hsa04020','hsa04110','hsa04115','hsa04151']
        dis_pathway['LUAD']=['hsa05223','hsa04010','hsa04012','hsa04014','hsa04020','hsa04110','hsa04115','hsa04151']
        dis_pathway['HNSC']=['hsa05200','hsa03320','hsa04010','hsa04020','hsa04024','hsa04060','hsa04066','hsa04110','hsa04115','hsa04150','hsa04151','hsa04210','hsa04310','hsa04330','hsa04340','hsa04350','hsa04370','hsa04510','hsa04512','hsa04520','hsa04630','hsa04915']
        dis_pathway_list=[]
        for dis in list(dis_pathway.keys()):
            dis_pathway_list.extend(dis_pathway[dis])
        dis_pathway_list=list(set(dis_pathway_list))




        pathway=pd.read_csv('./for_ref/pathway_ensembl.csv')
        pathway2=pathway.drop('Unnamed: 0',axis=1)
        pathways=pathway2.loc[:,'pathway'].drop_duplicates()
        pathways.index=list(range(pathways.shape[0]))
        pathway_dict={}
        pathway_iter=1
        for jj in range(pathways.shape[0]):
            if pathways.loc[jj] in dis_pathway_list:
                gene_list=[pathway2.loc[i,'gene_id'] for i in range(pathway2.shape[0]) if pathway2.loc[i,'pathway']==pathways.loc[jj]]
                pathway_dict[pathways.loc[jj]]=gene_list
            per=str((pathway_iter/pathways.shape[0])*100)
            if (pathway_iter/pathways.shape[0])*100 != 100:
                if  per.index('.')==1:

                    print(per[:3] + '% pathway list make',end='\r',flush=True)
                else:
                    print(per[:4] + '% pathway list make',end='\r',flush=True)
        
            else:
                print(per[:5] + '% pathway list make')

            pathway_iter+=1


        add_col=[]
        add_col_dir_=[]
        pathway_iter=1
        for idx in range(last_data_protein.shape[0]):
            count=0
            count_dir=0
            important_pathway=0
            pathway_idx=0
            for pathway in dis_pathway[dis_name]:
                if pathway in list(pathway_dict.keys()):
                    if last_data_protein.loc[idx,'gene'] in pathway_dict[pathway]:
                        if pathway_idx==0:
                            count_dir=1
                        else:
                            count+=1

                    pathway_idx+=1
                        
            add_col_dir_.append(count_dir)
            add_col.append(count)
            per=str((pathway_iter/last_data_protein.shape[0])*100)
            if (pathway_iter/last_data_protein.shape[0])*100 != 100:
                if  per.index('.')==1:

                    print(per[:3] + '% pathway feature make',end='\r',flush=True)
                else:
                    print(per[:4] + '% pathway feature make',end='\r',flush=True)
        
            else:
                print(per[:5] + '% pathway feature make')

            pathway_iter+=1

        div_fact=len(dis_pathway[dis_name])-1
        last_data_protein['related_pathway']=np.array(add_col)
        last_data_protein['dir_pathway']=add_col_dir_







                

        train_data=last_data_protein
        splice=np.array(train_data.loc[:,'splice_region_variant'])
        inframe=np.array(train_data.loc[:,'inframe_insertion'])+np.array(train_data.loc[:,'conservative_inframe_deletion'])+np.array(train_data.loc[:,'inframe_deletion'])
        lost=np.array(train_data.loc[:,'start_lost'])+np.array(train_data.loc[:,'stop_lost'])
        col=['gene','synonymous_variant','stop_gained','missense_variant','frameshift_variant']
        
        miss_ratio=[]
        for i in range(train_data.shape[0]):
            if train_data.loc[i,'synonymous_variant']+train_data.loc[i,'missense_variant'] != 0:
                miss_ratio.append(train_data.loc[i,'missense_variant']/(train_data.loc[i,'synonymous_variant']+train_data.loc[i,'missense_variant']))
            else:
                miss_ratio.append(0)

        
        new_train=train_data.loc[:,col]
        new_train['splice']=splice
        new_train['inframe']=inframe
        new_train['lost_stop and start']=lost
        new_train['deg']=train_data.loc[:,'deg']
        new_train['related_pathway']=train_data.loc[:,'related_pathway']
        new_train['dir_pathway']=train_data.loc[:,'dir_pathway']
        new_train['muta_count']=train_data.loc[:,'muta_count']
        new_train['miss_ratio']=miss_ratio

        
        col_l_2=list(new_train.columns)
        idxss=0
        new_train['gene_ens']=new_train['gene']
        for col in col_l_2:
            if col=='gene':
                col_l_2[idxss]='gene_ens'
        gene_syms=[]
        for gene in list(new_train.loc[:,'gene']):
            gene_syms.append(ens_sym[gene])
        new_train['gene_symbol']=gene_syms
        col_l_2.insert(1,'gene_symbol')
        new_train=new_train.loc[:,col_l_2]

        print(new_train)


        grap_data=pd.read_csv('./for_ref/humannetv3_ens_ppi.csv')
        grap_data=cut_threshold(grap_data,3.211)


        grap1_=list(grap_data.loc[:,'entrez_id_1'])
        grap2_=list(grap_data.loc[:,'entrez_id_2'])
        grap1_.extend(grap2_)
        grap1_=list(set(grap1_))
        print(len(grap1_))

        ad_node=make_inter_dict(grap_data)

        last_data_protein=new_train






        
        
        


        last_data_protein=last_data_protein.set_index('gene_ens')
        cancer_type_muta=list(last_data_protein.columns)
        cancer_type_muta=['stop_gained','splice','frameshift_variant','missense_variant']
        print(cancer_type_muta)
        graph_feat=[]
        max_=last_data_protein.loc[:,'muta_count']
        max_ = int(max_.quantile(.75))
        for_com_per=1
        under_t=len(list(last_data_protein.index))
        for g_name in list(last_data_protein.index):
            if last_data_protein.loc[g_name,cancer_type_muta].sum(axis=0)>0 and g_name in list(ad_node.keys()):
                count=0
                for adj_node in ad_node[g_name]:
                    if adj_node in list(last_data_protein.index):
    
                        if last_data_protein.loc[adj_node,'muta_count']>max_ :
                            count+=1
                graph_feat.append(count)
            else:
                graph_feat.append(0)

            per=str((for_com_per/under_t)*100)
            if (for_com_per/under_t)*100 != 100:
                if  per.index('.')==1:

                    print(per[:3] + '% PPI feature make',end='\r',flush=True)
                else:
                    print(per[:4] + '% PPI feature make',end='\r',flush=True)
        
            else:
                print(per[:5] + '% PPI feature make')
            for_com_per+=1
        last_data_protein['PPI']=graph_feat

        path_std=last_data_protein.loc[:,'related_pathway'].std()
        path_mean=last_data_protein.loc[:,'related_pathway'].mean()
        path_up=last_data_protein.loc[:,'related_pathway']-path_mean
        last_data_protein['related_pathway']=list(path_up/path_std)

        muta_std=last_data_protein.loc[:,'muta_count'].std()
        muta_mean=last_data_protein.loc[:,'muta_count'].mean()
        muta_up=last_data_protein.loc[:,'muta_count']-muta_mean
        last_data_protein['muta_count']=list(muta_up/muta_std)
        

        ppi_std=last_data_protein.loc[:,'PPI'].std()
        ppi_mean=last_data_protein.loc[:,'PPI'].mean()
        ppi_up=last_data_protein.loc[:,'PPI']-ppi_mean
        last_data_protein['PPI']=list(ppi_up/ppi_std)
        print(last_data_protein.shape)

        last_data_protein.to_csv(inputs['save_dir']+dis_name+'_input_data.csv')
        print('data_save')

        
    def gene_to_ensembl(self,data,re_dict):
        print('start gene ensembl mapping')
        data.index=list(range(data.shape[0]))
        ensembl=[]

        for i  in range(data.shape[0]):
            try:
                #print(self.data.gene_ids_of_gene_name(data.loc[i,'gene'])[0])
                ensembl.append(self.data.gene_ids_of_gene_name(data.loc[i,'gene_symbol'])[0])
            except ValueError:
                if data.loc[i,'gene_symbol'] in list(re_dict.keys()):
                    ensembl.append(re_dict[data.loc[i,'gene_symbol']])
                else:
                    ensembl.append(np.nan)
            if ((i+1)/data.shape[0])*100 != 100:
                per=str(((i+1)/data.shape[0])*100)
  
                if  per.index('.')==1:
                    print(per[:3] + '% gene ensembl mapping',end='\r',flush=True)
                else:
                    print(per[:4] + '% gene ensembl mapping',end='\r',flush=True)


            else:
                print(str(((i+1)/data.shape[0])*100)[:5] + '% gene ensembl mapping')

          
        data['gene_ens']=ensembl
        print('gene ensembl mapping end')

        return data

    
    def make_last_data(self,concat_data,ref_data2):
        print('count mutation type')
    
        data_ref2=pd.read_csv(ref_data2,sep='\t',low_memory=False)
        so_varient=data_ref2.loc[:,'SO term']
        so_varient=[re.sub(" ","_",i) for i in so_varient] 
        so_varient=list(set(so_varient))
        so_varient.extend(['non_coding_transcript_variant'])



        last_dict=[]
        i=0
        groups__=concat_data.groupby(['Sample_ID','gene_ens'])
        for key ,gene_group in groups__:
            gene_var_feat=list(np.zeros((len(so_varient))))     
            var_eff=gene_group.loc[:,'effect']         
            for eff in var_eff:
                idxxx=[idx for idx in range(len(so_varient)) if eff.split(',')[1] in so_varient[idx]]
                gene_var_feat[idxxx[0]]+=1
            last_result=[key[0],key[1],gene_group.loc[list(gene_group.index)[0],'gene_symbol']]
            last_result.extend(gene_var_feat)
            last_dict.append(last_result)
            last_result=[]
            if ((i+1)/groups__.ngroups)*100 != 100:
                per=str(((i+1)/groups__.ngroups)*100)
                if  per.index('.')==1:
                    print(per[:3] + '% variant count',end='\r',flush=True)
                else:
                    print(per[:4] + '% variant count',end='\r',flush=True)
            else:
                print(str(((i+1)/groups__.ngroups)*100)[:5] + '% variant count')
            i+=1

        feature=['Sample_id','gene_ens','gene_symbol']
        feature.extend(so_varient)
        last_dict=np.array(last_dict)
        return_data=pd.DataFrame(last_dict,columns=feature)    
        print('count mutation type end')
        return return_data
    def remake_eff(self,data):
        idx=[]
        removes=[]
        data.dropna(subset=['effect'],inplace=True)
        data.index=list(range(data.shape[0]))
        for i in range(data.shape[0]):
            if ';' in data.loc[i,'effect']:
                idx.append(i)
                h=0
                for eff in str(data.loc[i,'effect']).split(';'):
                    df=data.loc[i,:].copy()
                    df.loc['effect']=eff
                    removes.append(df)
            if ((i+1)/data.shape[0])*100 != 100:
                per=str(((i+1)/data.shape[0])*100)
                if  per.index('.')==1:
                    print(per[:3] + '% gene data reconstruction',end='\r',flush=True)
                else:
                    print(per[:4] + '% gene data reconstruction',end='\r',flush=True)
            else:
                print(str(((i+1)/data.shape[0])*100)[:5] + '% gene data reconstruction')
                        
    
        return     pd.concat([data.drop(idx,axis=0),pd.concat(removes,axis=1).transpose()])

    def make_dict(self,ref_data):

        with open(ref_data, 'rb') as f:
            re=pickle.load(f)
 
        return re

    
    
    def make_muta_count_data(self,disease_name,data_exp_file,data_muta_file,ref_data2='C:/Users/user/Desktop/Park/project/few_shot_project/data/for_ref_data/soterm_list.txt',ref_data="C:/Users/user/Desktop/Park/project/few_shot_project/data/for_ref_data/prev_gene_to_new.txt"):


        data_muta=pd.read_table(data_muta_file,header=5)     

   
        col_n=['Tumor_Sample_Barcode','Hugo_Symbol','Chromosome','Start_Position','End_Position','Reference_Allele','Tumor_Seq_Allele2','HGVSp','all_effects','FILTER']
        data_muta=data_muta.loc[:,col_n]
        col_change={'Tumor_Sample_Barcode':'Sample_ID','Hugo_Symbol':'gene_symbol','Chromosome':'chrom','Start_Position':'start','End_Position':'end','Reference_Allele':'ref','Tumor_Seq_Allele2':'alt','HGVSp':'Amino_Acid_Change','all_effects':'effect','FILTER':'filter','':'dna_vaf'}
        new_col=[]
        for i in list(data_muta.columns):
            new_col.append(col_change[i])
        data_muta.columns=new_col
        new_sample_id=[]
        for s_id in list(data_muta.loc[:,'Sample_ID']):
            s_list=s_id.split('-')
            new_sample=''
            for iiiii in range(4):
                if iiiii==0:
                    new_sample+=s_list[iiiii]
                else:
                    new_sample=new_sample+'-'+s_list[iiiii]
            new_sample_id.append(new_sample)
        data_muta.loc[:,'Sample_ID']=new_sample_id
        gene_enbm=self.make_dict(ref_data)

        data_muta=self.gene_to_ensembl(data_muta, gene_enbm)

        data_muta=self.remake_eff(data_muta)
        data_muta.index=list(range(data_muta.shape[0]))



        data_muta_last_2=self.make_last_data(data_muta,ref_data2)
        try:
            os.mkdir('./data_preprocessed')
        except:
            print('folder already_exist')
        print('end step 1')
        base_dir_out='./data_preprocessed/'
        data_muta_last_2.to_csv(base_dir_out+disease_name+"_step_1.csv",index=False)
        step_data_dir=base_dir_out+disease_name+"_step_1.csv"
 
        print('start data_preprocessor_step2')















        #os.system('python data_preprocessor_step2.py -cancer_name '+disease_name+' -step_1_data_dir '+step_data_dir+' -out_dir '+base_dir_out+' -exp_data_dir '+data_exp_file)
        self.step_2({'dis_name':disease_name,'step_1_data':step_data_dir,'save_dir':base_dir_out,'exp_dir':data_exp_file})
    def make_every_data(self,disease_name,data_muta_dir,data_exp_dir,ref_data11,ref_data22):
        all_combine=[[data_exp_dir, data_muta_dir]]
        if __name__=='__main__':
            procs=[]
            for i in range(len(all_combine)):
                proc=Process(target=self.make_muta_count_data,args=(disease_name,all_combine[i][0],all_combine[i][1],ref_data22,ref_data11))
                proc.start()
                procs.append(proc)
            for proc in procs:
                proc.join()
def make_data(inputs):
    
    ref_data11='./for_ref/gene_ensembl_mapping.pickle'
    ref_data22='./for_ref/soterm_list.txt'
        
    make_input().make_every_data(inputs['dis_name'], inputs['muta_dir'], inputs['exp_dir'], ref_data11, ref_data22)



if __name__ == "__main__" :
    inputs = arg_parse()
    make_data(inputs)
    
    
