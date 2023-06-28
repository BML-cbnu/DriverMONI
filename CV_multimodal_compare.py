import pandas as pd
import numpy as np
import dgl 
from tqdm import tqdm
from torch.utils import data
import networkx as nx
import torch
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import nn
import os
import pickle
import argparse
import torch_models_linear
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cancer_name', '--dis_name', help="directory for ", required=True)

    args = vars(parser.parse_args())
    return args

class JSD(nn.Module):
    
    def __init__(self):
        super(JSD, self).__init__()
    
    def forward(self, net_1_logits, net_2_logits):
        net_1_probs =  F.softmax(net_1_logits, dim=1)
        net_2_probs=  F.softmax(net_2_logits, dim=1)
           
 
        m = 0.5 * (net_1_probs + net_2_probs)

        loss = 0.0
 

        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), m, reduction="batchmean") 
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), m, reduction="batchmean") 
     
        return (0.5 * loss)
class KLD_contrastive(nn.Module):
    def forward(self, inputs, targets,label):
        
        #inputs = F.log_softmax(inputs)
        #targets = F.softmax(targets)
        for i in range(targets.shape[0]):
            if i==0:
                #kl=F.kl_div(inputs[i], targets[i])
                kl=JSD()(torch.reshape(inputs,(1,-1)),torch.reshape(targets,(1,-1)))

                torch.reshape(kl,(1,-1))
                kl=torch.reshape(kl,(1,-1))
            else:
                kl2=F.kl_div(inputs[i], targets[i])
                kl2=JSD()(torch.reshape(inputs,(1,-1)),torch.reshape(targets,(1,-1)))

                kl2=torch.reshape(kl2,(1,-1))

                kl=torch.concat([kl,kl2],axis=0)
        kl=torch.reshape(kl,(-1,))
        return_v=torch.mean(label * kl+(1-label)*(2-kl))

        return return_v
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        
        
        # Expression 3
        # F-Dimension의 피쳐 스페이스가 single fc-layer 지나며 F'-Dimension으로 임베딩 
        self.fc = nn.Linear(in_dim, out_dim, bias=False)


        # i노드의 F' + j노드의 F' 길이의 벡터를 합쳐서 Attention Coefficient를 리턴 	
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

        
    # Expression 3에서 어텐션으로 넘어온 값을 Leaky Relu 적용하는 Layer
	# src는 source vertex, dst는 destination vertex의 약자	
    def edge_attention(self, edges):

        output = torch.concat([edges.src['z'], edges.dst['z']],dim=1)
        #output = cos(edges.src['z'], edges.dst['z'])
        output=self.attn_fc(output)

        #z2 = torch.dot(edges.src['z'], edges.dst['z'])
        #a = self.attn_fc(z2)
        return {'e': F.leaky_relu(output)}

    
    # dgl에서는 모든 노드에 함수를 병렬 적용 할 수 있는 update_all 이라는 api를 제공한다.
    # 해당 api 사용을 위해 텐서를 흘려보내는 역할을 한다고 한다.
	# 구체적인 update_all의 알고리즘은 잘 모르겠으니 그냥 input 함수라고 생각하자.
    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}


    # update_all에서는 흘려보내진 텐서를 각 노드의 mailbox라는 오브젝트에 저장하나 보다.
    # 각 노드에는 여러 이웃이 있으니 mailbox에는 여러개의 attention coefficient가 있다.
    # Expression 4에서 softmax 계수를 가중하여 element wise하게 합한다.  
    def reduce_func(self, nodes):
        alpha = F.sigmoid(nodes.mailbox['e'])
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    
    # (1) fc layer를 통해 피쳐를 임베딩
    # (2) 그레프에 임베딩 된 벡터를 저장
    # (3) apply_edges api를 모든 엣지에 적용하여 i - j 간의 attention coefficeint를 계산
    # (4) 그래프에 저장된 z와e를 텐서로 reduce_func에 전달하여 새로운 h' 를 얻는다.
    def forward(self, h,g):
        self.g = g

        
        self.g.ndata['z'] = self.fc(h)
        self.g.apply_edges(self.edge_attention)

        self.g.update_all(self.message_func, self.reduce_func)

        return self.g.ndata.pop('h')
class GAT_L(nn.Module):
    def __init__(self):
        super(GAT_L, self).__init__()
        
        self.GCN=GATLayer(13,64)


        self.bn= nn.BatchNorm1d(64)


        self.relu= nn.ReLU()
        #self.ll=nn.Linear(64, 256)
        self.ll=nn.Linear(64, 32)

        self.ll2=nn.Linear(256, 128)
        self.ll3=nn.Linear(128, 64)
        self.ll4=nn.Linear(64, 32)
        torch.nn.init.xavier_uniform_(self.ll.weight)
        torch.nn.init.xavier_uniform_(self.ll2.weight)
        torch.nn.init.xavier_uniform_(self.ll3.weight)
        torch.nn.init.xavier_uniform_(self.ll4.weight)


        #self.bn1= nn.BatchNorm1d(256)
        self.bn1= nn.BatchNorm1d(32)

        self.bn2= nn.BatchNorm1d(128)
        self.bn3= nn.BatchNorm1d(64)
        self.bn4=nn.BatchNorm1d(32)
        self.relu= nn.ReLU()



    def forward(self, input2, graph):
        graph.ndata['feat'] = self.GCN(input2,graph)
        #graph.ndata['feat'] = self.GCN2(graph.ndata['feat'].type(torch.FloatTensor).cuda(),graph)

        drug1_id = (graph.ndata['id'] == 1).nonzero().squeeze(1)
        drug1_emb = graph.ndata['feat'][drug1_id[:]]
        #drug1_emb =input2[drug1_id[:]]
        drug1_emb=self.bn(drug1_emb)
        drug1_emb = self.relu(drug1_emb)
        drug1_emb=self.ll(drug1_emb)
        drug1_emb=self.bn1(drug1_emb)

        drug1_emb=self.relu(drug1_emb)

        """

        drug1_emb=self.ll2(drug1_emb)

        drug1_emb=self.bn2(drug1_emb)

        drug1_emb=self.relu(drug1_emb)


        drug1_emb=self.ll3(drug1_emb)

        drug1_emb=self.bn3(drug1_emb)

        drug1_emb=self.relu(drug1_emb)

        drug1_emb=self.ll4(drug1_emb)

        drug1_emb=self.bn4(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        """
        return drug1_emb
    
class SiameseNetwork_GCN_train(nn.Module):
    def __init__(self):
        super(SiameseNetwork_GCN_train, self).__init__()
        

        self.GCN=GAT_L()

        
    def forward_once(self, graph):
        
        drug1_emb=self.GCN(graph.ndata['feat'].type(torch.FloatTensor).cuda(),graph)
        return drug1_emb
    def return_GCN(self):
        return self.GCN

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
"""
class feat_L(nn.Module):
    def __init__(self):
        super(feat_L, self).__init__()
        




        self.relu= nn.ReLU()
        self.ll=nn.Linear(13, 256)
        self.ll2=nn.Linear(256, 128)
        self.ll3=nn.Linear(128, 64)

        self.bn1= nn.BatchNorm1d(256)
        self.bn2= nn.BatchNorm1d(128)
        self.bn3= nn.BatchNorm1d(64)
        self.relu= nn.ReLU()
        self.dr1=nn.Dropout(p=0.7)
        self.dr2=nn.Dropout(p=0.6)

        torch.nn.init.xavier_uniform_(self.ll.weight)
        torch.nn.init.xavier_uniform_(self.ll2.weight)
        torch.nn.init.xavier_uniform_(self.ll3.weight)



    def forward(self, input2):
        


        drug1_emb=self.ll(input2)
        drug1_emb=self.bn1(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr1(drug1_emb)

        drug1_emb=self.ll2(drug1_emb)

        drug1_emb=self.bn2(drug1_emb)

        drug1_emb=self.relu(drug1_emb)
        drug1_emb=self.dr2(drug1_emb)


        drug1_emb=self.ll3(drug1_emb)

        drug1_emb=self.bn3(drug1_emb)

        drug1_emb=self.relu(drug1_emb)

        return drug1_emb
"""

class gcn_linc_embedder(nn.Module):
    def __init__(self,dis_name):
        super(gcn_linc_embedder, self).__init__()
        
        self.gcn_encoder=GAT_L()
        if dis_name=='BRCA':
            self.feat_enc=torch_models_linear.feat_L_BRCA()
        elif dis_name=='READ':
            self.feat_enc=torch_models_linear.feat_L_READ()

        self.l_classify=nn.Linear(96, 2)
        self.soft=nn.Softmax(dim=1)
        torch.nn.init.xavier_uniform_(self.l_classify.weight)



    def forward(self, feat, graph):
        feat=self.feat_enc(feat)
        embedd=self.gcn_encoder(graph.ndata['feat'].type(torch.FloatTensor).cuda(),graph)
        embedd=torch.cat([feat,embedd],axis=1)
        embedd=self.l_classify(embedd)


        return embedd
    
class gcn_linc_embedder_transfer(nn.Module):
    def __init__(self,gat_m,dis_name):
        super(gcn_linc_embedder_transfer, self).__init__()
        
        self.gcn_encoder=gat_m
        if dis_name=='BRCA':
            self.feat_enc=torch_models_linear.feat_L_BRCA()
        elif dis_name=='READ':
            self.feat_enc=torch_models_linear.feat_L_READ()     
        elif dis_name=='SKCM':   
            self.feat_enc=torch_models_linear.feat_L_SKCM()
        elif dis_name=='LUAD':   
            self.feat_enc=torch_models_linear.feat_L_LUAD()
        elif dis_name=='KIRC':
            self.feat_enc=torch_models_linear.feat_L_KIRC()
        elif dis_name=='PRAD':
            self.feat_enc=torch_models_linear.feat_L_PRAD()

        elif dis_name=='HNSC':
            self.feat_enc=torch_models_linear.feat_L_HNSC()

        elif dis_name=='BLCA':
            self.feat_enc=torch_models_linear.feat_L_BLCA()

        elif dis_name=='COAD':
            self.feat_enc=torch_models_linear.feat_L_COAD()

        elif dis_name=='KIRP':
            self.feat_enc=torch_models_linear.feat_L_KIRP()

        self.l_classify=nn.Linear(96, 2)
        #self.l_classify=nn.Linear(64, 2)

        self.gat_ll=nn.Linear(32, 32)
        self.bn= nn.BatchNorm1d(32)
        self.relu= nn.ReLU()

        torch.nn.init.xavier_uniform_(self.gat_ll.weight)
        #torch.nn.init.xavier_uniform_(self.l_classify.weight)


        self.soft=nn.Softmax(dim=1)



    def forward(self, feat, graph):
        feat=self.feat_enc(feat)
        embedd=self.gcn_encoder(graph.ndata['feat'].type(torch.FloatTensor).cuda(),graph)
        embedd=self.gat_ll(embedd)
        embedd=self.bn(embedd)
        embedd=self.relu(embedd)
        embedd=torch.cat([feat,embedd],axis=1)
        embedd=self.l_classify(embedd)


        return embedd

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc, balanced_accuracy_score
def test(m,g):
    pos_c=0
    pos_c_loss=0
    neg_c=0
    neg_c_loss=0
    m.eval()
    firsttt=1
    g_names=[]
    for g_name,d1, d2,label in tqdm(g):
        g_names.extend(g_name)
        d1=d1.cuda()
        label =  label.cuda()
        output1 = m(d1,d2)
        output1 = F.softmax(output1)
        if firsttt==1:
            firsttt=0
            pred=output1[:,1]
            ll=label[:,1]
        else:
            pred2=output1[:,1]    
            pred=torch.concat([pred,pred2],axis=0)
            ll2=label[:,1]
            ll=torch.concat([ll,ll2],axis=0)
    pred=torch.reshape(pred,(-1,1)).detach().cpu().numpy()        
    ll=torch.reshape(ll,(-1,1)).detach().cpu().numpy()  
    fpr, tpr, thresholds = roc_curve(ll, pred)
    predss=[]
    for p in pred:
        if p>=0.5:
            predss.append(1)
        else:
            predss.append(0)
    bacc=balanced_accuracy_score(np.reshape(ll,(-1,)),predss)
    auc_k = auc(fpr, tpr)
    print('auc: '+str(auc_k))
    print('bacc: '+str(bacc))
    print(len(g_names))
    result_df=pd.DataFrame({'gene':g_names,'pred_score':pred.reshape((-1,))})
    return auc_k, bacc, result_df

class classify_graph_data_encoder_humannet(data.Dataset):

    def __init__(self,string_gr,g_ne,df,df_all):
        'Initialization'
        self.ori_df = df.copy()
        df_all=df_all.drop_duplicates(subset='gene_ens')

        self.gene_feat_=df_all.set_index('gene_ens')
        self.graph = self.ssp_multigraph_to_dgl(string_gr)
        self.g_ne=g_ne
        
        self.feat_exi_g=list(self.gene_feat_.index)


    def ssp_multigraph_to_dgl(self,graph):# graph는 csc matrix의 list
        """
        Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
        """
        
        g_nx = nx.Graph()# 여러 type의 edge를 고려하기 위해 무방향인 Multigraph를 선언
        node_l=[]
        node_l.extend(self.gene_feat_.index)
        node_l.extend(graph.loc[:,'entrez_id_1'])
        node_l.extend(graph.loc[:,'entrez_id_2'])
        nodes_l=list(set(node_l))
        self.all_nodes=nodes_l
        g_nx.add_nodes_from(nodes_l)# 전체 노드 그래프에 추가
        #각 tyep별 edges 추가
        nx_triplets=[]
        n1=[]
        n2=[]
        over_90=[]
        for idx in tqdm(range(len(nodes_l))):
            n1.append(idx)
            n2.append(idx)
            over_90.append(0)

        for idx in tqdm(range(graph.shape[0])):
            # Convert adjacency matrix to tuples for nx0

            node1=graph.loc[idx,'entrez_id_1']
            node2=graph.loc[idx,'entrez_id_2']
            n1.append(nodes_l.index(node1))
            n2.append(nodes_l.index(node2))






        #g_nx.add_edges_from(nx_triplets)

        # make dgl graph
        g_dgl = dgl.DGLGraph()# nx graph를 torch에 넣기 좋은 dgl로 변환 하기 위해 dgl Multigraph 선언
        #dgl.convert.from_networkx()
        g_dgl.from_networkx(g_nx)# edge_attrs에 'type'을 줌으로써 각 edge 별로 type이 구분된 multigraph 생성

        g_dgl.add_edges(n1,n2)


        # add node features
        return g_dgl

    def _prepare_subgraphs(self, nodes):
        new_nodes=[]
        for nn in nodes:
            new_nodes.append(self.all_nodes.index(nn))
        subgraph = dgl.DGLGraph(self.graph.subgraph(new_nodes))
        #subgraph=dgl.transform.add_self_loop(subgraph)
        subgraph.ndata['idx'] = torch.LongTensor(np.array(new_nodes))
        #subgraph.ndata['feat']= torch.from_numpy(np.array(self.gene_feat_.loc[nodes,['synonymous_variant', 'stop_gained', 'missense_variant','frameshift_variant', 'splice', 'inframe', 'lost_stop and start', 'deg','related_pathway', 'dir_pathway', 'muta_count', 'miss_ratio', 'PPI']]))
        subgraph.ndata['feat'] = torch.FloatTensor(np.array(self.gene_feat_.loc[nodes,['synonymous_variant', 'stop_gained', 'missense_variant','frameshift_variant', 'splice', 'inframe', 'lost_stop and start', 'deg','related_pathway', 'dir_pathway', 'muta_count', 'miss_ratio', 'PPI']]).astype(np.float64))
        #subgraph.ndata['feat'] = torch.FloatTensor(np.array(self.gene_feat_.loc[nodes,['synonymous_variant', 'stop_gained', 'missense_variant','frameshift_variant', 'splice', 'inframe', 'lost_stop and start', 'deg','related_pathway', 'dir_pathway', 'muta_count', 'miss_ratio']]).astype(np.float64))

        direct_edge=[]
        for i in range(subgraph.edges()[0].shape[0]):
            if torch.tensor(0) in [subgraph.edges()[0][i],subgraph.edges()[1][i]]:
                direct_edge.append(1)
            else:
                direct_edge.append(0)
        subgraph.edata['direct_con'] = torch.FloatTensor(np.array(direct_edge))
        n_ids = np.zeros(len(nodes))
        n_ids[0] = 1
        subgraph.ndata['id'] = torch.FloatTensor(n_ids) 
        return subgraph
    def __len__(self):
        'Denotes the total number of samples'
        return self.ori_df.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label

        g_name1=self.ori_df.loc[index,'gene_ens']
        
        neighbor1=self.g_ne[g_name1]
        neighbor1=list(set(neighbor1) & set(self.feat_exi_g))
        neighbor1=list(set(neighbor1))

        #print(d_v.shape)
        #print(input_mask_d.shape)
        #print(p_v.shape)
        #print(input_mask_p.shape)
        y = self.gene_feat_.loc[g_name1,'label']
        if y==1:
            y=[0,1]
        else:
            y=[1,0]

        nodes_pos1=[]

        nodes_pos1.append(g_name1)
        if g_name1 in neighbor1:
            neighbor1.remove(g_name1)
        nodes_pos1.extend(neighbor1)
        
        subgraph_pos1 = self._prepare_subgraphs(nodes_pos1)     



        #subgraph_pos = remove_self_loop(subgraph_pos)
        #subgraph_pos = transform(subgraph_pos)
        #return np.array(self.gene_feat_.loc[g_name1,['synonymous_variant', 'stop_gained', 'missense_variant','frameshift_variant', 'splice', 'inframe', 'lost_stop and start', 'deg','related_pathway', 'dir_pathway', 'muta_count', 'miss_ratio', 'PPI']]),subgraph_pos1, y
        return g_name1, np.array(self.gene_feat_.loc[g_name1,['synonymous_variant', 'stop_gained', 'missense_variant','frameshift_variant', 'splice', 'inframe', 'lost_stop and start', 'deg','related_pathway', 'dir_pathway', 'muta_count', 'miss_ratio']]),subgraph_pos1, y


class classify_graph_data_encoder(data.Dataset):

    def __init__(self,string_gr,g_ne,df,df_all):
        'Initialization'
        self.ori_df = df.copy()
        self.distortion=self.ori_df.loc[:,'distortion']
        df_all=df_all.drop_duplicates(subset='gene_symbol')

        self.gene_feat_=df_all.set_index('gene_symbol')
        self.graph = self.ssp_multigraph_to_dgl(string_gr)
        self.g_ne=g_ne
        
        self.feat_exi_g=list(self.gene_feat_.index)


    def ssp_multigraph_to_dgl(self,graph):# graph는 csc matrix의 list
        """
        Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
        """
        
        g_nx = nx.Graph()# 여러 type의 edge를 고려하기 위해 무방향인 Multigraph를 선언
        node_l=[]
        node_l.extend(self.gene_feat_.index)
        node_l.extend(graph.loc[:,'protein1'])
        node_l.extend(graph.loc[:,'protein2'])
        nodes_l=list(set(node_l))
        self.all_nodes=nodes_l
        g_nx.add_nodes_from(nodes_l)# 전체 노드 그래프에 추가
        #각 tyep별 edges 추가
        nx_triplets=[]
        n1=[]
        n2=[]
        over_90=[]
        for idx in tqdm(range(len(nodes_l))):
            n1.append(idx)
            n2.append(idx)
            over_90.append(0)

        for idx in tqdm(range(graph.shape[0])):
            # Convert adjacency matrix to tuples for nx0
            node1=graph.loc[idx,'protein1']
            node2=graph.loc[idx,'protein2']
            n1.append(nodes_l.index(node1))
            n2.append(nodes_l.index(node2))
            if 0.9<=graph.loc[idx,'combined_score']:
                over_90.append(1)
                #nx_triplets.append((node1, node2))
            else:
                over_90.append(0)






        #g_nx.add_edges_from(nx_triplets)

        # make dgl graph
        g_dgl = dgl.DGLGraph()# nx graph를 torch에 넣기 좋은 dgl로 변환 하기 위해 dgl Multigraph 선언
        #dgl.convert.from_networkx()
        g_dgl.from_networkx(g_nx)# edge_attrs에 'type'을 줌으로써 각 edge 별로 type이 구분된 multigraph 생성

        g_dgl.add_edges(n1,n2)

        g_dgl.edata['over_90']=torch.tensor(over_90)

        # add node features
        return g_dgl

    def _prepare_subgraphs(self, nodes):
        new_nodes=[]
        for nn in nodes:
            new_nodes.append(self.all_nodes.index(nn))
        subgraph = dgl.DGLGraph(self.graph.subgraph(new_nodes))
        #subgraph=dgl.transform.add_self_loop(subgraph)
        subgraph.ndata['idx'] = torch.LongTensor(np.array(new_nodes))
        #subgraph.ndata['feat']= torch.from_numpy(np.array(self.gene_feat_.loc[nodes,['synonymous_variant', 'stop_gained', 'missense_variant','frameshift_variant', 'splice', 'inframe', 'lost_stop and start', 'deg','related_pathway', 'dir_pathway', 'muta_count', 'miss_ratio', 'PPI']]))
        subgraph.ndata['feat'] = torch.FloatTensor(np.array(self.gene_feat_.loc[nodes,['synonymous_variant', 'stop_gained', 'missense_variant','frameshift_variant', 'splice', 'inframe', 'lost_stop and start', 'deg','related_pathway', 'dir_pathway', 'muta_count', 'miss_ratio', 'PPI']]).astype(np.float64))
        subgraph.edata['over_90'] = self.graph.edata['over_90'][self.graph.subgraph(new_nodes).parent_eid]
        direct_edge=[]
        for i in range(subgraph.edges()[0].shape[0]):
            if torch.tensor(0) in [subgraph.edges()[0][i],subgraph.edges()[1][i]]:
                direct_edge.append(1)
            else:
                direct_edge.append(0)
        subgraph.edata['direct_con'] = torch.FloatTensor(np.array(direct_edge))
        n_ids = np.zeros(len(nodes))
        n_ids[0] = 1
        subgraph.ndata['id'] = torch.FloatTensor(n_ids) 
        return subgraph
    def __len__(self):
        'Denotes the total number of samples'
        return self.ori_df.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label

        g_name1=self.ori_df.loc[index,'gene_symbol']
        
        neighbor1=self.g_ne[g_name1]
        neighbor1=list(set(neighbor1) & set(self.feat_exi_g))
        neighbor1=list(set(neighbor1))

        #print(d_v.shape)
        #print(input_mask_d.shape)
        #print(p_v.shape)
        #print(input_mask_p.shape)
        y = self.gene_feat_.loc[g_name1,'label']
        if y==1:
            y=[0,1]
        else:
            y=[1,0]

        nodes_pos1=[]

        nodes_pos1.append(g_name1)
        if g_name1 in neighbor1:
            neighbor1.remove(g_name1)
        nodes_pos1.extend(neighbor1)
        
        subgraph_pos1 = self._prepare_subgraphs(nodes_pos1)





        
        if self.ori_df.loc[index,'distortion']==1:
            g1_eids=subgraph_pos1.all_edges('eid')
            over_90=torch.where(subgraph_pos1.edata['over_90']==1)[0].numpy()
            dir_con=torch.where(subgraph_pos1.edata['direct_con']==1)[0].numpy()

            remove=np.random.choice(over_90, over_90.shape[0]//3, replace=False)
            remove2=np.random.choice(dir_con, dir_con.shape[0]//3, replace=False)
            remove=torch.tensor(np.concatenate([remove,remove2],axis=0))
            remove=g1_eids[remove]
            subgraph_pos1.remove_edges(remove)           



        #subgraph_pos = remove_self_loop(subgraph_pos)
        #subgraph_pos = transform(subgraph_pos)
        return np.array(self.gene_feat_.loc[g_name1,['synonymous_variant', 'stop_gained', 'missense_variant','frameshift_variant', 'splice', 'inframe', 'lost_stop and start', 'deg','related_pathway', 'dir_pathway', 'muta_count', 'miss_ratio', 'PPI']]),subgraph_pos1, y

class meta_learning_graph_data_encoder(data.Dataset):

    def __init__(self,triplet_pair,string_gr,g_ne,df):
        'Initialization'
        df=df.drop_duplicates(subset='gene_symbol')
        self.gene_feat_=df.set_index('gene_symbol')
        self.graph = self.ssp_multigraph_to_dgl(string_gr)
        self.g_ne=g_ne
        
        self.feat_exi_g=list(self.gene_feat_.index)
        self.triplet_pair=triplet_pair

    def ssp_multigraph_to_dgl(self,graph):# graph는 csc matrix의 list
        """
        Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
        """
        
        g_nx = nx.Graph()# 여러 type의 edge를 고려하기 위해 무방향인 Multigraph를 선언
        node_l=[]
        node_l.extend(self.gene_feat_.index)
        node_l.extend(graph.loc[:,'protein1'])
        node_l.extend(graph.loc[:,'protein2'])
        nodes_l=list(set(node_l))
        self.all_nodes=nodes_l
        g_nx.add_nodes_from(nodes_l)# 전체 노드 그래프에 추가
        #각 tyep별 edges 추가
        nx_triplets=[]
        n1=[]
        n2=[]
        over_90=[]
        for idx in tqdm(range(len(nodes_l))):
            n1.append(idx)
            n2.append(idx)
            over_90.append(0)

        for idx in tqdm(range(graph.shape[0])):
            # Convert adjacency matrix to tuples for nx0
            node1=graph.loc[idx,'protein1']
            node2=graph.loc[idx,'protein2']
            n1.append(nodes_l.index(node1))
            n2.append(nodes_l.index(node2))

            if 0.9<=graph.loc[idx,'combined_score']:
                over_90.append(1)
                #nx_triplets.append((node1, node2))
            else:
                over_90.append(0)
                #nx_triplets.append((node1, node2))





        #g_nx.add_edges_from(nx_triplets)

        # make dgl graph
        g_dgl = dgl.DGLGraph()# nx graph를 torch에 넣기 좋은 dgl로 변환 하기 위해 dgl Multigraph 선언
        #dgl.convert.from_networkx()
        g_dgl.from_networkx(g_nx)# edge_attrs에 'type'을 줌으로써 각 edge 별로 type이 구분된 multigraph 생성

        g_dgl.add_edges(n1,n2)

        g_dgl.edata['over_90']=torch.tensor(over_90)

        # add node features
        return g_dgl

    def _prepare_subgraphs(self, nodes):
        new_nodes=[]
        for nn in nodes:
            new_nodes.append(self.all_nodes.index(nn))
        subgraph = dgl.DGLGraph(self.graph.subgraph(new_nodes))
        #subgraph=dgl.transform.add_self_loop(subgraph)
        subgraph.ndata['idx'] = torch.LongTensor(np.array(new_nodes))

        subgraph.ndata['feat'] = torch.FloatTensor(np.array(self.gene_feat_.loc[nodes,['synonymous_variant', 'stop_gained', 'missense_variant','frameshift_variant', 'splice', 'inframe', 'lost_stop and start', 'deg','related_pathway', 'dir_pathway', 'muta_count', 'miss_ratio', 'PPI']]).astype(np.float64))
        subgraph.edata['over_90'] = self.graph.edata['over_90'][self.graph.subgraph(new_nodes).parent_eid]
        direct_edge=[]
        for i in range(subgraph.edges()[0].shape[0]):
            if torch.tensor(0) in [subgraph.edges()[0][i],subgraph.edges()[1][i]]:
                direct_edge.append(1)
            else:
                direct_edge.append(0)
        subgraph.edata['direct_con'] = torch.FloatTensor(np.array(direct_edge))
        n_ids = np.zeros(len(nodes))
        n_ids[0] = 1
        subgraph.ndata['id'] = torch.FloatTensor(n_ids) 
        return subgraph
    def __len__(self):
        'Denotes the total number of samples'
        return self.triplet_pair.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label

        g_name1=self.triplet_pair.loc[index,'gene1']
        
        neighbor1=self.g_ne[g_name1]
        neighbor1=list(set(neighbor1) & set(self.feat_exi_g))
        neighbor1=list(set(neighbor1))

        #print(d_v.shape)
        #print(input_mask_d.shape)
        #print(p_v.shape)
        #print(input_mask_p.shape)
        y = self.triplet_pair.loc[index,'label']
        nodes_pos1=[]

        nodes_pos1.append(g_name1)
        if g_name1 in neighbor1:
            neighbor1.remove(g_name1)
        nodes_pos1.extend(neighbor1)
        
        subgraph_pos1 = self._prepare_subgraphs(nodes_pos1)


        g_name2=self.triplet_pair.loc[index,'gene2']
        neighbor2=self.g_ne[g_name2]
        neighbor2=list(set(neighbor2) & set(self.feat_exi_g))
        neighbor2=list(set(neighbor2))


        #print(d_v.shape)
        #print(input_mask_d.shape)
        #print(p_v.shape)
        #print(input_mask_p.shape)
        nodes_pos2=[]

        nodes_pos2.append(g_name2)
        if g_name2 in neighbor2:
            neighbor2.remove(g_name2) 

        nodes_pos2.extend(neighbor2)
        
        subgraph_pos2 = self._prepare_subgraphs(nodes_pos2)
        if self.triplet_pair.loc[index,'gene1_d']==1:
            g1_eids=subgraph_pos1.all_edges('eid')
            over_90=torch.where(subgraph_pos1.edata['over_90']==1)[0].numpy()
            dir_con=torch.where(subgraph_pos1.edata['direct_con']==1)[0].numpy()

            remove=np.random.choice(over_90, over_90.shape[0]//3, replace=False)
            remove2=np.random.choice(dir_con, dir_con.shape[0]//3, replace=False)
            remove=torch.tensor(np.concatenate([remove,remove2],axis=0))

            remove=g1_eids[remove]
            subgraph_pos1.remove_edges(remove)

        if self.triplet_pair.loc[index,'gene2_d']==1:
            g2_eids=subgraph_pos2.all_edges('eid')
            over_90=torch.where(subgraph_pos2.edata['over_90']==1)[0].numpy()
            dir_con=torch.where(subgraph_pos2.edata['direct_con']==1)[0].numpy()

            remove=np.random.choice(over_90, over_90.shape[0]//3, replace=False)
            remove2=np.random.choice(dir_con, dir_con.shape[0]//3, replace=False)
            remove=torch.tensor(np.concatenate([remove,remove2],axis=0))

            remove=g2_eids[remove]

            subgraph_pos2.remove_edges(remove)

        #subgraph_pos = remove_self_loop(subgraph_pos)
        #subgraph_pos = transform(subgraph_pos)
        return subgraph_pos1,subgraph_pos2, y

class meta_learning_graph_data_encoder_humannet(data.Dataset):

    def __init__(self,triplet_pair,string_gr,g_ne,df):
        'Initialization'
        #df=df.drop_duplicates(subset='gene_symbol')
        #self.gene_feat_=df.set_index('gene_symbol')
        df=df.drop_duplicates(subset='gene_ens')
        self.gene_feat_=df.set_index('gene_ens')

        self.graph = self.ssp_multigraph_to_dgl(string_gr)
        self.g_ne=g_ne
        
        self.feat_exi_g=list(self.gene_feat_.index)
        self.triplet_pair=triplet_pair

    def ssp_multigraph_to_dgl(self,graph):# graph는 csc matrix의 list
        """
        Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
        """

        g_nx = nx.Graph()# 여러 type의 edge를 고려하기 위해 무방향인 Multigraph를 선언
        node_l=[]
        node_l.extend(self.gene_feat_.index)
        #node_l.extend(graph.loc[:,'protein1'])
        #node_l.extend(graph.loc[:,'protein2'])
        node_l.extend(graph.loc[:,'entrez_id_1'])
        node_l.extend(graph.loc[:,'entrez_id_2'])
        nodes_l=list(set(node_l))
        self.all_nodes=nodes_l
        g_nx.add_nodes_from(nodes_l)# 전체 노드 그래프에 추가
        #각 tyep별 edges 추가
        nx_triplets=[]
        n1=[]
        n2=[]
        over_90=[]
        for idx in tqdm(range(len(nodes_l))):
            n1.append(idx)
            n2.append(idx)
            over_90.append(0)

        for idx in tqdm(range(graph.shape[0])):
            # Convert adjacency matrix to tuples for nx0
            #node1=graph.loc[idx,'protein1']
            #node2=graph.loc[idx,'protein2']
            node1=graph.loc[idx,'entrez_id_1']
            node2=graph.loc[idx,'entrez_id_2']
            n1.append(nodes_l.index(node1))
            n2.append(nodes_l.index(node2))

            #if 0.9<=graph.loc[idx,'combined_score']:
            if 3.112<=graph.loc[idx,'LLS']:

                over_90.append(1)
                #nx_triplets.append((node1, node2))
            else:
                over_90.append(0)
                #nx_triplets.append((node1, node2))





        #g_nx.add_edges_from(nx_triplets)

        # make dgl graph
        g_dgl = dgl.DGLGraph()# nx graph를 torch에 넣기 좋은 dgl로 변환 하기 위해 dgl Multigraph 선언
        #dgl.convert.from_networkx()
        g_dgl.from_networkx(g_nx)# edge_attrs에 'type'을 줌으로써 각 edge 별로 type이 구분된 multigraph 생성

        g_dgl.add_edges(n1,n2)

        g_dgl.edata['over_90']=torch.tensor(over_90)

        # add node features
        return g_dgl

    def _prepare_subgraphs(self, nodes):
        new_nodes=[]
        for nn in nodes:
            new_nodes.append(self.all_nodes.index(nn))
        subgraph = dgl.DGLGraph(self.graph.subgraph(new_nodes))
        #subgraph=dgl.transform.add_self_loop(subgraph)
        subgraph.ndata['idx'] = torch.LongTensor(np.array(new_nodes))

        subgraph.ndata['feat'] = torch.FloatTensor(np.array(self.gene_feat_.loc[nodes,['synonymous_variant', 'stop_gained', 'missense_variant','frameshift_variant', 'splice', 'inframe', 'lost_stop and start', 'deg','related_pathway', 'dir_pathway', 'muta_count', 'miss_ratio','PPI']]).astype(np.float64))
        subgraph.edata['over_90'] = self.graph.edata['over_90'][self.graph.subgraph(new_nodes).parent_eid]
        direct_edge=[]
        for i in range(subgraph.edges()[0].shape[0]):
            if torch.tensor(0) in [subgraph.edges()[0][i],subgraph.edges()[1][i]]:
                direct_edge.append(1)
            else:
                direct_edge.append(0)
        subgraph.edata['direct_con'] = torch.FloatTensor(np.array(direct_edge))
        n_ids = np.zeros(len(nodes))
        n_ids[0] = 1
        subgraph.ndata['id'] = torch.FloatTensor(n_ids) 
        return subgraph
    def __len__(self):
        'Denotes the total number of samples'
        return self.triplet_pair.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label

        g_name1=self.triplet_pair.loc[index,'gene1']
        
        neighbor1=self.g_ne[g_name1]
        neighbor1=list(set(neighbor1) & set(self.feat_exi_g))
        neighbor1=list(set(neighbor1))

        #print(d_v.shape)
        #print(input_mask_d.shape)
        #print(p_v.shape)
        #print(input_mask_p.shape)
        y = self.triplet_pair.loc[index,'label']
        nodes_pos1=[]

        nodes_pos1.append(g_name1)
        if g_name1 in neighbor1:
            neighbor1.remove(g_name1)
        nodes_pos1.extend(neighbor1)
        
        subgraph_pos1 = self._prepare_subgraphs(nodes_pos1)


        g_name2=self.triplet_pair.loc[index,'gene2']
        neighbor2=self.g_ne[g_name2]
        neighbor2=list(set(neighbor2) & set(self.feat_exi_g))
        neighbor2=list(set(neighbor2))


        #print(d_v.shape)
        #print(input_mask_d.shape)
        #print(p_v.shape)
        #print(input_mask_p.shape)
        nodes_pos2=[]

        nodes_pos2.append(g_name2)
        if g_name2 in neighbor2:
            neighbor2.remove(g_name2) 

        nodes_pos2.extend(neighbor2)
        
        subgraph_pos2 = self._prepare_subgraphs(nodes_pos2)
        if self.triplet_pair.loc[index,'gene1_d']==1:
            g1_eids=subgraph_pos1.all_edges('eid')
            over_90=torch.where(subgraph_pos1.edata['over_90']==1)[0].numpy()
            dir_con=torch.where(subgraph_pos1.edata['direct_con']==1)[0].numpy()

            remove=np.random.choice(over_90, over_90.shape[0]//3, replace=False)
            remove2=np.random.choice(dir_con, dir_con.shape[0]//3, replace=False)
            remove=torch.tensor(np.concatenate([remove,remove2],axis=0))

            remove=g1_eids[remove]
            subgraph_pos1.remove_edges(remove)

        if self.triplet_pair.loc[index,'gene2_d']==1:
            g2_eids=subgraph_pos2.all_edges('eid')
            over_90=torch.where(subgraph_pos2.edata['over_90']==1)[0].numpy()
            dir_con=torch.where(subgraph_pos2.edata['direct_con']==1)[0].numpy()

            remove=np.random.choice(over_90, over_90.shape[0]//3, replace=False)
            remove2=np.random.choice(dir_con, dir_con.shape[0]//3, replace=False)
            remove=torch.tensor(np.concatenate([remove,remove2],axis=0))

            remove=g2_eids[remove]

            subgraph_pos2.remove_edges(remove)

        #subgraph_pos = remove_self_loop(subgraph_pos)
        #subgraph_pos = transform(subgraph_pos)
        return subgraph_pos1,subgraph_pos2, y

def send_graph_to_device(g, device): 
    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device)

    # edges
    labels = g.edge_attr_schemes()
    for l in labels.keys():
        g.edata[l] = g.edata.pop(l).to(device)
    return g

def move_batch_to_device_dgl(g_dgl_pos, device):


    #drug_idx = torch.LongTensor(drug_idx).to(device=device)
    # targets_neg = torch.LongTensor(targets_neg).to(device=device)
    # r_labels_neg = torch.LongTensor(r_labels_neg).to(device=device)

    g_dgl_pos = send_graph_to_device(g_dgl_pos, device)
    # g_dgl_neg = send_graph_to_device(g_dgl_neg, device)

    return g_dgl_pos
def collate_dgl(samples):
    # The input `samples` is a list of pairs
    g_name,feat,g2,label = map(list, zip(*samples))

    #print(graphs_pos, g_labels_pos, r_labels_pos, samples)
    batched_graph_pos2 = dgl.batch(g2)

    batched_graph_pos2=move_batch_to_device_dgl(batched_graph_pos2,torch.device('cuda:0'))

    #print(batched_graph_pos)
    
    # graphs_neg = [item for sublist in graphs_negs for item in sublist]
    # g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
    # r_labels_neg = [item for sublist in r_labels_negs for item in sublist]

    #batched_graph_neg = dgl.batch(graphs_neg)
    return g_name,torch.FloatTensor(feat)   ,batched_graph_pos2,torch.tensor(label)   



def collate_dgl_meta(samples):
    # The input `samples` is a list of pairs
    g1,g2,label = map(list, zip(*samples))

    #print(graphs_pos, g_labels_pos, r_labels_pos, samples)
    batched_graph_pos1 = dgl.batch(g1)
    batched_graph_pos2 = dgl.batch(g2)

    batched_graph_pos1=move_batch_to_device_dgl(batched_graph_pos1,torch.device('cuda:0'))
    batched_graph_pos2=move_batch_to_device_dgl(batched_graph_pos2,torch.device('cuda:0'))

    #print(batched_graph_pos)
    
    # graphs_neg = [item for sublist in graphs_negs for item in sublist]
    # g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
    # r_labels_neg = [item for sublist in r_labels_negs for item in sublist]

    #batched_graph_neg = dgl.batch(graphs_neg)
    return batched_graph_pos1,batched_graph_pos2,torch.tensor(label)   

def test_linear(m,g):
    criterion2 = JSD()
    pos_c=0
    pos_c_loss=0
    neg_c=0
    neg_c_loss=0
    m.eval()
    for d1, d2,label in tqdm(g):
        label =  label.cuda()
        output1,output2 = m(d1,d2)
        if label[0]==1:
            pos=criterion2(output1,output2)
            pos_c_loss+=pos
            pos_c+=1
        else:
            neg=criterion2(output1,output2)
            neg_c_loss+=neg
            neg_c+=1  
    print('pos kl: '+str(pos_c_loss/pos_c))
    print('neg kl: '+str(neg_c_loss/neg_c))
    return neg_c_loss/neg_c-pos_c_loss/pos_c,pos_c_loss/pos_c
def train_data(inputs):

    dis_name=inputs['dis_name']



    string=pd.read_csv('./for_ref/humannetv3_ens_ppi.csv')
    string.index=list(range(string.shape[0]))

    node_feature=pd.read_csv('./sample_preprocessed_file/'+dis_name+'_input_data.csv')
    node_feature.drop_duplicates(subset='gene_symbol',inplace=True)
    node_feature.index=list(range(node_feature.shape[0]))
    if inputs['compare']=='T':
        new_idx=[]
        df=pd.read_csv('./sample_preprocessed_file/'+dis_name+'_input_data.csv')
        filtered_g_list=pd.read_csv('./for_concat_data/after_concat_'+dis_name+'_genes2.csv')
        filtered_g_list=list(filtered_g_list.loc[:,'gene'])
        df.drop_duplicates(subset='gene_ens',inplace=True)
        df.set_index('gene_ens',inplace=True)
        gene_symbol=list(df.loc[filtered_g_list,'gene_symbol'])

        final_gene_node = sorted(list(set(gene_symbol)))
        gene = pd.read_csv("./gene_info_for_GOSemSim.csv")
        gene_list = list(set(gene['Symbol']))
        final_gene_node=list(set(final_gene_node) & set(gene_list))
        for i in range(node_feature.shape[0]):
            if node_feature.loc[i,'gene_symbol'] in final_gene_node:
                new_idx.append(i)
        node_feature=node_feature.loc[new_idx,:]
        node_feature.index=list(range(node_feature.shape[0]))
    driver_gene_list=list(pd.read_csv('./using_Data/'+dis_name+'_label.csv').loc[:,'gene_label'])
    train_labels=[]
    for train_gene in list(node_feature.loc[:,'gene_ens']):
        if train_gene in driver_gene_list:
            train_labels.append(1)
        else:
            train_labels.append(0)
    node_feature['label']=train_labels
    gene_n_file_l=os.listdir('./string/')
    if 'gene_n_humannet_'+dis_name+'.pickle' in gene_n_file_l:
        with open('./string/gene_n_humannet_'+dis_name+'.pickle', 'rb') as fr:
            g_ne = pickle.load(fr)
    else:
        edgess=[]
        for idx in tqdm(range(string.shape[0])):
            edgess.append((string.loc[idx,'entrez_id_1'],string.loc[idx,'entrez_id_2']))
        print(string)
        G = nx.Graph()
        G.add_edges_from(edgess)
        g_ne={}
        node_feature.index=list(range(node_feature.shape[0]))
        for d_n in tqdm(range(node_feature.shape[0])):
            try:
                n_list=list(nx.ego_graph(G,node_feature.loc[d_n,'gene_ens'],1).nodes())
                #n_list.extend(list(nx.ego_graph(G,node_feature.loc[d_n,'gene_symbol'],2).nodes()))
                n_list=list(set(n_list))
                g_ne[node_feature.loc[d_n,'gene_ens']]=n_list
            except:
                g_ne[node_feature.loc[d_n,'gene_ens']]=[]
        with open('./string/gene_n_humannet_'+dis_name+'.pickle','wb') as fw:
            pickle.dump(g_ne, fw)

    skf = StratifiedKFold(n_splits=3,shuffle=True)

    x=np.array(node_feature.loc[:,['synonymous_variant', 'stop_gained', 'missense_variant','frameshift_variant', 'splice', 'inframe', 'lost_stop and start', 'deg','related_pathway', 'dir_pathway', 'muta_count', 'miss_ratio', 'PPI']])
    y=np.array(node_feature.loc[:,'label'])
    max_aucss=[]
    max_baccs=[]

    for iter_cv, (train_index, test_index) in enumerate(skf.split(x, y)):
        train_feat=node_feature.loc[train_index,:]
        train_feat.index=list(range(train_feat.shape[0]))
        test_feat=node_feature.loc[test_index,:]
        test_feat.index=list(range(test_feat.shape[0]))

        X_train_pos=train_feat[train_feat['label']==1]
        X_train_pos.index=list(range(X_train_pos.shape[0]))
        X_train_neg=train_feat[train_feat['label']==0]
        X_train_neg.index=list(range(X_train_neg.shape[0]))

        X_test_pos=test_feat[test_feat['label']==1]
        X_test_pos.index=list(range(X_test_pos.shape[0]))

        X_test_neg=test_feat[test_feat['label']==0]
        X_test_neg.index=list(range(X_test_neg.shape[0]))




        train_enc=classify_graph_data_encoder_humannet(string,g_ne,train_feat,node_feature)
        test_enc=classify_graph_data_encoder_humannet(string,g_ne,test_feat,node_feature)

        pos_num=train_feat[train_feat['label']==1].shape[0]
        neg_num=train_feat[train_feat['label']==0].shape[0]
        total__=pos_num+neg_num
        weight_for_0 = (1 / neg_num) * (total__ / 2.0)
        weight_for_1 = (1 / pos_num) * (total__ / 2.0)

        classw_w=torch.tensor([weight_for_0,weight_for_1],dtype=float).cuda()
        params = {'batch_size': 32,
                    'shuffle': True,
                    'num_workers': 0,
                    'drop_last': False,
                    'collate_fn':collate_dgl}

        training_generator = data.DataLoader(train_enc, **params)

        test_generator = data.DataLoader(test_enc, **params)

        GAT_m=GAT_L().cuda()
        meta_model=gcn_linc_embedder_transfer(GAT_m,dis_name).cuda()
        optimizer = optim.Adam(meta_model.parameters(),lr = 0.001 )
        epochs=10
        #max_auc=0
        max_bacc=0
        max_auc=0
        max_bacc=0
        for ep in range(epochs):
            firsttt=1
            loss_accumulate=0
            count=0
            total=training_generator.__len__()
            for __,feat,graph,label in training_generator:


                feat=feat.cuda()
                output = meta_model(feat,graph)
                #label = label.to(torch.LongTensor)
                label = label.type(torch.LongTensor)   
                label = label[:,1]
                #label = torch.reshape(label,(-1,2))
                #label = label.to(torch.float32)
                label =  label.cuda()

                optimizer.zero_grad()


                loss_contrastive = torch.nn.functional.nll_loss(F.log_softmax(output, dim=1,dtype=float), label,weight =classw_w)# auc 0.94
                #loss_contrastive = nn.CrossEntropyLoss(weight =classw_w)(output,label)
                #loss_contrastive = nn.BCELoss(weight =classw_w)(F.softmax(output, dim=1),label)
                loss_contrastive.backward()
                optimizer.step()
                loss_accumulate += loss_contrastive
                count += 1
                print('percent: '+str(count/total)+' loss: '+str(loss_accumulate/count),end='\r',flush=True)
            with torch.set_grad_enabled(False):
                #train_auc,train_bacc=test(meta_model,training_generator)
                test_auc,test_bacc,pred_df=test(meta_model,test_generator)
                #test_auc_distorted,test_bacc_distorted, pred_df=test(meta_model,test_generator_distorted)

                #if test_auc>max_auc:
                #   max_auc=test_auc
                #    max_bacc=test_bacc
                if test_auc>max_auc:
                    max_auc=test_auc
                    max_bacc=test_bacc
                    pred_df_max=pred_df
                #print('max auc: '+str(max_auc))
                #print('max bacc: '+str(max_bacc))
                print('max distorted auc: '+str(max_auc))
                print('max distorted bacc: '+str(max_bacc))
        max_aucss.append(max_auc)
        max_baccs.append(max_bacc)
        #max_aucss_distorted.append(max_auc_distorted)
        #max_baccs_distorted.append(max_bacc_distorted)
        if iter_cv==0:
            pred_df_max_=pred_df_max
        else:
            pred_df_max_=pd.concat([pred_df_max_,pred_df_max],axis=0)
    #print(max_aucss)
    #print(max_baccs)
    result_df=pd.DataFrame({'fold':['f1','f2','f3'],'auc_ori':max_aucss,'bacc_ori':max_baccs}).T
    #result_df=pd.DataFrame({'fold':['f1','f2','f3'],'distorted_auc':max_aucss_distorted,'distorted_bacc':max_baccs_distorted}).T

    try:
        os.mkdir('./result_'+inputs['dis_name'])
    except:
        print('exisit')
    result_df.to_csv('./result_'+inputs['dis_name']+'/'+inputs['graph_name_']+'cv_not_contrastive_result_for_compare_part_data_distorted_no_network_feat.csv')
    #pred_df_max_.to_csv('./result_'+inputs['dis_name']+'/'+inputs['graph_name_']+'_pred_score_no_network_feat.csv')

if __name__ == "__main__" :
    inputs = arg_parse()
    inputs['compare']='T'
    train_data(inputs)