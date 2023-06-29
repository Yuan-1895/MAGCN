# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:20:23 2018

@author: gk
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d, Linear
import copy
import math
"""
x-> [batch_num,in_channels,num_nodes,tem_size],
"""

class TATT(nn.Module):
    def __init__(self,c_in,num_nodes,tem_size):
        super(TATT,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(num_nodes, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(num_nodes,c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(tem_size,tem_size), requires_grad=True)
        
        self.v=nn.Parameter(torch.rand(tem_size,tem_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)
        
    def forward(self,seq):
        c1 = seq.permute(0,1,3,2)#b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze(1)#b,l,n
        
        c2 = seq.permute(0,2,1,3)#b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze(1)#b,c,l
     
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        logits = torch.matmul(self.v,logits)
        ##normalization
        a,_ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits,-1)
        return coefs
    
class SATT(nn.Module):
    def __init__(self,c_in,num_nodes,tem_size):
        super(SATT,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(tem_size, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(tem_size,c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        
        self.v=nn.Parameter(torch.rand(num_nodes,num_nodes), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)
        
    def forward(self,seq):
        c1 = seq
        f1 = self.conv1(c1).squeeze(1)#b,n,l
        
        c2 = seq.permute(0,3,1,2)#b,c,n,l->b,l,n,c
        f2 = self.conv2(c2).squeeze(1)#b,c,n
     
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        logits = torch.matmul(self.v,logits)
        ##normalization
        a,_ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits,-1)
        return coefs

class cheby_conv_ds(nn.Module):
    def __init__(self,c_in,c_out,K):
        super(cheby_conv_ds,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.K=K
        
    def forward(self,x,adj,ds):
        nSample, feat_in,nNode, length  = x.shape
        Ls = []
        L0 = torch.eye(nNode).cuda()
        L1 = adj
    
        L = ds*adj
        I = ds*torch.eye(nNode).cuda()
        Ls.append(I)
        Ls.append(L)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            L3 =ds*L2
            Ls.append(L3)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out  

    
###ASTGCN_block
class ST_BLOCK_0(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_0,self).__init__()
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.TATT=TATT(c_in,num_nodes,tem_size)
        self.SATT=SATT(c_in,num_nodes,tem_size)
        self.dynamic_gcn=cheby_conv_ds(c_in,c_out,K)
        self.K=K
        
        self.time_conv=Conv2d(c_out, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        #self.bn=BatchNorm2d(c_out)
        self.bn=LayerNorm([c_out,num_nodes,tem_size])
        
    def forward(self,x,supports):
        x_input=self.conv1(x)
        T_coef=self.TATT(x)
        T_coef=T_coef.transpose(-1,-2)
        x_TAt=torch.einsum('bcnl,blq->bcnq',x,T_coef)
        S_coef=self.SATT(x)#B x N x N
        
        spatial_gcn=self.dynamic_gcn(x_TAt,supports,S_coef)
        spatial_gcn=torch.relu(spatial_gcn)
        time_conv_output=self.time_conv(spatial_gcn)
        out=self.bn(torch.relu(time_conv_output+x_input))
        
        return  out,S_coef,T_coef    
     


###1
###DGCN_Mask&&DGCN_Res
class T_cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(T_cheby_conv,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        Lap = Lap.transpose(-1,-2)
        #print(Lap)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 

class ST_BLOCK_1(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_1,self).__init__()
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.TATT_1=TATT_1(c_out,num_nodes,tem_size)
        self.dynamic_gcn=T_cheby_conv(c_out,2*c_out,K,Kt)
        self.K=K
        self.time_conv=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        #self.bn=BatchNorm2d(c_out)
        self.c_out=c_out
        self.bn=LayerNorm([c_out,num_nodes,tem_size])
    def forward(self,x,supports):
        x_input=self.conv1(x)
        x_1=self.time_conv(x)
        x_1=F.leaky_relu(x_1)
        x_1=F.dropout(x_1,0.5,self.training)
        x_1=self.dynamic_gcn(x_1,supports)
        filter,gate=torch.split(x_1,[self.c_out,self.c_out],1)
        x_1=torch.sigmoid(gate)*F.leaky_relu(filter)
        x_1=F.dropout(x_1,0.5,self.training)
        T_coef=self.TATT_1(x_1)
        T_coef=T_coef.transpose(-1,-2)
        x_1=torch.einsum('bcnl,blq->bcnq',x_1,T_coef)
        out=self.bn(F.leaky_relu(x_1)+x_input)
        return out,supports,T_coef
        
    
###2    
##DGCN_R  
class T_cheby_conv_ds(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(T_cheby_conv_ds,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).repeat(nSample,1,1).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 


    
class SATT_2(nn.Module):
    def __init__(self,c_in,num_nodes):
        super(SATT_2,self).__init__()
        self.conv1=Conv2d(c_in, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(c_in, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.bn=LayerNorm([num_nodes,num_nodes,12])
        self.c_in=c_in
    def forward(self,seq):
        shape = seq.shape
        f1 = self.conv1(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,3,1,4,2).contiguous()
        f2 = self.conv2(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,1,3,4,2).contiguous()
        
        logits = torch.einsum('bnclm,bcqlm->bnqlm',f1,f2)
        logits=logits.permute(0,3,1,2,4).contiguous()
        logits = torch.sigmoid(logits)
        logits = torch.mean(logits,-1)
        return logits
  

class TATT_1(nn.Module):
    def __init__(self,c_in,num_nodes,tem_size):
        super(TATT_1,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1), stride=(1,1), bias=False)
        #self.conv1= Conv2d(c_in, 1, (1, 3), padding=(1,1))

        self.conv2=Conv2d(num_nodes, 1, kernel_size=(1, 1), stride=(1,1), bias=False)
        #self.conv2= Conv2d(num_nodes, 1, (1, 3))
        self.w=nn.Parameter(torch.rand(num_nodes,c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(tem_size,tem_size), requires_grad=True)
        
        self.v=nn.Parameter(torch.rand(tem_size,tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn=BatchNorm1d(tem_size)
        
    def forward(self,seq):
        #print(seq.shape)
        c1 = seq.permute(0,1,3,2)#b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()#b,l,n
        #print(f1.shape)
        
        c2 = seq.permute(0,2,1,3)#b,c,n,l->b,n,c,l
        #print(c2.shape)
        f2 = self.conv2(c2).squeeze()#b,c,n
        #print(f2.shape)
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        logits = torch.matmul(self.v,logits)
        logits = logits.permute(0,2,1).contiguous()
        logits=self.bn(logits).permute(0,2,1).contiguous()
        coefs = torch.softmax(logits,-1)
        return coefs 

class TATT_my(nn.Module):
    def __init__(self,c_in,num_nodes,tem_size):
        super(TATT_my,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1), stride=(1,1), bias=False)
        #self.conv1= Conv2d(c_in, 1, (1, 3), padding=(1,1))

        self.conv2=Conv2d(num_nodes, 1, kernel_size=(1, 1), stride=(1,1), bias=False)
        #self.conv2= Conv2d(num_nodes, 1, (1, 3))
        self.w=nn.Parameter(torch.rand(num_nodes,c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(tem_size,tem_size), requires_grad=True)
        
        self.v=nn.Parameter(torch.rand(tem_size,tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn=BatchNorm1d(tem_size)
        
    def forward(self,seq):
        #print(seq.shape)
        c1 = seq.permute(0,1,3,2)#b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()#b,l,n
        #print(f1.shape)
        
        c2 = seq.permute(0,2,1,3)#b,c,n,l->b,n,c,l
        #print(c2.shape)
        f2 = self.conv2(c2).squeeze()#b,c,n
        #print(f2.shape)
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        logits = torch.matmul(self.v,logits)
        logits = logits.permute(0,2,1).contiguous()
        logits=self.bn(logits).permute(0,2,1).contiguous()
        coefs = torch.softmax(logits,-1)
        return coefs   


class ST_BLOCK_2_r(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_2_r,self).__init__()
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.TATT_1=TATT_1(c_out,num_nodes,tem_size)
        
        self.SATT_2=SATT_2(c_out,num_nodes)
        self.dynamic_gcn=T_cheby_conv_ds(c_out,2*c_out,K,Kt)
        self.LSTM=nn.LSTM(num_nodes,num_nodes,batch_first=True)#b*n,l,c
        self.K=K
        self.tem_size=tem_size
        self.time_conv=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.bn=BatchNorm2d(c_out)
        self.c_out=c_out
        #self.bn=LayerNorm([c_out,num_nodes,tem_size])
        
        
    def forward(self,x,supports):
        x_input=self.conv1(x)
        x_1=self.time_conv(x)
        x_1=F.leaky_relu(x_1)
        S_coef=self.SATT_2(x_1)
        shape=S_coef.shape
        h = Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        c=Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        hidden=(h,c)
        S_coef=S_coef.permute(0,2,1,3).contiguous().view(shape[0]*shape[2],shape[1],shape[3])
        S_coef=F.dropout(S_coef,0.5,self.training) #2020/3/28/22:17,试验下效果
        _,hidden=self.LSTM(S_coef,hidden)
        adj_out=hidden[0].squeeze().view(shape[0],shape[2],shape[3]).contiguous()
        adj_out1=(adj_out)*supports
        x_1=F.dropout(x_1,0.5,self.training)
        x_1=self.dynamic_gcn(x_1,adj_out1)
        filter,gate=torch.split(x_1,[self.c_out,self.c_out],1)
        x_1=torch.sigmoid(gate)*F.leaky_relu(filter)
        x_1=F.dropout(x_1,0.5,self.training)
        T_coef=self.TATT_1(x_1)
        T_coef=T_coef.transpose(-1,-2)
        x_1=torch.einsum('bcnl,blq->bcnq',x_1,T_coef)
        out=self.bn(F.leaky_relu(x_1)+x_input)
        return out,adj_out,T_coef

###DGCN_GAT
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features,out_features,length,Kt, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.length = length
        self.alpha = alpha
        self.concat = concat
        
        self.conv0=Conv2d(self.in_features, self.out_features, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        
        self.conv1=Conv1d(self.out_features*self.length, 1, kernel_size=1,
                          stride=1, bias=False)
        self.conv2=Conv1d(self.out_features*self.length, 1, kernel_size=1,
                          stride=1, bias=False)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, input, adj):
        '''
        :param input: 输入特征 (batch,in_features,nodes,length)->(batch,in_features*length,nodes)
        :param adj:  邻接矩阵 (batch,batch)
        :return: 输出特征 (batch,out_features)
        '''
        input=self.conv0(input)
        shape=input.shape
        input1=input.permute(0,1,3,2).contiguous().view(shape[0],-1,shape[2]).contiguous()
        
        f_1=self.conv1(input1)
        f_2=self.conv1(input1)
        
        logits = f_1 + f_2.permute(0,2,1).contiguous()
        attention = F.softmax(self.leakyrelu(logits)+adj, dim=-1)  # (batch,nodes,nodes)
        #attention1 = F.dropout(attention, self.dropout, training=self.training) # (batch,nodes,nodes)
        attention=attention.transpose(-1,-2)
        h_prime = torch.einsum('bcnl,bnq->bcql',input,attention) # (batch,out_features)        
        return h_prime,attention

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads,length,Kt):
        """
        Dense version of GAT.
        :param nfeat: 输入特征的维度
        :param nhid:  输出特征的维度
        :param nclass: 分类个数
        :param dropout: dropout
        :param alpha: LeakyRelu中的参数
        :param nheads: 多头注意力机制的个数
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid,length=length,Kt=Kt, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            
        #self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        fea=[]
        for att in self.attentions:
            f,S_coef=att(x, adj)
            fea.append(f)
        x = torch.cat(fea, dim=1)
        #x = torch.mean(x,-1)
        return x,S_coef



###Gated-STGCN(IJCAI)
class cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(cheby_conv,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 

class ST_BLOCK_4(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_4,self).__init__()
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.gcn=cheby_conv(c_out//2,c_out,K,1)
        self.conv2=Conv2d(c_out, c_out*2, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.c_out=c_out
        self.conv_1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        #self.conv_2=Conv2d(c_out//2, c_out, kernel_size=(1, 1),
          #                stride=(1,1), bias=True)

    def forward(self,x,supports):
        x_input1=self.conv_1(x)
        x1=self.conv1(x)
        filter1,gate1=torch.split(x1,[self.c_out//2,self.c_out//2],1)
        x1=(filter1)*torch.sigmoid(gate1)
        x2=self.gcn(x1,supports)
        x2=torch.relu(x2)
        #x_input2=self.conv_2(x2)
        x3=self.conv2(x2)
        filter2,gate2=torch.split(x3,[self.c_out,self.c_out],1)
        x=(filter2+x_input1)*torch.sigmoid(gate2)
        return x

###GRCN(ICLR)
class gcn_conv_hop(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ] - input of one single time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : gcn_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(gcn_conv_hop,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv1d(c_in_new, c_out, kernel_size=1,
                          stride=1, bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode  = x.shape
        
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcn,knq->bckq', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode)
        out = self.conv1(x)
        return out 



class ST_BLOCK_5(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_5,self).__init__()
        self.gcn_conv=gcn_conv_hop(c_out+c_in,c_out*4,K,1)
        self.c_out=c_out
        self.tem_size=tem_size
        
        
    def forward(self,x,supports):
        shape = x.shape
        h = Variable(torch.zeros((shape[0],self.c_out,shape[2]))).cuda()
        c = Variable(torch.zeros((shape[0],self.c_out,shape[2]))).cuda()
        out=[]
        
        for k in range(self.tem_size):
            input1=x[:,:,:,k]
            tem1=torch.cat((input1,h),1)
            fea1=self.gcn_conv(tem1,supports)
            i,j,f,o = torch.split(fea1, [self.c_out, self.c_out, self.c_out, self.c_out], 1)
            new_c=c*torch.sigmoid(f)+torch.sigmoid(i)*torch.tanh(j)
            new_h=torch.tanh(new_c)*(torch.sigmoid(o))
            c=new_c
            h=new_h
            out.append(new_h)
        x=torch.stack(out,-1)
        return x 

    
###OTSGGCN(ITSM)
class cheby_conv1(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(cheby_conv1,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 

class ST_BLOCK_6(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_6,self).__init__()
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.gcn=cheby_conv(c_out,2*c_out,K,1)
        
        self.c_out=c_out
        self.conv_1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        
    def forward(self,x,supports):
        x_input1=self.conv_1(x)
        x1=self.conv1(x)   
        x2=self.gcn(x1,supports)
        filter,gate=torch.split(x2,[self.c_out,self.c_out],1)
        x=(filter+x_input1)*torch.sigmoid(gate)
        return x    
    
    
##gwnet
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        A=A.transpose(-1,-2)
        #x = torch.einsum('ncvl,ncvw->ncwl',(x,A))  
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class nconv1(nn.Module):
    def __init__(self):
        super(nconv1,self).__init__()

    def forward(self,x, A):
        
        A=A.transpose(-1,-2)
        x = torch.einsum('ncvl,ncvw->ncwl',(x,A))  
        #x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class multi_gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(multi_gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    

##cluster    

    
class nconv_batch(nn.Module):
    def __init__(self):
        super(nconv_batch,self).__init__()

    def forward(self,x, A):
        A=A.transpose(-1,-2)
        #try:
       #     x = torch.einsum('ncvl,vw->ncwl',(x,A))
        #except:
        x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        return x.contiguous()
    
class linear_time(nn.Module):
    def __init__(self,c_in,c_out,Kt):
        super(linear_time,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class multi_gcn_d(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(multi_gcn_d,self).__init__()
        self.nconv = nconv()
        self.nconv1 = nconv1()
        #c_in = (order*support_len+1)*c_in
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):#x 64, 32, 792, 8
        out = [x]
        
        batch_size, in_channels, num_of_vertices, num_of_timesteps = x.shape
    
        x_t = x.permute(0, 3, 2, 1).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        score = torch.matmul(x_t, x_t.transpose(1, 2)) / math.sqrt(in_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)
 
        score = F.dropout(F.softmax(score, dim=-1), self.dropout)  # the sum of each row is 1; (b*t, N, N)
        
        att= score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))
     
        x= x.permute(0, 3, 2, 1)

        if(len(support) ==1):
            support.append(att)
        else:
            support[1]=att

        x1 = self.nconv(x,support[0])

        out.append(x1.permute(0, 3, 2, 1))
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1,support[0])

            out.append(x2.permute(0, 3, 2, 1))
            x1 = x2

        x1 = self.nconv1(x, support[1])

        out.append(x1.permute(0, 3, 2, 1))
        for k in range(2, self.order + 1):
            x2 = self.nconv1(x1, support[1])

            out.append(x2.permute(0, 3, 2, 1))
            x1 = x2
         
        h = torch.cat(out,dim=1)
        #print(h.shape)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h



class multi_gcn_d_new(nn.Module):
    def __init__(self, c_in, c_out, Kt, dropout, support_len=3, order=2):
        super(multi_gcn_d_new, self).__init__()
        self.nconv1 = nconv1()
        self.nconv = nconv()
        # c_in = (order*support_len+1)*c_in
        c_in = (order * support_len + 1) * c_in
        #self.mlp = linear_time(c_in, c_out, Kt)
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):  # x 64, 32, 792, 8
        out = [x] #x:B F N T

        batch_size, in_channels, num_of_vertices, num_of_timesteps = x.shape

        x_t = x.permute(0, 3, 2, 1).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        score = torch.matmul(x_t, x_t.transpose(1, 2)) / math.sqrt(
            in_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)

        score = F.dropout(F.softmax(score, dim=-1), self.dropout)  # the sum of each row is 1; (b*t, N, N)

        att = score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))

        x = x.permute(0, 3, 2, 1) #x B T N F
        for a in support:
            a1=a.mul(att)
            x1 = self.nconv1(x, a1)

            out.append(x1.permute(0, 3, 2, 1))
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)

                out.append(x2.permute(0, 3, 2, 1))
                x1 = x2

        h = torch.cat(out, dim=1)
        # print(h.shape)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class multi_gcn_d_new2(nn.Module):
    def __init__(self, c_in, c_out, Kt, dropout, support_len=3, order=2):
        super(multi_gcn_d_new2, self).__init__()
        self.nconv1 = nconv1()
        self.nconv = nconv()
        # c_in = (order*support_len+1)*c_in
        c_in = (order * support_len + 1) * c_in
        #self.mlp = linear_time(c_in, c_out, Kt)
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):  # x 64, 32, 792, 8
        out = [x] #x:B F N T

        batch_size, in_channels, num_of_vertices, num_of_timesteps = x.shape

        x_t = x.permute(0, 3, 2, 1).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        score = torch.matmul(x_t, x_t.transpose(1, 2)) / math.sqrt(
            in_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)

        score = F.dropout(F.softmax(score, dim=-1), self.dropout)  # the sum of each row is 1; (b*t, N, N)

        att = score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))

        x = x.permute(0, 3, 2, 1) #x B T N F
        print(x.shape)
        print(support[0].shape)
        support[0]=support[0].mul(att)
        x1 = self.nconv1(x, support[0])

        out.append(x1.permute(0, 3, 2, 1))
        for k in range(2, self.order + 1):
            x2 = self.nconv1(x1, support[0])

            out.append(x2.permute(0, 3, 2, 1))
            x1 = x2

        support[1]=support[1].mul(att)
        x1 = self.nconv1(x, support[1])

        out.append(x1.permute(0, 3, 2, 1))
        for k in range(2, self.order + 1):
            x2 = self.nconv1(x1, support[1])

            out.append(x2.permute(0, 3, 2, 1))
            x1 = x2

        x1 = self.nconv(x, support[2])

        out.append(x1.permute(0, 3, 2, 1))
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, support[2])

            out.append(x2.permute(0, 3, 2, 1))
            x1 = x2

        h = torch.cat(out, dim=1)
        # print(h.shape)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class multi_gcn_time(nn.Module):
    def __init__(self,c_in,c_out,Kt,dropout,support_len=3,order=2):
        super(multi_gcn_time,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear_time(c_in,c_out,Kt)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):#x 64, 32, 792, 8
        
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class multi_gcn_time1(nn.Module):
    def __init__(self,c_in,c_out,Kt,dropout,support_len=3,order=2):
        super(multi_gcn_time1,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear_time(96,c_out,Kt)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):#x 64, 32, 792, 8
        out = [x]
        x= x.permute(0, 3, 2, 1)
        for a in support: 
            x1 = self.nconv(x,a)
            out.append(x1.permute(0, 3, 2, 1))
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2.permute(0, 3, 2, 1))
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class SATT_pool(nn.Module):
    def __init__(self,c_in,num_nodes):
        super(SATT_pool,self).__init__()
        self.conv1=Conv2d(c_in, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(c_in, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.c_in=c_in
    def forward(self,seq):
        shape = seq.shape
        f1 = self.conv1(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,3,1,4,2).contiguous()
        f2 = self.conv2(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,1,3,4,2).contiguous()
        
        logits = torch.einsum('bnclm,bcqlm->bnqlm',f1,f2)
        
        logits=logits.permute(0,3,1,2,4).contiguous()
        logits = F.softmax(logits,2)
        logits = torch.mean(logits,-1)
        return logits

class SATT_h_gcn(nn.Module):
    def __init__(self,c_in,tem_size):
        super(SATT_h_gcn,self).__init__()
        self.conv1=Conv2d(c_in, c_in//8, kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(c_in, c_in//8, kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=False)
        self.c_in=c_in
    def forward(self,seq,a):
        shape = seq.shape
        f1 = self.conv1(seq).squeeze().permute(0,2,1).contiguous()
        f2 = self.conv2(seq).squeeze().contiguous()
        
        logits = torch.matmul(f1,f2)
        
        logits=F.softmax(logits,-1)
        
        return logits

class multi_gcn_batch(nn.Module):
    def __init__(self,c_in,c_out,Kt,dropout,support_len=3,order=2):
        super(multi_gcn_batch,self).__init__()
        self.nconv = nconv_batch()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear_time(c_in,c_out,Kt)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:            
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gate(nn.Module):
    def __init__(self,c_in):
        super(gate,self).__init__()
        self.conv1=Conv2d(c_in, c_in//2, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        
        
        
    def forward(self,seq,seq_cluster):
        
        #x=torch.cat((seq_cluster,seq),1)     
        #gate=torch.sigmoid(self.conv1(x)) 
        out=torch.cat((seq,(seq_cluster)),1)    
        
        return out
           
    
class Transmit(nn.Module):
    def __init__(self,c_in,tem_size,transmit,num_nodes,cluster_nodes):
        super(Transmit,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(tem_size, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(tem_size,c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(num_nodes,cluster_nodes), requires_grad=True)
        self.c_in=c_in
        self.transmit=transmit
        
    def forward(self,seq,seq_cluster):
        
        c1 = seq
        f1 = self.conv1(c1).squeeze(1)#b,n,l
        
        c2 = seq_cluster.permute(0,3,1,2)#b,c,n,l->b,l,n,c
        f2 = self.conv2(c2).squeeze(1)#b,c,n
        logits=torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        a = torch.mean(logits, 1, True)
        logits = logits - a
        logits = torch.sigmoid(logits)
        
        coefs = (logits)*self.transmit
        return coefs    

class T_cheby_conv_ds_1(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(T_cheby_conv_ds_1,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, Kt),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).repeat(nSample,1,1).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 
    
class dynamic_adj(nn.Module):
    def __init__(self,c_in,num_nodes):
        super(dynamic_adj,self).__init__()
        
        self.SATT=SATT_pool(c_in,num_nodes)
        self.LSTM=nn.LSTM(num_nodes,num_nodes,batch_first=True)#b*n,l,c
    def forward(self,x):
        S_coef=self.SATT(x)        
        shape=S_coef.shape
        h = Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        c=Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        hidden=(h,c)
        S_coef=S_coef.permute(0,2,1,3).contiguous().view(shape[0]*shape[2],shape[1],shape[3])
        S_coef=F.dropout(S_coef,0.5,self.training) #2020/3/28/22:17,试验下效果
        _,hidden=self.LSTM(S_coef,hidden)
        adj_out=hidden[0].squeeze().view(shape[0],shape[2],shape[3]).contiguous()
        
        return adj_out
    
    
class GCNPool_dynamic(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,
                 Kt,dropout,pool_nodes,support_len=3,order=2):
        super(GCNPool_dynamic,self).__init__()
        self.dropout=dropout
        self.time_conv=Conv2d(c_in, 2*c_out, kernel_size=(1, Kt),padding=(0,0),
                          stride=(1,1), bias=True,dilation=2)
        
        self.multigcn=multi_gcn_time(c_out,2*c_out,Kt,dropout,support_len,order)
        self.multigcn1=multi_gcn_batch(c_out,2*c_out,Kt,dropout,support_len,order)
        self.dynamic_gcn=T_cheby_conv_ds_1(c_out,2*c_out,order+1,Kt)
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        self.TAT=TATT_1(c_out,num_nodes,tem_size)
        self.c_out=c_out
        #self.gate=gate1(c_out)
        self.bn=BatchNorm2d(c_out)
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.SATT=SATT_pool(c_out,num_nodes)
        self.LSTM=nn.LSTM(num_nodes,num_nodes,batch_first=True)#b*n,l,c
        
    
    def forward(self,x,support):
        residual = self.conv1(x)
        
        x=self.time_conv(x)
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*torch.sigmoid(x2)
        
        
        x=self.multigcn(x,support) 
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*(torch.sigmoid(x2)) 
        
          
        T_coef=self.TAT(x)
        T_coef=T_coef.transpose(-1,-2)
        x=torch.einsum('bcnl,blq->bcnq',x,T_coef)       
        out=self.bn(x+residual[:, :, :, -x.size(3):])
        #out=torch.sigmoid(x)
        return out



class GCNPool_h(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,
                 Kt,dropout,pool_nodes,support_len=3,order=2):
        super(GCNPool_h,self).__init__()
        self.time_conv=Conv2d(c_in, 2*c_out, kernel_size=(1, Kt),padding=(0,0),
                          stride=(1,1), bias=True,dilation=2)
        
        self.multigcn=multi_gcn_time(c_out,2*c_out,Kt,dropout,support_len,order)
        self.multigcn1=multi_gcn_batch(c_out,2*c_out,Kt,dropout,support_len,order)
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        self.TAT=TATT_1(c_out,num_nodes,tem_size)
        self.c_out=c_out
        #self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn=BatchNorm2d(c_out)
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        
        self.dynamic_gcn=T_cheby_conv_ds_1(c_out,2*c_out,order+1,Kt)
        self.gate=gate1(2*c_out)
    
    def forward(self,x,support,A):
        residual = self.conv1(x)
        
        x=self.time_conv(x)
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*torch.sigmoid(x2)
        #print(x.shape)
        #dynamic_adj=self.SATT(x)
        new_support=[]
        new_support.append(support[0]+A)
        new_support.append(support[1]+A)
        new_support.append(support[2]+A)
        x=self.multigcn1(x,new_support)        
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*(torch.sigmoid(x2))
        
        
        T_coef=self.TAT(x)
        T_coef=T_coef.transpose(-1,-2)
        x=torch.einsum('bcnl,blq->bcnq',x,T_coef)  
              
        out=self.bn(x+residual[:, :, :, -x.size(3):])       
        return out
   
           
class GCNPool(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,
                 Kt,dropout,pool_nodes,support_len=3,order=2):
        super(GCNPool,self).__init__()
        self.time_conv=Conv2d(c_in, 2*c_out, kernel_size=(1, Kt),padding=(0,0),
                          stride=(1,1), bias=True,dilation=2)
        
        self.multigcn=multi_gcn_time(c_out,2*c_out,Kt,dropout,support_len,order) # 32 64
        
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        self.TAT=TATT_1(c_out,num_nodes,tem_size)
        self.c_out=c_out
        #self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn=BatchNorm2d(c_out)
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)

        # 64-64 32-64
         
    
    def forward(self,x,support):
        residual = self.conv1(x)
        #print(' before time conv')
       
        x=self.time_conv(x)
        #print(' after time conv')
        #print(x.shape)# 64, 64, 792, 8
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*torch.sigmoid(x2)
        #print(' after temp') #64, 32, 792, 8
        #print(x.shape)
        
        x=self.multigcn(x,support)
        #print(' after gcn') #64, 64, 792, 6
        #print(x.shape)
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*(torch.sigmoid(x2)) 
        #x=F.dropout(x,0.3,self.training)
        #print(' after temp') # 64, 32, 792, 6
        #print(x.shape)
        T_coef=self.TAT(x)
        #print(' after tat') #64, 32, 792, 6
        #print(x.shape) 
        #print(T_coef.shape)
        T_coef=T_coef.transpose(-1,-2)
        x=torch.einsum('bcnl,blq->bcnq',x,T_coef)
        out=self.bn(x+residual[:, :, :, -x.size(3):])
        #print('shape of out')
        #print(out.shape)
        return out



class GCNPool_my(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size,
                 Kt, dropout, pool_nodes, support_len=3, order=2):
        super(GCNPool_my, self).__init__()
        self.time_conv = Conv2d(c_in, 2 * c_out, kernel_size=(1, Kt), padding=(0, 0),
                                stride=(1, 1), bias=True, dilation=2)

        self.multigcn = multi_gcn_time(c_out, 2 * c_out, Kt, dropout, support_len, order)

        self.num_nodes = num_nodes
        self.tem_size = tem_size
        #self.TAT = TATT_1(c_out, num_nodes, tem_size)
        
        self.enc_embedding1 = nn.Conv2d(in_channels=32,
                                    out_channels=128,
                                    kernel_size=(1, 1))
        self.enc_embedding2 = nn.Conv2d(in_channels=128,
                                        out_channels=32,
                                        kernel_size=(1, 1))
        
        
        
        self.encoder= Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, 1, attention_dropout=0.05,
                                        output_attention='store_true'),
                        128, 8),
                    128,
                    2048,
                    moving_avg=25,
                    dropout=0.05,
                    activation='geLu'
                ) for l in range(2)
            ],
            norm_layer=my_Layernorm(128)
        )
        self.c_out = c_out
        self.bn = LayerNorm([c_out, num_nodes, tem_size])
        self.bn = BatchNorm2d(c_out)

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)

    def forward(self, x, support):
        residual = self.conv1(x)

        x = self.time_conv(x)
        x1, x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * torch.sigmoid(x2)

        x = self.multigcn(x, support)
        x1, x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * (torch.sigmoid(x2))
        # x=F.dropout(x,0.3,self.training)

        # T_coef = self.TAT(x)
        #print('before embedding')
        #print(x.shape)
        x = self.enc_embedding1(x)
        #print('after embedding')
        #print(x.shape)

        shape = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(shape[0] * shape[2], shape[3], shape[1])
       
        x, att = self.encoder(x)
        
        #print('after encoder')
        #print(x.shape)
        x = x.view(shape[0],shape[2],shape[3],-1).permute(0,3,1,2).contiguous()
        #print(x.shape)
        x = self.enc_embedding2(x)
        #print(x.shape)
        out = self.bn(x + residual[:, :, :, -x.size(3):])
        return out


class GCNPool_my2(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size,
                 Kt, dropout, pool_nodes, support_len=3, order=2):
        super(GCNPool_my2, self).__init__()
        self.time_conv = Conv2d(32, 2 * c_out, kernel_size=(1, Kt), padding=(0, 0),
                                stride=(1, 1), bias=True, dilation=2)

        self.multigcn = multi_gcn_d(c_out, 2 * c_out, Kt, dropout, support_len, order)  # 32 64

        self.num_nodes = num_nodes
        self.tem_size = tem_size
        self.TAT = TATT_1(c_out, num_nodes, tem_size)
        self.c_out = c_out
        # self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn = BatchNorm2d(c_out)

        self.conv1 = Conv2d(32, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)

        # 64-64 32-64
        self.trend_att = MultiHeadAttentionAwareTemporalContex_q1d_k1d(8, c_in,1, 12, 3, dropout)
        self.time_embed1 = nn.Conv2d(in_channels=32,
                                     out_channels=64,
                                     kernel_size=(1, 1))
        self.time_embed2 = nn.Conv2d(in_channels=64,
                                     out_channels=32,
                                     kernel_size=(1, 1))

        self.conv_temp1 = nn.Conv2d(in_channels=10,
                                    out_channels=6,
                                    kernel_size=(1, 1))
        self.conv_temp2 = nn.Conv2d(in_channels=5,
                                    out_channels=3,
                                    kernel_size=(1, 1))

    def forward(self, x, support):
        residual = self.conv1(x)
        # print(' before time conv')
        # print(x.shape) #64 64 792 12
        # x=self.time_conv(x)
        # print(x.shape)
        x = self.time_embed1(x)
        x = x.permute(0, 2, 3, 1)

        x = self.trend_att(x, x, x)
        x = x.permute(0, 3, 1, 2)
        x = self.time_embed2(x)

        # print(' after time conv')
        # print(x.shape)# 64, 64, 792, 8
        '''
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*torch.sigmoid(x2)
        '''
        # print(' after temp') #64, 32, 792, 8
        # print(x.shape)

        x = self.multigcn(x, support)
        # print(' after gcn') #64, 64, 792, 6
        # print(x.shape)
        x1, x2 = torch.split(x, [self.c_out, self.c_out], 1)
        x = torch.tanh(x1) * (torch.sigmoid(x2))
        # x=F.dropout(x,0.3,self.training)
        # print(' after temp') # 64, 32, 792, 6
        if x.shape[3] == 10:
            x = self.conv_temp1(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        else:
            x = self.conv_temp2(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        T_coef = self.TAT(x)
        # print(' after tat') #64, 32, 792, 6
        # print(x.shape)
        # print(T_coef.shape)
        T_coef = T_coef.transpose(-1, -2)
        x = torch.einsum('bcnl,blq->bcnq', x, T_coef)

        # print(x.shape)
        # print(residual.shape)
        out = self.bn(x + residual[:, :, :, -x.size(3):])
        # print('shape of out')
        # print(out.shape)
        return out
   
    
class GCNPool_my3(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,
                 Kt,dropout,pool_nodes,support_len=3,order=2):
        super(GCNPool_my3,self).__init__()
        
        self.multigcn=multi_gcn_time(c_out,2*c_out,Kt,dropout,support_len,order) # 32 64
        
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        self.TAT=TATT_1(c_out,num_nodes,tem_size)
        self.c_out=c_out
        #self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn=BatchNorm2d(c_out)
        
        self.conv1=Conv2d(32, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)

        # 64-64 32-64
        self.trend_att= MultiHeadAttentionAwareTemporalContex_q1d_k1d(8, 64, 1, 12, 3, dropout)
        self.time_embed1= nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=(1,1))
        self.time_embed2= nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=(1,1))

        self.conv_temp1 = nn.Conv2d(in_channels=10,
                                    out_channels=6,
                                    kernel_size=(1,1))
        self.conv_temp2 = nn.Conv2d(in_channels=5,
                                    out_channels=3,
                                    kernel_size=(1,1))

    def forward(self,x,support):
        residual = self.conv1(x)
        #print(' before time conv')
        #print(x.shape) #64 64 792 12
        #x=self.time_conv(x)
        #print(x.shape)  
        #print(' after time conv')
        #print(x.shape)# 64, 64, 792, 8
        x=self.multigcn(x,support)
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*torch.sigmoid(x2)
        
        #print(' after temp') #64, 32, 792, 8
        #print(x.shape)
       
        #x=F.dropout(x,0.3,self.training)
        #print(' after temp') # 64, 32, 792, 6
        #print(x.shape)
        x=self.conv_temp1(x.permute(0,3,2,1)).permute(0,3,2,1)
      
        T_coef=self.TAT(x)
        #print(' after tat') #64, 32, 792, 6
        #print(x.shape) 
        #print(T_coef.shape)
        T_coef=T_coef.transpose(-1,-2)
        x=torch.einsum('bcnl,blq->bcnq',x,T_coef)
       
        #print(' after gcn') #64, 64, 792, 6
        #print(x.shape)
        #print(x
        #print(residual.shape)
        out=self.bn(x+residual[:, :, :, -x.size(3):])
        #print('shape of out')
        #print(out.shape)
        return out


class GCNPool_my4(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,
                 Kt,dropout,pool_nodes,support_len=3,order=2):
        super(GCNPool_my4,self).__init__()
        self.time_conv=Conv2d(32 ,2*c_out, kernel_size=(1, Kt),padding=(0,0),
                          stride=(1,1), bias=True,dilation=2)
        
        self.multigcn=multi_gcn_d(c_out,2*c_out,Kt,dropout,support_len,order) # 32 64
        
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        self.TAT=TATT_1(c_out,num_nodes,tem_size)
        self.c_out=c_out
        #self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn=BatchNorm2d(c_out)
        
        self.conv1=Conv2d(32, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)

        # 64-64 32-64
        self.trend_att= MultiHeadAttentionAwareTemporalContex_q1d_k1d(8, 64, 1, 12, 3, dropout)
        self.time_embed1= nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=(1,1))
        self.time_embed2= nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=(1,1))

        self.conv_temp1 = nn.Conv2d(in_channels=10,
                                    out_channels=6,
                                    kernel_size=(1,1))
        self.conv_temp2 = nn.Conv2d(in_channels=2,
                                    out_channels=3,
                                    kernel_size=(1,1))


    def forward(self,x,support):
        residual = self.conv1(x)
        #print(' before time conv')
        #print(x.shape) #64 64 792 12
        #x=self.time_conv(x)
        #print(x.shape)
        x= self.time_embed1(x)
        x=x.permute(0,2,3,1)

        x=self.trend_att(x,x,x)
        x=x.permute(0,3,1,2)
        x= self.time_embed2(x)
        
        #print(' after time conv')
        #print(x.shape)# 64, 64, 792, 8
        '''
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*torch.sigmoid(x2)
        '''
        #print(' after temp') #64, 32, 792, 8
        #print(x.shape)
        
        x=self.multigcn(x,support)
        #print(' after gcn') #64, 64, 792, 6
        #print(x.shape)
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*(torch.sigmoid(x2)) 
        #x=F.dropout(x,0.3,self.training)
        #print(' after temp') # 64, 32, 792, 6
        if x.shape[3]==10:
           x=self.conv_temp1(x.permute(0,3,2,1)).permute(0,3,2,1)
        
        else:
           x=self.conv_temp2(x.permute(0,3,2,1)).permute(0,3,2,1)
        
        T_coef=self.TAT(x)
        #print(' after tat') #64, 32, 792, 6
        #print(x.shape) 
        #print(T_coef.shape)
        T_coef=T_coef.transpose(-1,-2)
        x=torch.einsum('bcnl,blq->bcnq',x,T_coef)
        
        
        #print(x.shape)
        #print(residual.shape)
        out=self.bn(x+residual[:, :, :, -x.size(3):])
        #print('shape of out')
        #print(out.shape)
        return out


class GCNPool_my5(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,
                 Kt,dropout,pool_nodes,support_len=3,order=2):
        super(GCNPool_my5,self).__init__()
        self.time_conv=Conv2d(32 ,2*c_out, kernel_size=(1, Kt),padding=(0,0),
                          stride=(1,1), bias=True,dilation=2)
        
        self.multigcn=multi_gcn_time1(c_out,2*c_out,Kt,dropout,support_len,order) # 32 64
        
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        self.TAT=TATT_1(c_out,num_nodes,tem_size)
        self.c_out=c_out
        #self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn=BatchNorm2d(c_out)
        
        self.conv1=Conv2d(32, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)

        # 64-64 32-64
        self.trend_att= MultiHeadAttentionAwareTemporalContex_q1d_k1d(8, 64, 1, 12, 3, dropout)
        self.time_embed1= nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=(1,1))
        self.time_embed2= nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=(1,1))

        self.conv_temp1 = nn.Conv2d(in_channels=10,
                                    out_channels=6,
                                    kernel_size=(1,1))
        self.conv_temp2 = nn.Conv2d(in_channels=5,
                                    out_channels=3,
                                    kernel_size=(1,1))

    def forward(self,x,support):
        residual = self.conv1(x)
        #print(' before time conv')
        #print(x.shape) #64 64 792 12
        #x=self.time_conv(x)
        #print(x.shape)
        x= self.time_embed1(x)
        x=x.permute(0,2,3,1)

        x=self.trend_att(x,x,x)
        x=x.permute(0,3,1,2)
        x= self.time_embed2(x)
        
        #print(' after time conv')
        #print(x.shape)# 64, 64, 792, 8
        '''
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*torch.sigmoid(x2)
        '''
        #print(' after temp') #64, 32, 792, 8
        #print(x.shape)
        x=self.multigcn(x, support)
        #print(' after gcn') #64, 64, 792, 6
        #print(x.shape)
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*(torch.sigmoid(x2)) 
        #x=F.dropout(x,0.3,self.training)
        #print(' after temp') # 64, 32, 792, 6
        if x.shape[3]==10:
           x=self.conv_temp1(x.permute(0,3,2,1)).permute(0,3,2,1)
        
        else:
           x=self.conv_temp2(x.permute(0,3,2,1)).permute(0,3,2,1)
        
        T_coef=self.TAT(x)
        #print(' after tat') #64, 32, 792, 6
        #print(x.shape) 
        #print(T_coef.shape)
        T_coef=T_coef.transpose(-1,-2)
        x=torch.einsum('bcnl,blq->bcnq',x,T_coef)
        
        
        #print(x.shape)
        #print(residual.shape)
        out=self.bn(x+residual[:, :, :, -x.size(3):])
        #print('shape of out')
        #print(out.shape)
        return out


class GCNPool_my6(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,
                 Kt,dropout,pool_nodes,support_len=3,order=2):
        super(GCNPool_my6,self).__init__()
        self.time_conv=Conv2d(32 ,2*c_out, kernel_size=(1, Kt),padding=(0,0),
                          stride=(1,1), bias=True,dilation=2)
        
        self.multigcn=multi_gcn_time1(c_out,2*c_out,Kt,dropout,support_len,order) # 32 64
        
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        self.TAT=TATT_1(c_out,num_nodes,tem_size)
        self.c_out=c_out
        #self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn=BatchNorm2d(c_out)
        
        self.conv1=Conv2d(32, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)

        # 64-64 32-64
        self.trend_att= MultiHeadAttentionAwareTemporalContex_q1d_k1d(8, 64, 1, 12, 3, dropout)
        self.time_embed1= nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=(1,1))
        self.time_embed2= nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=(1,1))

        self.conv_temp1 = nn.Conv2d(in_channels=10,
                                    out_channels=6,
                                    kernel_size=(1,1))
        self.conv_temp2 = nn.Conv2d(in_channels=5,
                                    out_channels=3,
                                    kernel_size=(1,1))

    def forward(self,x,support):
        residual = self.conv1(x)
        #print(' before time conv')
        #print(x.shape) #64 64 792 12
        #x=self.time_conv(x)
        #print(x.shape)
        '''
        x= self.time_embed1(x)
        x=x.permute(0,2,3,1)

        x=self.trend_att(x,x,x)
        x=x.permute(0,3,1,2)
        x= self.time_embed2(x)
        '''
        #print(' after time conv')
        #print(x.shape)# 64, 64, 792, 8
        '''
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*torch.sigmoid(x2)
        '''
        #print(' after temp') #64, 32, 792, 8
        #print(x.shape)
        x=self.multigcn(x, support)
        #print(' after gcn') #64, 64, 792, 6
        #print(x.shape)
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*(torch.sigmoid(x2)) 
        #x=F.dropout(x,0.3,self.training)
        #print(' after temp') # 64, 32, 792, 6
        if x.shape[3]==10:
           x=self.conv_temp1(x.permute(0,3,2,1)).permute(0,3,2,1)
        
        else:
           x=self.conv_temp2(x.permute(0,3,2,1)).permute(0,3,2,1)
        
        T_coef=self.TAT(x)
        #print(' after tat') #64, 32, 792, 6
        #print(x.shape) 
        #print(T_coef.shape)
        T_coef=T_coef.transpose(-1,-2)
        x=torch.einsum('bcnl,blq->bcnq',x,T_coef)
        
        
        #print(x.shape)
        #print(residual.shape)
        out=self.bn(x+residual[:, :, :, -x.size(3):])
        #print('shape of out')
        #print(out.shape)
        return out



class cat(nn.Module):
    def __init__(self,c_in,tem_size):
        super(cat,self).__init__()
    
        self.w1=nn.Parameter(torch.rand(c_in,tem_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w1)
        self.w2=nn.Parameter(torch.rand(c_in,tem_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w2)

        self.c_in=c_in
        
    def forward(self,seq,seq_dis):
        
        c1 = seq.transpose(1,3)
        c2 = seq_dis.transpose(1,3)
        print(c1.shape)
        A= torch.mul(c1,self.w1)
        B=torch.mul(c2,self.w2)

        return (A+B).transpose(1,3)


class cat1(nn.Module):
    def __init__(self,num_nodes):
        super(cat1,self).__init__()
    
        self.w1=nn.Parameter(torch.rand(num_nodes,num_nodes), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w1)
        self.w2=nn.Parameter(torch.rand(num_nodes,num_nodes), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w2)

        
    def forward(self,seq,seq_dis):
        
        c1 = torch.einsum('ncvl,vw->ncwl',(seq,self.w1))
        c2 = torch.einsum('ncvl,vw->ncwl',(seq_dis,self.w2))

        return c1+c2


class cat2(nn.Module):
    def __init__(self,c_in,tem_size):
        super(cat2,self).__init__()
    
        self.w1=nn.Parameter(torch.rand(c_in,tem_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w1)
        self.w2=nn.Parameter(torch.rand(c_in,tem_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w2)
        self.w3=nn.Parameter(torch.rand(c_in,tem_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w3)

        self.c_in=c_in
        
    def forward(self,seq,seq_dis, seq_d):
        c1 = seq.transpose(1,3)# b*t, n, f_in
        c2 = seq_dis.transpose(1,3)
        c3 = seq_d.transpose(1,3)
    
        return (torch.mul(c1,self.w1)+torch.mul(c2,self.w2)+torch.mul(c3, self.w3)).transpose(1,3)


def attention(query, key, value, att,mask=None, dropout=None):
    '''

    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)
    try:
        att = att[:,:,:,-scores.shape(3):,:]
    except:
        att = 0

    scores=scores+att
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)


def clones(module, N):
    '''
    Produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttentionAwareTemporalContex_qc_kc(nn.Module):  # key causal; query causal;
    def __init__(self, nb_head, d_model, num_of_hours, points_per_hour, kernel_size=3, dropout=.0):
        '''
        :param nb_head:
        :param d_model:
        :param num_of_hours:
        :param points_per_hour:
        :param kernel_size:
        :param dropout:
        '''
        super(MultiHeadAttentionAwareTemporalContex_qc_kc, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.padding = kernel_size - 1
        self.conv1Ds_aware_temporal_context = clones(nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)), 2)  # # 2 causal conv: 1  for query, 1 for key



        self.dropout = nn.Dropout(p=dropout)
        self.h_length = num_of_hours * points_per_hour


    def forward(self, query, key, value,att, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.h_length > 0:
                query_h, key_h = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :], key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query, key = [l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []

            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value,att, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x), self.attn



class MultiHeadAttentionAwareTemporalContex_q1d_k1d(nn.Module):  # 1d conv on query, 1d conv on key
    def __init__(self, nb_head, d_model, num_of_hours, points_per_hour, kernel_size=3, dropout=.0):

        super(MultiHeadAttentionAwareTemporalContex_q1d_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.padding = (kernel_size - 1)//2

        self.conv1Ds_aware_temporal_context = clones(nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)),  2)  # # 2 causal conv: 1  for query, 1 for key
        self.dropout = nn.Dropout(p=dropout)
        self.h_length = num_of_hours * points_per_hour


    def forward(self, query, key, value,att, mask=None, query_multi_segment=False, key_multi_segment=False):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :param query_multi_segment: whether query has mutiple time segments
        :param key_multi_segment: whether key has mutiple time segments
        if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []

            if self.h_length > 0:
                query_h, key_h = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :], key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :]))]
                query_list.append(query_h)
                key_list.append(key_h)

            query = torch.cat(query_list, dim=3)
            key = torch.cat(key_list, dim=3)

        elif (not query_multi_segment) and (not key_multi_segment):

            query, key = [l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        elif (not query_multi_segment) and (key_multi_segment):

            query = self.conv1Ds_aware_temporal_context[0](query.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)

            key_list = []


            if self.h_length > 0:
                key_h = self.conv1Ds_aware_temporal_context[1](key[:, :, self.w_length + self.d_length:self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
                key_list.append(key_h)

            key = torch.cat(key_list, dim=3)

        else:
            import sys
            print('error')
            sys.out

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, att,mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)

        return self.linears[-1](x),self.attn



class GCNPool_my8(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,
                 Kt,dropout,pool_nodes,support_len=3,order=2):
        super(GCNPool_my8,self).__init__()
        
        self.multigcn=multi_gcn_d(c_in,2*c_out,Kt,dropout,support_len,order) # 32 64
        
        self.num_nodes=num_nodes
        self.tem_size=tem_size
        self.c_out=c_out
        #self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.bn=BatchNorm2d(c_out)
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)

        # 64-64 32-64
        self.trend_att= MultiHeadAttentionAwareTemporalContex_q1d_k1d(8, 64, 1, 12, 3, dropout)
        self.time_embed1= nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=(1,1))
        self.time_embed2= nn.Conv2d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=(1,1))


    def forward(self,x,support):
        residual = self.conv1(x)
   
        
        x=self.multigcn(x,support)
        #print(' after gcn') #64, 64, 792, 6
        #print(x.shape)
        x1,x2=torch.split(x,[self.c_out,self.c_out],1)
        x=torch.tanh(x1)*(torch.sigmoid(x2)) 
        #x=F.dropout(x,0.3,self.training)
        #print(' after temp') # 64, 32, 792, 6
       
        
        x= self.time_embed1(x)
        x=x.permute(0,2,3,1)

        x=self.trend_att(x,x,x)
        x=x.permute(0,3,1,2)
        x= self.time_embed2(x)
        
        
        #print(x.shape)
        #print(residual.shape)
        out=self.bn(x+residual[:, :, :, -x.size(3):])
        #print('shape of out')
        #print(out.shape)
        return out




class GCN(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, support):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, F_in) 4 256 307
        :return: (batch_size, N, F_out)
        '''
        x=x.permute(0,2,1)
        return F.relu(self.Theta(torch.matmul(support[0], x)))

class GCN1(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(GCN1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, support):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, F_in) 4 307 128
        :return: (batch_size, N, F_out)
        '''
        return F.relu(self.Theta(torch.matmul(support[0], x)))

class spatialGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(spatialGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, supports):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''

        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        A= self.Theta(torch.matmul(supports[0], x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2)
        A1= self.Theta(torch.matmul(supports[2], x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2)
        return F.relu(A+A1)


