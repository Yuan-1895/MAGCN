import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import copy

import numpy as np
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter,LayerNorm,InstanceNorm2d

from utils import ST_BLOCK_0 #ASTGCN
from utils import ST_BLOCK_1 #DGCN_Mask/DGCN_Res
from utils import ST_BLOCK_2_r #DGCN_recent
from utils import *
from utils import ST_BLOCK_4 #Gated-STGCN
from utils import ST_BLOCK_5 #GRCN
from utils import ST_BLOCK_6 #OTSGGCN
from utils import multi_gcn #gwnet

from utils import multi_gcn_d

from utils import cat1

from utils import MultiHeadAttentionAwareTemporalContex_q1d_k1d, MultiHeadAttentionAwareTemporalContex_qc_kc
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
"""
the parameters:
x-> [batch_num,in_channels,num_nodes,tem_size],
"""
def clones(module, N):
    '''
    Produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ASTGCN_Recent(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(ASTGCN_Recent,self).__init__()
        self.block1=ST_BLOCK_0(in_dim,dilation_channels,num_nodes,length,K,Kt)
        self.block2=ST_BLOCK_0(dilation_channels,dilation_channels,num_nodes,length,K,Kt)
        self.final_conv=Conv2d(length,12,kernel_size=(1, dilation_channels),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        
    def forward(self,input):
        x=self.bn(input)
        adj=self.supports[0]
        x,_,_ = self.block1(x,adj)
        x,d_adj,t_adj = self.block2(x,adj)
        x = x.permute(0,3,2,1)
        x = self.final_conv(x)#b,12,n,1
        return x,d_adj,t_adj
    


class LSTM(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None, in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(LSTM,self).__init__()
        self.lstm=nn.LSTM(in_dim,dilation_channels,batch_first=True)#b*n,l,c
        self.c_out=dilation_channels
        tem_size=out_dim
        self.tem_size=tem_size
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        
        
    def forward(self,input):
        x=input
        shape = x.shape
        h = Variable(torch.zeros((1,shape[0]*shape[2],self.c_out)))#.cuda()
        c = Variable(torch.zeros((1,shape[0]*shape[2],self.c_out)))#.cuda()
        hidden=(h,c)
        
        x=x.permute(0,2,3,1).contiguous().view(shape[0]*shape[2],shape[3],shape[1])
        x,hidden=self.lstm(x,hidden)
        x=x.view(shape[0],shape[2],shape[3],self.c_out).permute(0,3,1,2).contiguous()
        x=self.conv1(x)#b,c,n,l
        print(x.shape)
        return x,hidden[0],hidden[0]

        
class Gated_STGCN(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(Gated_STGCN,self).__init__()
        tem_size=length
        self.block1=ST_BLOCK_4(in_dim,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_4(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block3=ST_BLOCK_4(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
    def forward(self,input):
        x=self.bn(input)
        adj=self.supports[0]
        
        x=self.block1(x,adj)
        x=self.block2(x,adj)
        x=self.block3(x,adj)
        x=self.conv1(x)#b,12,n,1
        return x,adj,adj 




#gwnet    
class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1,out_dim=24,residual_channels=32,
                 dilation_channels=32,skip_channels=256,
                 end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        
        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len +=1
            



        
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                
                new_dilation *=2
                receptive_field += additional_scope
                
                additional_scope *= 2
                
                self.gconv.append(multi_gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_1=BatchNorm2d(in_dim,affine=False)

    def forward(self, input):
        input=self.bn_1(input[:,0:1,:,:])
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]56

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
           
            
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)           

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x,adp,adp


class mymymy(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32,
                 dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(mymymy, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs1 = nn.ModuleList()
        self.gate_convs1 = nn.ModuleList()
        self.filter_convs2 = nn.ModuleList()
        self.gate_convs2 = nn.ModuleList()
        self.residual_convs1 = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.residual_convs2 = nn.ModuleList()
        self.trend1 = nn.ModuleList()
        self.trend2 = nn.ModuleList()
        self.bn0 = nn.ModuleList()
        self.gconv0 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()


        self.trans = nn.ModuleList()



        self.start_conv1 = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.start_conv2 = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)

        self.nodevec3 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec4 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len += 1
        for b in range(blocks):
            self.trend1.append(MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head=8, d_model=residual_channels,
                                                                             num_of_hours=1, points_per_hour=13-3*b,
                                                                             kernel_size=3, dropout=dropout))
            self.trend2.append(MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head=8, d_model=residual_channels,
                                                                             num_of_hours=1, points_per_hour=13-3*b,
                                                                             kernel_size=3, dropout=dropout))
            self.trend1.append(MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head=8, d_model=residual_channels,
                                                                             num_of_hours=1, points_per_hour=12-3*b,
                                                                             kernel_size=3, dropout=dropout))
            self.trend2.append(MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head=8, d_model=residual_channels,
                                                                             num_of_hours=1, points_per_hour=12-3*b,
                                                                             kernel_size=3, dropout=dropout))


        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs1.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs1.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.filter_convs2.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs2.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs1.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.residual_convs2.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn0.append(nn.BatchNorm2d(residual_channels))


                self.bn2.append(nn.BatchNorm2d(residual_channels))

                self.trans.append(cat1(num_nodes))

                new_dilation *= 2
                receptive_field += additional_scope

                additional_scope *= 2

                self.gconv0.append(
                    multi_gcn(dilation_channels, residual_channels, dropout, support_len=1))
                self.gconv2.append(
                    multi_gcn(dilation_channels, residual_channels, dropout, support_len=1))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_1 = BatchNorm2d(in_dim, affine=False)

    def forward(self, input):
        #1
        '''
        input1=input[:,0:1,:,:]
        input2 = input[:, 1:2, :, :]
        
        #2
        '''
        input1=input[:,0:1,:,:]
        input2 = input[:, 0:1, :, :]
        
        #3
        '''
        input1=input
        input2 = input
        '''
        input1 = self.bn_1(input1)
        input2 = self.bn_1(input2)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            input1 = nn.functional.pad(input1, (self.receptive_field - in_len, 0, 0, 0))
            input2 = nn.functional.pad(input2, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input1

        x1=input1
        x2 = input2
        x1 = self.start_conv1(x1)
        x2 = self.start_conv2(x2)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports1 = []

        adp1 = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        new_supports1.append(adp1)

        new_supports2 = []

        adp2 = F.softmax(F.relu(torch.mm(self.nodevec3, self.nodevec4)), dim=1)

        new_supports2.append(adp2)

        # WaveNet layers
        att1=0
        att2=0
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]56

            # residual = dilation_func(x, dilation, init_dilation, i)


            # parametrized skip connection
            residual1 = x1
            residual2 = x2
            #att
            x1 = x1.permute(0, 2, 3, 1)  # B N T D
            x1 ,att1= self.trend1[i](x1, x1, x1, att1)  # x:B N T1 D
            x1 = x1.permute(0, 3, 1, 2)  # B D N T1

            x2 = x2.permute(0, 2, 3, 1)  # B N T D
            x2 ,att2= self.trend2[i](x2, x2, x2,att2)  # x:B N T1 D
            x2 = x2.permute(0, 3, 1, 2)  # B D N T1

            x1 = self.gconv0[i](x1, new_supports1)
            x2 = self.gconv2[i](x2, new_supports2)

            # dilated convolution

            filter1 = self.filter_convs1[i](x1)
            filter1 = torch.tanh(filter1)
            gate1 = self.gate_convs1[i](x1)
            gate1 = torch.sigmoid(gate1)
            x1 = filter1 * gate1

            filter2 = self.filter_convs2[i](x2)
            filter2 = torch.tanh(filter1)
            gate2 = self.gate_convs2[i](x2)
            gate2 = torch.sigmoid(gate2)
            x2 = filter2 * gate2

            x1 = x1 + residual1[:, :, :, -x1.size(3):]
            x2 = x2 + residual2[:, :, :, -x2.size(3):]
             #fusion
            x1=self.trans[i](x1,x2)
            ######
            s = x1
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x1 = self.bn0[i](x1)
            x2 = self.bn2[i](x2)


        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x, adp1, adp1




class mymymy_solo(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32,
                 dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(mymymy_solo, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs1 = nn.ModuleList()
        self.gate_convs1 = nn.ModuleList()

        
        self.residual_convs1 = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
       
        self.trend1 = nn.ModuleList()
        
        self.bn0 = nn.ModuleList()
        self.gconv0 = nn.ModuleList()
        


        self.trans = nn.ModuleList()



        self.start_conv1 = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
      

        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)

       
        self.supports_len += 1
        for b in range(blocks):
            self.trend1.append(MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head=8, d_model=residual_channels,
                                                                             num_of_hours=1, points_per_hour=13-3*b,
                                                                             kernel_size=3, dropout=dropout))
           
            self.trend1.append(MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head=8, d_model=residual_channels,
                                                                             num_of_hours=1, points_per_hour=12-3*b,
                                                                             kernel_size=3, dropout=dropout))
          


        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs1.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs1.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

             

                # 1x1 convolution for residual connection
                self.residual_convs1.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn0.append(nn.BatchNorm2d(residual_channels))


                new_dilation *= 2
                receptive_field += additional_scope

                additional_scope *= 2

                self.gconv0.append(
                    multi_gcn_d(dilation_channels, residual_channels, dropout, support_len=2))
                
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_1 = BatchNorm2d(in_dim, affine=False)

    def forward(self, input):
        input1=input
        print(input1.shape)
        input1 = self.bn_1(input1)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            input1 = nn.functional.pad(input1, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input1

        x1=input1
        print(x1.shape)
        x1 = self.start_conv1(x1)
    
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports1 = []

        adp1 = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        new_supports1.append(adp1)

        

        # WaveNet layers
        att1=0
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]56

            # residual = dilation_func(x, dilation, init_dilation, i)


            # parametrized skip connection
            residual1 = x1
       
            #att
            x1 = x1.permute(0, 2, 3, 1)  # B N T D
            x1 ,att1= self.trend1[i](x1, x1, x1, att1)  # x:B N T1 D
            x1 = x1.permute(0, 3, 1, 2)  # B D N T1

            sns.set(font_scale=1.25)
            hm = sns.heatmap(att1,
                             cbar=True,
                             annot=True,
                             square=True,
                             fmt=".3f",
                             vmin=0,  # 刻度阈值
                             vmax=1,
                             linewidths=.5,
                             cmap="RdPu",  # 刻度颜色
                             annot_kws={"size": 10},
                             xticklabels=True,
                             yticklabels=True)  # seaborn.heatmap相关属性
            # 解决中文显示问题
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            # plt.ylabel(fontsize=15,)
            # plt.xlabel(fontsize=15)
            plt.title("主要变量之间的相关性强弱", fontsize=20)
            plt.show()

            x1 = self.gconv0[i](x1, new_supports1)
         

            # dilated convolution

            filter1 = self.filter_convs1[i](x1)
            filter1 = torch.tanh(filter1)
            gate1 = self.gate_convs1[i](x1)
            gate1 = torch.sigmoid(gate1)
            x1 = filter1 * gate1
            print(x1.shape)

          

            x1 = x1 + residual1[:, :, :, -x1.size(3):]
             #fusion
            ######
            s = x1
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x1 = self.bn0[i](x1)


        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        print(x.shape)
        return x, adp1, adp1

    

class mymymy_n(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32,
                 dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(mymymy_n, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs1 = nn.ModuleList()
        self.gate_convs1 = nn.ModuleList()
        self.filter_convs2 = nn.ModuleList()
        self.gate_convs2 = nn.ModuleList()
        self.residual_convs1 = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.residual_convs2 = nn.ModuleList()
        self.trend1 = nn.ModuleList()
        self.trend2 = nn.ModuleList()
        self.bn0 = nn.ModuleList()
        self.gconv0 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()


        self.trans = nn.ModuleList()



        self.start_conv1 = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.start_conv2 = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)

        self.nodevec3 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec4 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len += 1
        for b in range(blocks):
            self.trend1.append(MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head=8, d_model=residual_channels,
                                                                             num_of_hours=1, points_per_hour=13-3*b,
                                                                             kernel_size=3, dropout=dropout))
            self.trend2.append(MultiHeadAttentionAwareTemporalContex_qc_kc(nb_head=8, d_model=residual_channels,
                                                                             num_of_hours=1, points_per_hour=13-3*b,
                                                                             kernel_size=3, dropout=dropout))
            self.trend1.append(MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head=8, d_model=residual_channels,
                                                                             num_of_hours=1, points_per_hour=12-3*b,
                                                                             kernel_size=3, dropout=dropout))
            self.trend2.append(MultiHeadAttentionAwareTemporalContex_qc_kc(nb_head=8, d_model=residual_channels,
                                                                             num_of_hours=1, points_per_hour=12-3*b,
                                                                             kernel_size=3, dropout=dropout))


        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs1.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs1.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.filter_convs2.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs2.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs1.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.residual_convs2.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn0.append(nn.BatchNorm2d(residual_channels))


                self.bn2.append(nn.BatchNorm2d(residual_channels))

                self.trans.append(cat1(num_nodes))

                new_dilation *= 2
                receptive_field += additional_scope

                additional_scope *= 2

                self.gconv0.append(
                    multi_gcn(dilation_channels, residual_channels, dropout, support_len=1))
                self.gconv2.append(
                    multi_gcn(dilation_channels, residual_channels, dropout, support_len=1))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_1 = BatchNorm2d(in_dim, affine=False)

    def forward(self, input):
        input1=input[:,0:1,:,:]
        input2 = input[:, 0:1, :, :]
        input1 = self.bn_1(input1)
        input2 = self.bn_1(input2)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            input1 = nn.functional.pad(input1, (self.receptive_field - in_len, 0, 0, 0))
            input2 = nn.functional.pad(input2, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input1

        x1=input1
        x2 = input2
        x1 = self.start_conv1(x1)
        x2 = self.start_conv2(x2)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports1 = []

        adp1 = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        new_supports1.append(adp1)

        new_supports2 = []

        adp2 = F.softmax(F.relu(torch.mm(self.nodevec3, self.nodevec4)), dim=1)

        new_supports2.append(adp2)

        # WaveNet layers
        att1=0
        att2=0
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]56

            # residual = dilation_func(x, dilation, init_dilation, i)


            # parametrized skip connection
            residual1 = x1
            residual2 = x2
            #att
            x1 = x1.permute(0, 2, 3, 1)  # B N T D
            x1 ,att1= self.trend1[i](x1, x1, x1, att1)  # x:B N T1 D
            x1 = x1.permute(0, 3, 1, 2)  # B D N T1

            x2 = x2.permute(0, 2, 3, 1)  # B N T D
            x2 ,att2= self.trend2[i](x2, x2, x2,att2)  # x:B N T1 D
            x2 = x2.permute(0, 3, 1, 2)  # B D N T1

            

            x1 = self.gconv0[i](x1, new_supports1)
            x2 = self.gconv2[i](x2, new_supports2)

            # dilated convolution

            filter1 = self.filter_convs1[i](x1)
            filter1 = torch.tanh(filter1)
            gate1 = self.gate_convs1[i](x1)
            gate1 = torch.sigmoid(gate1)
            x1 = filter1 * gate1

            filter2 = self.filter_convs2[i](x2)
            filter2 = torch.tanh(filter1)
            gate2 = self.gate_convs2[i](x2)
            gate2 = torch.sigmoid(gate2)
            x2 = filter2 * gate2

            x1 = x1 + residual1[:, :, :, -x1.size(3):]
            x2 = x2 + residual2[:, :, :, -x2.size(3):]
            # fusion
            x1 = self.trans[i](x1, x2)
            ######
            s = x1
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x1 = self.bn0[i](x1)
            x2 = self.bn2[i](x2)


        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x, adp1, adp1



class mymymy_n_att(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32,
                 dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(mymymy_n_att, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs1 = nn.ModuleList()
        self.gate_convs1 = nn.ModuleList()
        self.filter_convs2 = nn.ModuleList()
        self.gate_convs2 = nn.ModuleList()
        self.residual_convs1 = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.residual_convs2 = nn.ModuleList()
        self.trend1 = nn.ModuleList()
        self.trend2 = nn.ModuleList()
        self.bn0 = nn.ModuleList()
        self.gconv0 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()


        self.trans = nn.ModuleList()



        self.start_conv1 = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.start_conv2 = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)

        self.nodevec3 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec4 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len += 1
        for b in range(blocks):
            self.trend1.append(MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head=8, d_model=residual_channels,
                                                                             num_of_hours=1, points_per_hour=13-3*b,
                                                                             kernel_size=3, dropout=dropout))
            self.trend2.append(MultiHeadAttentionAwareTemporalContex_qc_kc(nb_head=8, d_model=residual_channels,
                                                                             num_of_hours=1, points_per_hour=13-3*b,
                                                                             kernel_size=3, dropout=dropout))
            self.trend1.append(MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head=8, d_model=residual_channels,
                                                                             num_of_hours=1, points_per_hour=12-3*b,
                                                                             kernel_size=3, dropout=dropout))
            self.trend2.append(MultiHeadAttentionAwareTemporalContex_qc_kc(nb_head=8, d_model=residual_channels,
                                                                             num_of_hours=1, points_per_hour=12-3*b,
                                                                             kernel_size=3, dropout=dropout))


        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs1.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs1.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.filter_convs2.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs2.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs1.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.residual_convs2.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn0.append(nn.BatchNorm2d(residual_channels))


                self.bn2.append(nn.BatchNorm2d(residual_channels))

                self.trans.append(cat1(num_nodes))

                new_dilation *= 2
                receptive_field += additional_scope

                additional_scope *= 2

                self.gconv0.append(
                    multi_gcn_d(dilation_channels, residual_channels, dropout, support_len=2))
                self.gconv2.append(
                    multi_gcn_d(dilation_channels, residual_channels, dropout, support_len=2))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_1 = BatchNorm2d(in_dim, affine=False)

    def forward(self, input):
        input1=input[:,0:1,:,:]
        input2 = input[:, 0:1, :, :]
        input1 = self.bn_1(input1)
        input2 = self.bn_1(input2)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            input1 = nn.functional.pad(input1, (self.receptive_field - in_len, 0, 0, 0))
            input2 = nn.functional.pad(input2, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input1

        x1=input1
        x2 = input2
        x1 = self.start_conv1(x1)
        x2 = self.start_conv2(x2)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports1 = []

        adp1 = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        new_supports1.append(adp1)

        new_supports2 = []

        adp2 = F.softmax(F.relu(torch.mm(self.nodevec3, self.nodevec4)), dim=1)

        new_supports2.append(adp2)

        # WaveNet layers
        att1=0
        att2=0
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]56

            # residual = dilation_func(x, dilation, init_dilation, i)


            # parametrized skip connection
            residual1 = x1
            residual2 = x2
            #att
            x1 = x1.permute(0, 2, 3, 1)  # B N T D
            x1 ,att1= self.trend1[i](x1, x1, x1, att1)  # x:B N T1 D
            x1 = x1.permute(0, 3, 1, 2)  # B D N T1

            x2 = x2.permute(0, 2, 3, 1)  # B N T D
            x2 ,att2= self.trend2[i](x2, x2, x2,att2)  # x:B N T1 D
            x2 = x2.permute(0, 3, 1, 2)  # B D N T1

            filter1 = self.filter_convs1[i](x1)
            filter1 = torch.tanh(filter1)
            gate1 = self.gate_convs1[i](x1)
            gate1 = torch.sigmoid(gate1)
            x1 = filter1 * gate1

            filter2 = self.filter_convs2[i](x2)
            filter2 = torch.tanh(filter1)
            gate2 = self.gate_convs2[i](x2)
            gate2 = torch.sigmoid(gate2)
            x2 = filter2 * gate2

            x1 = x1 + residual1[:, :, :, -x1.size(3):]
            x2 = x2 + residual2[:, :, :, -x2.size(3):]
            # fusion
            x1 = self.trans[i](x1, x2)
            ######
            s = x1
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x1 = self.gconv0[i](x1, new_supports1)
            x2 = self.gconv2[i](x2, new_supports2)

            # dilated convolution



            x1 = self.bn0[i](x1)
            x2 = self.bn2[i](x2)


        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        print(x.shape)
        return x, adp1, adp1


class gcn_operation(nn.Module):
    def __init__(self, adj, in_dim, out_dim, num_vertices, activation='GLU'):
        """
        图卷积模块
        :param adj: 邻接图
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param num_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(gcn_operation, self).__init__()
        self.adj = adj
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.activation = activation

        assert self.activation in {'GLU', 'relu'}

        if self.activation == 'GLU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def forward(self, x, mask=None):
        """
        :param x: (3*N, B, Cin)
        :param mask:(3*N, 3*N)
        :return: (3*N, B, Cout)
        """
        adj = self.adj
        if mask is not None:
            adj = adj.to(mask.device) * mask

        x = torch.einsum('nm, mbc->nbc', adj.to(x.device), x)  # 3*N, B, Cin

        if self.activation == 'GLU':
            lhs_rhs = self.FC(x)  # 3*N, B, 2*Cout
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # 3*N, B, Cout

            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs

            return out

        elif self.activation == 'relu':
            return torch.relu(self.FC(x))  # 3*N, B, Cout


