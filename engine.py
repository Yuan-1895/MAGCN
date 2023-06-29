import torch.optim as optim
from model import *
import util
import torch



class trainer1():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, decay):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

class trainer01():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, decay):
        self.model = mymymy_solo(device, num_nodes, dropout, supports=supports, length=12,
                           in_dim=in_dim, out_dim=24,
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse
    
    
class trainer2():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = ASTGCN_Recent(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse    
    
class trainer3():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = GRCN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse     
    
class trainer4():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = Gated_STGCN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse     

class trainer10():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = LSTM(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,1,num_nodes,12]
       
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse 



class trainerGNN():
    def __init__(self, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, decay):
        self.model = STSGCN(adj=supports, history=history, num_of_vertices=num_of_vertices, in_dim=in_dim, hidden_dims=hidden_dims,
            first_layer_embedding_size=first_layer_embedding_size, out_layer_dim=out_layer_dim, activation=activation, use_mask=use_mask, temporal_emb=temporal_emb,
            spatial_emb=spatial_emb, horizon=horizon, strides=strides)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae

        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output= self.model(input,real_val)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output= self.model(input,real_val)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse