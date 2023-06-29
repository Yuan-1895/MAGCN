import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import *
import os
import shutil
import random

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cpu',help='')
parser.add_argument('--data',type=str,default='data/PEMS04_2hour',help='data path')
parser.add_argument('--adjdata',type=str,default='data/PEMS04_2hour/adj_mat.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=307,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=8,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--force', type=str, default=False,help="remove params dir", required=False)
parser.add_argument('--save',type=str,default='./garage/PEMS04_2hour',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--model',type=str,default='mymymy_solo_att',help='adj type')
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')
parser.add_argument("--res_dir", type=str, default="res/gwnet", help="Output directory.")
args = parser.parse_args()
##model repertition
seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


    
def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    adj_dis, adj = util.load_adj_my(args.adjdata, args.adjtype) #demo1
    #adj_dis = np.zeros((shape[0],shape[0]))
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    # scaler = dataloader['scaler']

    #supports = [torch.tensor(i).to(device) for i in adj_mx]
    supports = [torch.tensor(i).to(device) for i in adj_dis]

    print(args)
    if args.model=='gwnet':
        engine = trainer1( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    if args.model=='mymymy_solo_att':
        engine = trainer01( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    elif args.model=='ASTGCN_Recent':
        engine = trainer2( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    elif args.model=='GRCN':
        engine = trainer3( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    elif args.model=='Gated_STGCN':
        engine = trainer4( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    elif args.model=='LSTM':
        engine = trainer10( args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )
    elif args.model=='STSGCN':
        engine = trainerGNN(scaler=scaler, adj=local_adj,history=args.history,num_of_vertices=args.num_of_vertices, in_dim=args.in_dim, hidden_dims=args.hidden_dims,
                     first_layer_embedding_size=args.first_layer_embedding_size, out_layer_dim=args.out_layer_dim, lrate=args.learning_rate, device=device, activation=args.activation,
                     use_mask=args.use_mask, max_grad_norm=args.max_grad_norm, lr_decay=args.lr_decay, temporal_emb=args.temporal_emb, spatial_emb=args.spatial_emb,
                     horizon=12, strides=3)
    
    # check parameters file
    params_path=args.save+"/"+args.model
    if os.path.exists(params_path) and not args.force:
        raise SystemExit("Params folder exists! Select a new params path please!")
    else:
        if os.path.exists(params_path):
            shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Create params directory %s' % (params_path))

    print("start training...",flush=True)
    his_loss =[]
    his_mae=[]
    his_rmse=[]
    his_mape=[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
            #if iter % args.print_every == 0 :
             #   log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
              #  print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)
        his_mape.append(mvalid_mape)
        his_mae.append(mvalid_mae)
        his_rmse.append(mvalid_rmse)


        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), params_path+"/"+args.model+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    '''
    x1=his_loss
    x2=his_mae
    x3=his_rmse
    x4=his_mape
    np.savez_compressed(
        os.path.join(args.res_dir, "gwnet.npz"),
        loss=x1,
        mae=x2,
        rmse=x3,
        mape=x4
    )
    '''
    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(params_path+"/"+args.model+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))
    engine.model.eval()
    
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds,spatial_at,parameter_adj = engine.model(testx)
            preds=preds.transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]


    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    amae = []
    amape = []
    armse = []
    prediction=yhat
    for i in range(12):
        pred = prediction[:,:,i]
        #pred = scaler.inverse_transform(yhat[:,:,i])
        #prediction.append(pred)
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
    
    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(),params_path+"/"+args.model+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
    prediction_path=params_path+"/"+args.model+"_prediction_results"
    ground_truth=realy.cpu().detach().numpy()
    prediction=prediction.cpu().detach().numpy()
    spatial_at=spatial_at.cpu().detach().numpy()
    parameter_adj=parameter_adj.cpu().detach().numpy()
    np.savez_compressed(
            os.path.normpath(prediction_path),
            prediction=prediction,
            spatial_at=spatial_at,
            parameter_adj=parameter_adj,
            ground_truth=ground_truth
        )


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
