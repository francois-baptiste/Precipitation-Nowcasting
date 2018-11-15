# -*- coding: utf-8 -*-
#! /usr/bin/python3
"""
Created on Thu Sep 21 16:15:53 2017

@author: cx
"""

from util import *
from cell import ConvLSTMCell
import util
from model import ConvLSTM

def run_training(args,reload=False):     

    #Initialize model
    if reload:
        model_list = []
        print("Reloading exsiting model")
        maximum = 0
        model_name = "model_"+str(maximum)+".pkl"
        for model_name in os.listdir(args.model_dir):
            num = int(model_name.split("_")[1][:-4])
            if num > maximum:
                maximum = num
        model_name = "model_"+str(maximum)+".pkl"
        model = torch.load(args.model_dir+model_name)
        start = maximum+1

    else:
        print('Initiating new model')
        
        model = ConvLSTM()
        model = model.cuda()
        start = 0

    torch.manual_seed(1)
    summary = open(args.logs_train_dir+"5_10_2ly.txt","w") ## you can change the name of your summary. 
    self_built_dataset = util.Ucsd_loader(args.data_dir+args.trainset_name, args.data_dir+'train_gt/',
                                          args.seq_length)
#     self_built_dataset = util.Dataloader0(args.data_dir+args.trainset_name, args.seq_start, args.seq_length-args.seq_start)
    trainloader = DataLoader(
        self_built_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last = True)

#     criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    loss_ave = 0

######Train the model#######
    for epoch in range(args.epoches):

        print("--------------------------------------------")
        print("EPOCH:",epoch)
        t = time.time()
        step = 0
        for iteration, data in enumerate(trainloader,0):
            loss = 0
            step += 1
            # X is the given data while the Y is the real output
            X, Y = data
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()

            output_list = model(X)
            optimizer.zero_grad()         

            for i in range(args.seq_length):
#                 targetY = Y[0][i].data.cpu().numpy()

#                 A = output_list[i][0,0,:,:].data.cpu().numpy()

#                 gt_cnt = np.sum(targetY)
#                 pre_cnt = np.sum(A)
#                 mae += abs(np.sum(targetY) - np.sum(A))
#                 mse += (np.sum(targetY) - np.sum(A)) ** 2

#                 print('Gt cnt: %f, Pred cnt: %f' % (gt_cnt, pre_cnt))
                target = Y[:,i,:,:]
                loss += criterion(output_list[i].reshape(target.shape), target)
            
            loss_ave += loss
            loss.backward()
            optimizer.step()
        print("Epoch %d: %f" % (epoch, loss_ave / step))
        loss_ave = 0
        print("Finished an epoch.Saving the net....... ")
        torch.save(model,args.model_dir+"model_{0}.pkl".format(epoch))    

    summary.close()

if __name__=="__main__":
    torch.cuda.set_device(3)
    run_training(args)
