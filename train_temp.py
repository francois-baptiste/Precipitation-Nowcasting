# -*- coding: utf-8 -*-
#! /usr/bin/python3
"""
Created on Thu Sep 21 16:15:53 2017

@author: cx
"""

from util import *
import util
from convlstm import CGRU
import convlstm

def run_training(args,reload=False):     

    #Initialize model
    if reload:
        model_list = []
        print("Reloading exsiting model")
        maximum = 0
        model_name = "model4_"+str(maximum)+".pkl"
        for model_name in os.listdir(args.model_dir):
            num = int(model_name.split("_")[1][:-4])
            if num > maximum:
                maximum = num
        model_name = "model4_"+str(17)+".pkl"
        print(model_name)
        model = torch.load(args.model_dir+model_name)
        start = maximum+1

    else:
        print('Initiating new model')
        
        model = CGRU((args.img_size, args.img_size), 1, 5, (128, 64, 64, 64), 4)
        model.apply(convlstm.weights_init)
        model = model.cuda()
        start = 0

    torch.manual_seed(1)
    self_built_dataset = util.Ucsd_loader(args.data_dir+args.trainset_name, args.data_dir+'train_gt/',
                                          args.seq_length)
    trainloader = DataLoader(
        self_built_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last = True)

    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    loss_ave = 0

######Train the model#######
    for epoch in range(args.epoches):

        print("--------------------------------------------")
        print("EPOCH:",epoch)
        step = 0
        for iteration, data in enumerate(trainloader,0):
            optimizer.zero_grad() 

            loss = 0
            step += 1
            # X is the given data while the Y is the real output
            X, Y = data
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
            
            hidden_state = model.init_hidden(args.batch_size)
            output_list = model(torch.unsqueeze(X, 2), hidden_state)
            

            for i in range(args.seq_length):
                target = Y[:,i,:,:]
                output = torch.squeeze(output_list[i], 1)
                loss += criterion(output, target)
            
            loss_ave += loss.data
            loss.backward()
            optimizer.step()
        print("Epoch %d: %f" % (epoch, loss_ave / step))
        loss_ave = 0
        print("Finished an epoch.Saving the net....... ")
        torch.save(model,args.model_dir+"modelr_{0}.pkl".format(epoch))    

    summary.close()

if __name__=="__main__":
    torch.cuda.set_device(3)
    run_training(args, reload=False)
