from util import *
from training import *

 
def restore_net(idx):

    print("Reloading previous model")
    net = torch.load(args.model_dir+"model_{0}.pkl".format(idx))
    return net


def test(idx,model,reload=False):

    if reload:
        model = restore_net(idx)
        model.eval()
### loading validation dataset
    self_built_dataset = util.Ucsd_loader(args.data_dir+args.testset_name,
                                          args.data_dir+'test_gt',
                                          args.seq_length)
    trainloader = DataLoader(
        self_built_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True) 

    for iteration,valid_data in enumerate(trainloader,0):
        
        print(iteration)
        valid_X,valid_Y = valid_data
        valid_X = Variable(valid_X, requires_grad=False).cuda()

        output_list = model(valid_X)
        mae = 0
        mse = 0
        for j in range(args.batch_size):

            for i in range(args.seq_length):
                targetY = valid_Y[j][i].data.cpu().numpy()

                A = output_list[i][j,0,:,:].data.cpu().numpy()

                gt_cnt = np.sum(targetY)
                pre_cnt = np.sum(A)
                mae += abs(np.sum(targetY) - np.sum(A))
                mse += (np.sum(targetY) - np.sum(A)) ** 2

                print('Gt cnt: %f, Pred cnt: %f' % (gt_cnt, pre_cnt))
                
#                 path = args.img_dir+str(iteration)+str(i)+str(j)+'.png'
#                 A.save(path)

        output_list = None

if __name__== "__main__":
    torch.cuda.set_device(2)

    test(7,None,reload=True)




