from util import *
from training import *
# from convlstm import CLSTM
import matplotlib.pyplot as plt

def restore_net(idx):

    print("Reloading previous model")
    net = torch.load(args.model_dir+"model1_{0}.pkl".format(idx))
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

    total_mae = 0
    total_mse = 0
    step = 0
    
    # mask for ucsd
    roi = np.loadtxt('data/ucsd_mask.txt')
    roi /= 255
    
    for iteration,valid_data in enumerate(trainloader,0):
        print(iteration)
        valid_X,valid_Y = valid_data
        valid_X = Variable(valid_X, requires_grad=False).cuda()

        hidden_state = model.init_hidden(args.batch_size)
        output_list = model(torch.unsqueeze(valid_X, 2), hidden_state)
        mae = 0
        mse = 0
        for j in range(args.batch_size):

            for i in range(args.seq_length):
                targetY = valid_Y[j][i].data.cpu().numpy()

                A = output_list[i][j,0,:,:].data.cpu().numpy() # only the first color channel 0
#                 A = 255*A/np.max(A)
#                 A = A.astype(np.uint8)
#                 print(A)
#                 A = A * roi
                gt_cnt = np.sum(targetY)
                pre_cnt = np.sum(A)
                mae += abs(np.sum(targetY) - np.sum(A))
                mse += (np.sum(targetY) - np.sum(A)) ** 2

                print('Gt cnt: %f, Pred cnt: %f' % (gt_cnt, pre_cnt))
#                 A = Image.fromarray(A).convert("L")                
                path = args.img_dir+str(iteration)+str(i)+str(j)+'.png'
#                 A.save(path)
                plt.imsave(path, A)
                step += 1
        total_mse += mse
        total_mae += mae
        output_list = None
    print('total mse: ', np.sqrt(total_mse/step))
    print('total mae: ', total_mae/step)

if __name__== "__main__":
    torch.cuda.set_device(2)

    test(55,None,reload=True)




