from util import *
from cell import ConvLSTMCell
import util

class ConvLSTM(nn.Module):

    def __init__(self):
        super(ConvLSTM,self).__init__()

###declare some parameters that might be used 
        self.conv_pad = 0
        self.conv_kernel_size = 3
        self.conv_stride = 1
        self.pool_pad = 0
        self.pool_kernel_size = 3
        self.pool_stride = 3
        self.hidden_size = 64
        self.size = int((args.img_size+2*self.conv_pad-(self.conv_kernel_size-1)-1)/self.conv_stride+1)
        self.size1 = int((self.size+2*self.pool_pad-(self.pool_kernel_size-1)-1)/self.pool_stride+1)
###define layers
        self.conv = nn.Conv2d(
             in_channels=1,
             out_channels=8,
             kernel_size=3,
             stride=1,
             padding=0)
        self.pool = nn.MaxPool2d(
                     kernel_size=3
                     )
        self.convlstm1 = ConvLSTMCell(
                        shape=[self.size1,self.size1], 
                        input_channel=8, 
                        filter_size=3,
                        hidden_size=self.hidden_size)
        self.convlstm2 = ConvLSTMCell(
                        shape=[self.size1,self.size1], 
                        input_channel=self.hidden_size, 
                        filter_size=3,
                        hidden_size=self.hidden_size)
        self.deconv = nn.ConvTranspose2d(
                        in_channels=self.hidden_size , 
                        out_channels=1, 
                        kernel_size=6,
                        stride=3,
                        padding=0, 
                        output_padding=1, 
                        )
        self.relu = F.relu


    def forward(self,X):
        X_chunked = torch.chunk(X,args.seq_length,dim=1)
        X = None
        output = [None]*args.seq_length
        state_size = [args.batch_size, self.hidden_size]+[self.size1,self.size1]
        hidden1 = Variable(torch.zeros(state_size)).cuda()
        cell1 = Variable(torch.zeros(state_size)).cuda()
        hidden2 = Variable(torch.zeros(state_size)).cuda()
        cell2 = Variable(torch.zeros(state_size)).cuda()
        
        
        for i in range(0, args.seq_length):                                                       
            output[i] = self.conv(X_chunked[i])
            output[i] = self.pool(output[i])
            hidden1, cell1 = self.convlstm1(output[i],(hidden1,cell1))
            hidden2, cell2 = self.convlstm2(hidden1,(hidden2,cell2))
            output[i] = self.deconv(hidden2)
            output[i] = self.relu(output[i])
            
        return output

# from util import *
# from cell import ConvLSTMCell
# import util

# class ConvLSTM(nn.Module):

#     def __init__(self):
#         super(ConvLSTM,self).__init__()

# ###declare some parameters that might be used 
#         self.conv_pad = 0
#         self.conv_kernel_size = 3
#         self.conv_stride = 1
#         self.pool_pad = 0
#         self.pool_kernel_size = 3
#         self.pool_stride = 3
#         self.hidden_size = 64
#         self.size = int((args.img_size+2*self.conv_pad-(self.conv_kernel_size-1)-1)/self.conv_stride+1)
#         self.size1 = int((self.size+2*self.pool_pad-(self.pool_kernel_size-1)-1)/self.pool_stride+1)
# ###define layers
#         self.conv = nn.Conv2d(
#              in_channels=1,
#              out_channels=8,
#              kernel_size=3,
#              stride=1,
#              padding=0)
#         self.pool = nn.MaxPool2d(
#                      kernel_size=3
#                      )
#         self.convlstm1 = ConvLSTMCell(
#                         shape=[self.size1,self.size1], 
#                         input_channel=8, 
#                         filter_size=3,
#                         hidden_size=self.hidden_size)
#         self.convlstm2 = ConvLSTMCell(
#                         shape=[self.size1,self.size1], 
#                         input_channel=self.hidden_size, 
#                         filter_size=3,
#                         hidden_size=self.hidden_size)
#         self.deconv = nn.ConvTranspose2d(
#                         in_channels=self.hidden_size , 
#                         out_channels=1, 
#                         kernel_size=6,
#                         stride=3,
#                         padding=0, 
#                         output_padding=1, 
#                         )
#         self.relu = F.relu


#     def forward(self,X):
#         X_chunked = torch.chunk(X,args.seq_length,dim=1)
#         X = None
#         output = [None]*args.seq_length
#         state_size = [args.batch_size, self.hidden_size]+[self.size1,self.size1]
#         hidden1 = Variable(torch.zeros(state_size)).cuda()
#         cell1 = Variable(torch.zeros(state_size)).cuda()
#         hidden2 = Variable(torch.zeros(state_size)).cuda()
#         cell2 = Variable(torch.zeros(state_size)).cuda()
        
        
#         for i in range(1, args.seq_length):
                                                        
#             output[i] = self.conv(X_chunked[i])     
#             output[i] = self.pool(output[i])
#             hidden1, cell1 = self.convlstm1(output[i],(hidden1,cell1))
#             hidden2, cell2 = self.convlstm2(hidden1,(hidden2,cell2))
#             output[i] = self.deconv(hidden2)
#             output[i] = self.relu(output[i])
            
#         return output