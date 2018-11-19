import torch.nn as nn
from torch.autograd import Variable
import torch
from util import *
from convolutional_rnn import Conv2dLSTMCell



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      hidden_dim: int thats the num of channels of the states, like hidden_size
      
    """
    def __init__(self, shape, input_chans, filter_size, hidden_dim):
        super(CLSTM_cell, self).__init__()
        
        self.shape = shape#H,W
        self.input_chans=input_chans
        self.filter_size=filter_size
        self.hidden_dim = hidden_dim
        #self.batch_size=batch_size
        self.padding=(filter_size-1)/2#in this way the output has the same size
        self.conv = nn.Conv2d(self.input_chans + self.hidden_dim, 4*self.hidden_dim, self.filter_size, 1, self.padding)
        
    
    def forward(self, input, hidden_state):
        hidden,c=hidden_state#hidden and c are images with several channels
        #print 'hidden ',hidden.size()
        #print 'input ',input.size()
        combined = torch.cat((input, hidden), 1)#oncatenate in the channels
        #print 'combined',combined.size()
        A=self.conv(combined)
        (ai,af,ao,ag)=torch.split(A,self.hidden_dim,dim=1)#it should return 4 tensors
        i=torch.sigmoid(ai)
        f=torch.sigmoid(af)
        o=torch.sigmoid(ao)
        g=torch.tanh(ag)
                
        next_c=f*c+i*g
        next_h=o*torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self,batch_size):
        return (Variable(torch.zeros(batch_size,self.hidden_dim,self.shape[0],self.shape[1])).cuda(),
                Variable(torch.zeros(batch_size,self.hidden_dim,self.shape[0],self.shape[1])).cuda())


class CLSTM(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int tuple that is the height and width of the filters for state-to-state and input-to-sta
      hidden_dim: int tuple thats the num of channels of the states, like hidden_size
      
    """
    def __init__(self, shape, input_chans, filter_size, hidden_dim,num_layers):
        super(CLSTM, self).__init__()
        
        self.shape = shape#H,W
        self.input_chans=input_chans
        self.filter_size=filter_size
        self.hidden_dim = hidden_dim
        self.num_layers=num_layers
        cell_list=[]
        cell_list.append(Conv2dLSTMCell(in_channels=self.input_chans, 
                                        kernel_size=self.filter_size, 
                                        out_channels=self.hidden_dim[0]))#the first
        #one has a different number of input channels
        
        self.conv = nn.Conv2d(in_channels=hidden_dim[-1], out_channels=1, kernel_size=1, stride=1, padding=0)    
        self.relu = nn.ReLU()
        for idcell in range(1,self.num_layers):
            cell_list.append(Conv2dLSTMCell(in_channels=self.hidden_dim[idcell-1], 
                                            kernel_size=self.filter_size, 
                                            out_channels=self.hidden_dim[idcell]))
        self.cell_list=nn.ModuleList(cell_list)      

    
    def forward(self, input, hidden_state):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W
        """

        current_input = input.transpose(0, 1)#now is seq_len,B,C,H,W
        #current_input=input
#         next_hidden=[]#hidden states(h and c)
        seq_len=current_input.size(0)
        
        for idlayer in range(self.num_layers):#loop for every layer

            hidden_c=hidden_state[idlayer]#hidden and c are images with several channels
            output_inner = []            
            for t in range(seq_len):#loop for every sequence
                hidden_c=self.cell_list[idlayer](current_input[t,...],hidden_c)#cell_list is a list with different conv_lstms 1 for every layer

                output_inner.append(hidden_c[0])
#             next_hidden.append(hidden_c)
            current_input = torch.cat(output_inner, 0).view(current_input.size(0), *output_inner[0].size())#seq_len,B,chans,H,W
        
        prediction = []
#         print('next', output_inner[0].shape)
        for i in range(len(output_inner)):
            pred_i = self.conv(output_inner[i])
            pred_i = self.relu(pred_i)
            prediction.append(pred_i)
        
        return prediction
#         return next_hidden, current_input

    def init_hidden(self,batch_size):
        init_states=[]#this is a list of tuples
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states