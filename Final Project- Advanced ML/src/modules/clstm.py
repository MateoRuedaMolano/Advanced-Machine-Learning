import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, args, input_size, hidden_size, kernel_size, padding):
        super(ConvLSTMCell,self).__init__()
        self.use_gpu = args.use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + 2*hidden_size, 4 * hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_state_spatial, hidden_state_temporal):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state_spatial is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                prev_state_spatial = (
                    Variable(torch.zeros(state_size)).cuda(),
                    Variable(torch.zeros(state_size)).cuda()
                )
            else:
                prev_state_spatial = (
                    Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size))
                )
                
        if hidden_state_temporal is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                hidden_state_temporal = Variable(torch.zeros(state_size)).cuda()
            else:
                hidden_state_temporal = Variable(torch.zeros(state_size))


        prev_hidden_spatial, prev_cell_spatial = prev_state_spatial

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_hidden_spatial, hidden_state_temporal], 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell_spatial) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        state = [hidden,cell]

        return state

class ConvLSTMCellMask(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, args, input_size, hidden_size, kernel_size, padding):
        super(ConvLSTMCellMask,self).__init__()
        self.use_gpu = args.use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + 2*hidden_size + 1, 4 * hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_mask, prev_state_spatial, hidden_state_temporal):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state_spatial is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size) #concatena todo en un solo vector 
            if self.use_gpu:
                prev_state_spatial = (
                    Variable(torch.zeros(state_size)).cuda(),
                    Variable(torch.zeros(state_size)).cuda()
                )
            else:
                prev_state_spatial = (
                    Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size))
                )
                
        if hidden_state_temporal is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                hidden_state_temporal = Variable(torch.zeros(state_size)).cuda()
            else:
                hidden_state_temporal = Variable(torch.zeros(state_size))


        prev_hidden_spatial, prev_cell_spatial = prev_state_spatial

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_mask, prev_hidden_spatial, hidden_state_temporal], 1)
        del prev_hidden_spatial, hidden_state_temporal
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell_spatial) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)
        del cell_gate, out_gate, remember_gate, in_gate, gates, stacked_inputs

        state = [hidden,cell]

        return state

class ConvGRUMask(nn.Module):
    """
    Generate a convolutional GRU 
    """

    def __init__(self, args, input_size, hidden_size, kernel_size, padding):
        super(ConvGRUMask,self).__init__()
        self.use_gpu = args.use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + 2*hidden_size + 1, 4 * hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_mask, prev_state_spatial, hidden_state_temporal):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state_spatial is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size) #concatena todo en un solo vector 
            if self.use_gpu:
                prev_state_spatial = (
                    Variable(torch.zeros(state_size)).cuda()
                )
            else:
                prev_state_spatial = (
                    Variable(torch.zeros(state_size))
                )
            prev_hidden_spatial = prev_state_spatial
        else:
            prev_hidden_spatial = prev_state_spatial[0]

        if hidden_state_temporal is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                hidden_state_temporal = Variable(torch.zeros(state_size)).cuda()
            else:
                hidden_state_temporal = Variable(torch.zeros(state_size))
        

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_mask, prev_hidden_spatial, hidden_state_temporal], 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        update_gate, reset_gate, current_memory,_ = gates.chunk(4, 1)
        #update_gate, reset_gate = gates.chunk(2, 1)

        # apply sigmoid non linearity
        update_gate = f.sigmoid(update_gate)
        reset_gate = f.sigmoid(reset_gate)

        # apply tanh non linearity
        current_memory = f.tanh(current_memory*reset_gate)

        hidden = (update_gate * prev_hidden_spatial) + ((1-update_gate) * current_memory)

        del update_gate, reset_gate, prev_hidden_spatial, hidden_state_temporal, gates, stacked_inputs

        state = [hidden]

        return state

class ConvGRUCellMask(nn.Module):
    """
    Generate a convolutional GRU 
    """

    def __init__(self, args, input_size, hidden_size, kernel_size, padding):
        super(ConvGRUCellMask,self).__init__()
        self.use_gpu = args.use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + 2*hidden_size + 1, 2 * hidden_size, kernel_size, padding=padding)
        #self.Gates = nn.Conv2d(input_size + 2*hidden_size + 1, 3 * hidden_size, kernel_size, padding=padding)
        self.current= nn.Conv2d(input_size + 1,  hidden_size, kernel_size, padding=padding)
        self.reset_hidden= nn.Conv2d(hidden_size,  hidden_size, kernel_size, padding=padding)

    def forward(self, input_, prev_mask, prev_state_spatial, hidden_state_temporal):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state_spatial is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size) #concatena todo en un solo vector 
            if self.use_gpu:
                prev_state_spatial = (
                    Variable(torch.zeros(state_size)).cuda()
                )
            else:
                prev_state_spatial = (
                    Variable(torch.zeros(state_size))
                )
            prev_hidden_spatial = prev_state_spatial
        else:
            prev_hidden_spatial = prev_state_spatial[0]

        if hidden_state_temporal is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.use_gpu:
                hidden_state_temporal = Variable(torch.zeros(state_size)).cuda()
            else:
                hidden_state_temporal = Variable(torch.zeros(state_size))
        

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_mask, prev_hidden_spatial, hidden_state_temporal], 1)
        gates = self.Gates(stacked_inputs)
        stacked_inputs = torch.cat([input_,prev_mask],1)
        current = self.current(stacked_inputs)

        # chunk across channel dimension
        update_gate, reset_gate = gates.chunk(2, 1)
        #update_gate, reset_gate_temporal, reset_gate_spatial = gates.chunk(3, 1)

        # apply sigmoid non linearity
        update_gate = f.sigmoid(update_gate)
        reset_gate = f.sigmoid(reset_gate)
        #reset_gate_temporal = f.sigmoid(reset_gate_temporal)
        #reset_gate_spatial = f.sigmoid(reset_gate_spatial)

        reset_hidden = self.reset_hidden((reset_gate * prev_hidden_spatial) + (reset_gate*hidden_state_temporal))
        #reset_hidden = self.reset_hidden((reset_gate_spatial * prev_hidden_spatial) + (reset_gate_temporal *hidden_state_temporal))
 
        # apply tanh non linearity
        current_memory = f.tanh(current+reset_hidden)

        hidden = (update_gate * prev_hidden_spatial) + ((1-update_gate) * current_memory)

        del update_gate, reset_gate, prev_hidden_spatial, hidden_state_temporal, gates, stacked_inputs
        #del update_gate, reset_gate_temporal, reset_gate_spatial, prev_hidden_spatial, hidden_state_temporal, gates, stacked_inputs

        state = [hidden]

        return state
