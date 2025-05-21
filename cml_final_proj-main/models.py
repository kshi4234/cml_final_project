import torch
import torch.nn as nn

class Our_Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.k_size = kwargs['k_size']
        self.k_stride = kwargs['k_stride']
        self.k_pad = kwargs['k_pad']
        self.p_size = kwargs['p_size']
        self.p_stride = kwargs['p_stride']
        self.p_pad = kwargs['p_pad']
        
        self.cnn_block = torch.nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=self.k_size, stride=self.k_stride, padding=self.k_pad),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.p_size, stride=self.p_stride, padding=self.p_pad),
            nn.Conv2d(32, 64, kernel_size=self.k_size, stride=self.k_stride, padding=self.k_pad),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.p_size, stride=self.p_stride, padding=self.p_pad),
            nn.Conv2d(64, 128, kernel_size=self.k_size, stride=self.k_stride, padding=self.k_pad),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.flat_dim = 128 * self.calc_dim(kwargs)  # Multiplies num_channels and 2d_params to get total parameters for a single sample
        
        self.ffn_block = torch.nn.Sequential(
            nn.Linear(self.flat_dim, 1)
        )
    
    def forward(self, x):
        cnn_out = self.cnn_block(x)
        flattened = cnn_out.view(-1, self.flat_dim)
        lin_out = self.ffn_block(flattened)
        # print("RAW PRED: ", lin_out)
        pred = nn.functional.sigmoid(lin_out)
        return pred.view(-1)

    def calc_dim(self, kwargs):
        # Can literally just pass a 1x1x224x244 tensor through and check final dimensions
        H, W = kwargs['i_size'], kwargs['i_size']
        H, W = (H + 2*kwargs['k_pad'] - kwargs['k_size']) / kwargs['k_stride'] + 1, (W + 2*kwargs['k_pad'] - kwargs['k_size']) / kwargs['k_stride'] + 1
        H, W = (H + 2*kwargs['p_pad'] - kwargs['p_size']) / kwargs['p_stride'] + 1, (W + 2*kwargs['p_pad'] - kwargs['p_size']) / kwargs['p_stride'] + 1
        H, W = (H + 2*kwargs['k_pad'] - kwargs['k_size']) / kwargs['k_stride'] + 1, (W + 2*kwargs['k_pad'] - kwargs['k_size']) / kwargs['k_stride'] + 1
        H, W = (H + 2*kwargs['p_pad'] - kwargs['p_size']) / kwargs['p_stride'] + 1, (W + 2*kwargs['p_pad'] - kwargs['p_size']) / kwargs['p_stride'] + 1
        H, W = (H + 2*kwargs['k_pad'] - kwargs['k_size']) / kwargs['k_stride'] + 1, (W + 2*kwargs['k_pad'] - kwargs['k_size']) / kwargs['k_stride'] + 1
        # dummy = torch.rand([1, 1, 224, 224], dtype=torch.float32).to(device)
        # # print(dummy.shape)
        # dummy = self.cnn_block(dummy)
        # params_2d = dummy.shape[2] * dummy.shape[3]
        return int(H)*int(W)