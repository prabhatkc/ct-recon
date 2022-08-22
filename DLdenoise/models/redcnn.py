import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class REDcnn10(nn.Module):
    

    def __init__(self, d=96, s=5, batch_normalization=False, idmaps=3, bias=True):
        """ Pytorch implementation of Redcnn following the paper [1]_, [2]_.
        Notes
        -----
        In [1]_, authors have suggested three architectures:
        a. RED10, which has 10 layers and does not use any skip connection (hence skip_step = 0)
        b. RED20, which has 20 layers and uses skip_step = 2
        c. RED30, which has 30 layers and uses skip_step = 2
        d. It also shows that kernel size 7x7 & 9X9 yields better performance than (5x5) or (3x3)
           However, it argues that using large kernel size may lead to poor optimum for high-level tasks.
        e. Moreover, using filter size 64, while the kernel size is (3, 3), it shows 100x100 patch yeilds
           the best denoised result
        In [2]_, where RedNet was used for CT denoising, authors have suggested:
        a. Red10 with 3 skip connections
        b. patch-size (55x55)
        c. kernel-size (5X5) & no. of filters(96)
        
        Parameters
        ----------
        depth (hard coded to be 10, following [2]_)
            Number of fully convolutional layers in dncnn. In the original paper, the authors have used depth=17
            for non-blind denoising (same noise level) and depth=20 for blind denoising (different noise level).
        d : int
            Number of filters at each convolutional layer.
        s : kernel size (int)
            kernel window used to compute activations.

        Returns
        -------
        :class:`torch.nn.Module`
            Torch Model representing the Denoiser neural network
        References
        ----------
        .. [1] Mao, X., Shen, C., & Yang, Y. B. (2016). Image restoration using convolutional auto-encoders 
            with symmetric skip connections. In Advances in neural information processing systems 
        .. [2] Chen, Hu, et al. "Low-dose CT with a residual encoder-decoder convolutional neural network". 
           IEEE transactions on medical imaging 36.12 (2017):
        """
        super(REDcnn10, self).__init__()
        if (s==9): pad = 4
        elif (s==5): pad = 2
        else: pad = 1
        self.batch_normalization=batch_normalization
        self.idmaps = idmaps
        
        ## Encoding layers ##
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=d, kernel_size=s, padding=pad, bias=bias)          
        self.conv2 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=s, padding=pad, bias=bias)
        self.conv3 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=s, padding=pad, bias=bias)
        self.conv4 = nn.Conv2d(in_channels=d, out_channels=d, kernel_size=s, padding=pad, bias=bias)
        ## Decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(in_channels=(d), out_channels=d, kernel_size=s, stride=1, padding=pad, bias=bias)  
        self.t_conv2 = nn.ConvTranspose2d(in_channels=d, out_channels=d, kernel_size=s, stride=1, padding=pad, bias=bias) 
        self.t_conv3 = nn.ConvTranspose2d(in_channels=d, out_channels=d, kernel_size=s, stride=1, padding=pad, bias=bias)
        self.t_conv4 = nn.ConvTranspose2d(in_channels=d, out_channels=d, kernel_size=s, stride=1, padding=pad, bias=bias)
        self.t_cout  = nn.ConvTranspose2d(in_channels=d, out_channels=1, kernel_size=s, stride=1, padding=pad, bias=bias)
        self.inbn    = nn.InstanceNorm2d(d, affine=True) #normalizes each batch independently
        self.sbn     = nn.BatchNorm2d(d)
        
        '''for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
        
        self.relu = nn.ReLU(inplace=True) #True:optimally does operation when inplace is true
        #self._initialize_weights()
        #self.relu = nn.LeakyReLU(0.2, inplace=True)
        '''

    def forward(self, x):       
        if (self.batch_normalization==True):
            ''' This BN is my addition to the existing model.
            However a proper initilization for conv weights
            and bn layer is still required
            '''
            ## encode ##
            xinit = x
            x     = F.relu(self.sbn(self.conv1(x)))
            x2    = x.clone()
            x     = F.relu(self.sbn(self.conv2(x)))
            x     = F.relu(self.sbn(self.conv3(x)))
            x4    = x.clone()
            x     = F.relu(self.sbn(self.conv4(x)))
            ## decode ##    
            x = F.relu(self.sbn(self.t_conv1(x))+x4)
            x = F.relu(self.sbn(self.t_conv2(x)))
            x = F.relu(self.sbn(self.t_conv3(x))+x2)
            x = F.relu(self.sbn(self.t_conv4(x)))
            x = (self.t_cout(x)+xinit)
        else:
            ## endcode #
            xinit = x
            x     = F.relu(self.conv1(x))
            x2    = x.clone()
            x     = F.relu(self.conv2(x))
            x     = F.relu(self.conv3(x))
            x4    = x.clone()
            x     = F.relu(self.conv4(x))
            ## decode ##
            if (self.idmaps==1):    
                x = F.relu(self.t_conv1(x))
                x = F.relu(self.t_conv2(x))
                x = F.relu(self.t_conv3(x))
                x = F.relu(self.t_conv4(x))
                x = F.relu(self.t_cout(x)+xinit)
            else:
                x = F.relu(self.t_conv1(x)+x4)
                x = F.relu(self.t_conv2(x))
                x = F.relu(self.t_conv3(x)+x2)
                x = F.relu(self.t_conv4(x))
                x = F.relu(self.t_cout(x)+xinit)
        return x