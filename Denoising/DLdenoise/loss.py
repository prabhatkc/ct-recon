import torch
from torch import nn
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM
import sys

class combinedLoss(nn.Module):
    def __init__(self, model_args, reg_lambda=1):
        super(combinedLoss, self).__init__()
        
        self.loss_func = model_args.loss_func
        self.prior_type = model_args.prior_type
        self.reg_lambda = reg_lambda
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.prior_type is not None:
            self.prior_term = prior_term(self.reg_lambda, self.prior_type)
        
        if (self.loss_func == 'mse'):
            self.likelihood_term = nn.MSELoss()
        if (self.loss_func == 'l1'):
            self.likelihood_term = nn.L1Loss()
        if (self.loss_func == 'ce'):
            self.likelihood_term = nn.CrossEntropyLoss()
        if (self.loss_func == 'ms-ssim'):
            # channle is 1 as we have only black colour for dicom CT images
            # size average yields averages all the ms_ssim values obainted for each batch as is required to deteimine
            # the loss
            # ((win_size - 1) * (2 ** 4)) < h, w of the patches
            self.likelihood_term = MS_SSIM(data_range=1.0, size_average=True, channel=1, win_size=5).to(self.device)

    def forward(self, out_images, target_images):
        if (self.loss_func == 'ms-ssim'):
            #print('prior type is:', self.prior_type)
            #sys.exit()
            # (alpha)*(1-MS_SSIM(X, Y))+ (1-alpha)*|X-Y|
            obj_loss = 0.84*(1-self.likelihood_term(out_images, target_images)) + (1-0.84)*self.prior_term(out_images-target_images)
        else:
            if self.prior_type is not None:
                obj_loss = self.likelihood_term(out_images, target_images) + self.prior_term(out_images)
            else:
                obj_loss = self.likelihood_term(out_images, target_images)
        return obj_loss

    def __str__(self):
        if (self.loss_func == 'mse'):     
            string1 = ' Likelihood term is Mean Squared Error'
        elif (self.loss_func == 'l1'):
            string1 = ' Likelihood term is L1 norm'
        elif (self.loss_func == 'ms-ssim'):
            string1 = ' Likelihood term is (1-MS_SSIM)'
        else: #self.loss_func is 'ce':   
            string1 = ' Likelihood term is Cross-Entropy'

        if (self.prior_type == 'l1'):
            str2 = (' & Prior term is L1 norm')
        elif (self.prior_type == 'nl'):     
            str2 = (' & Prior term is Non-Local means')
        elif (self.prior_type == 'sobel'):  
            str2 = (' & Prior term is a sobel kernel')
        elif (self.prior_type == 'tv-fd'):  
            str2 = (' & Prior term is Total-Variation with Forward difference')
        elif (self.prior_type == 'tv-fbd'):
            str2 = (' & Prior term is Total-Variation with Forward-backward differences')
        else:
            str2= (' & Prior term is not used')
        return(string1 + str2)

class prior_term(nn.Module):
    def __init__(self, reg_lambda, prior_type):
        super(prior_term, self).__init__()
        self.reg_lambda = reg_lambda
        self.prior_type = prior_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        batch_size, c, h_x, w_x = x.size()
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        
        # lasso
        if self.prior_type == 'l1':
            return self.reg_lambda * (torch.abs(x.view([batch_size, c, h_x, w_x]))).sum()
        
        # Total-Variation Forward Difference 
        if self.prior_type == 'tv-fd':
            h_tv = torch.abs((x[:, :, 1:, :] - x[:, :, :h_x - 1, :])).sum()
            w_tv = torch.abs((x[:, :, :, 1:] - x[:, :, :, :w_x - 1])).sum()
            
            # here the division by count_h or count_w is to account for the fact that reduced mean is applied by default
            # to each pixel 
            # factor 2 is to account for the fact that reduced mean is not applied to the same point due to horizontal
            # and verticle difference
            # prefactor of 0.5 to h_tv and w_tv is to account for the fact that the convex hull of regularization kernal
            # remain 1 for each pixel

            return self.reg_lambda * 2 * (0.5*h_tv / count_h + 0.5*w_tv / count_w) / batch_size
        
        # Total-Variation Forward-Backward Difference 
        if self.prior_type == 'tv-fbd':
            #y-direction forward difference
            h_tv = torch.abs((x[:, :, 1:, :] - x[:, :, :- 1, :]))
            h_tv = (h_tv[:, :, 1:, :]).sum() # cleaning first row
            
            #y direction backward difference
            hb_tv = torch.abs((x[:, :, :-1, :] - x[:, :, 1:, :]))
            hb_tv = (hb_tv[:, :, :-1, :]).sum()

            w_tv = torch.abs((x[:, :, :, 1:] - x[:, :, :, :- 1]))
            w_tv = (w_tv[:, :, :, 1:]).sum()
            
            wb_tv = torch.abs((x[:, :, :, :-1] - x[:, :, :, 1:]))
            wb_tv = (wb_tv[:, :, :, :-1]).sum()

            return self.reg_lambda * 2 * (0.25*(h_tv+hb_tv)/(count_h-1) + 0.25*(w_tv+wb_tv) / (count_w-1)) / batch_size
        
        # using sobel kernel
        if self.prior_type == 'sobel':
            # along y direction 
            sy_filt = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float, device=self.device).view([1, 1, 3, 3])
            conv_ry = F.conv2d(x.view([-1, 1, h_x, w_x]), sy_filt, stride=1, padding=1)
            conv_ry = torch.abs(conv_ry.view([batch_size, c, h_x, w_x]))
            conv_ry = (conv_ry[:, :, 1:-1, 1:-1]).sum()

            sx_filt = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float, device=self.device).view([1, 1, 3, 3])
            conv_rx = F.conv2d(x.view([-1, 1, h_x, w_x]), sx_filt, stride=1, padding=1)
            conv_rx = torch.abs(conv_rx.view([batch_size, c, h_x, w_x]))
            conv_rx = (conv_rx[:, :, 1:-1, 1:-1]).sum()
            return self.reg_lambda * 2 * (0.1667*(conv_ry)/(count_h-1) + 0.1667*(conv_rx) / (count_w-1)) / batch_size
            #return self.tv_reg_lambda * 0.1667*convr/((count_h+1)*batch_size)

        # non-local kernel
        if self.prior_type == 'nl':
            nl_filt = torch.tensor([[-1/12, -1/6, -1/12], [-1/6, 0, -1/6], [-1/12, -1/6, -1/12]], dtype=torch.float, device=self.device).view([1, 1, 3, 3])
            conv_r = F.conv2d(x.view([-1, 1, h_x, w_x]), nl_filt, stride=1, padding=1)
            conv_r = torch.abs(conv_r.view([batch_size, c, h_x, w_x]))
            conv_r = (conv_r[:, :, 1:-1, 1:-1]).sum() # chucking off boundry values that may skew the results
            return self.reg_lambda * conv_r/((count_h-1)*batch_size)

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
