import utils
import torch 
import os
import torch.optim as optim
from loss import combinedLoss
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt

def opt_solver(args):
	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	torch.manual_seed(42)
	print('Command line arguements')
	print('\n---------------------------------------------------------------------------------')
	print('----------------------------------------------------------------------------------')
	for i in args.__dict__: print((i),':',args.__dict__[i])
	print('\n--------------------------------------------------------------------------------\n')


	#-----------------------------------
	# declare r/w paths
	#-----------------------------------
	in_paths      = utils.getimages4rmdir(os.path.join(args.input_folder+'/'+ args.input_gen_folder))
	folder_str    = (args.input_folder).split('/')[-1]+'/'+args.input_gen_folder
	output_folder = os.path.join((args.output_folder)+'/'+(args.input_folder).split('/')[-1]+'/'+args.input_gen_folder+'_denoised/'+ args.loss_func + '-'+args.prior_type+ '-lr-'+str(args.lr)+'-reg-lambda-'+str(args.reg_lambda))
	if not os.path.isdir(output_folder): os.makedirs(output_folder, exist_ok=True)
	quant_fname = os.path.join(output_folder+'-quant-vals.txt')
	gt_available = bool((args.target_gen_folder).strip())
	if gt_available:
		target_paths  = utils.getimages4rmdir(os.path.join(args.input_folder+'/'+ args.target_gen_folder))
		lr_rMSE_arr, lr_psnr_arr, lr_ssim_arr    = [], [], []
		opt_rMSE_arr, opt_psnr_arr, opt_ssim_arr = [], [], []
		quantfile = open(quant_fname, '+w')	
		quantfile.write('Folder-Name, nImgs, opt rMSE, (+,-std), opt PSNR [dB], (+,-std), opt SSIM, (+,-std), LD rMSE, (+,-std), LD PSNR [dB], (+,-std), LD SSIM, (+,-std)\n')

	# ---------------------------------------------------
	# processing each image in the input folder 
	# - ------------------------------------------------
	for i in range(len(in_paths)):
		if args.input_img_type=='dicom':
		  lr_img = utils.pydicom_imread(in_paths[i])
		  if gt_available: gt_img = utils.pydicom_imread(target_paths[i])
		elif args.input_img_type=='raw':
		  lr_img = utils.raw_imread(in_paths[i], (256, 256))
		else:
		  lr_img = utils.imageio_imread(in_paths[i])
		  if gt_available: gt_img = utils.imageio_imread(target_paths[i])

		h, w      = lr_img.shape
		nlr_img   = utils.normalize_data_ab(0.0, 1.0, lr_img.astype('float32'))
		ngt_img   = utils.normalize_data_ab(0.0, 1.0, gt_img.astype('float32')) if gt_available else None
		
		# torch solver
		opt_sol = variational(args, nlr_img, normalized_gt=ngt_img)
		if gt_available: diff_img = ngt_img - opt_sol
		lr_img  = lr_img.astype(args.out_dtype)

		# Renormalize
		opt_sol = utils.normalize_data_ab(np.min(lr_img), np.max(lr_img), opt_sol)
		opt_sol = opt_sol.astype(args.out_dtype)
		#utils.plot2dlayers(lr_img)
		#utils.plot2dlayers(opt_sol)
		img_str=in_paths[i]
		img_str= img_str.split('/')[-1]
		#img_no = img_str.split('.')[0]
		img_no = img_str.split('.')[-2]

		#error analysis for each image do this
		if gt_available:
			gt_img           = gt_img.astype(args.out_dtype)
			opt_max, opt_min = max(np.max(gt_img), np.max(opt_sol)), min(np.min(gt_img), np.min(opt_sol))
			opt_rMSE         = utils.relative_mse(gt_img, opt_sol)
			opt_psnr         = utils.psnr(gt_img, opt_sol, opt_max)
			opt_ssim         = compare_ssim(opt_sol.reshape(h, w, 1), gt_img.reshape(h, w, 1), multichannel=True, data_range=(opt_max-opt_min))
			lr_max, lr_min   = max(np.max(gt_img), np.max(lr_img)), min(np.min(gt_img), np.min(lr_img))
			lr_rMSE          = utils.relative_mse(gt_img, lr_img)
			lr_psnr          = utils.psnr(gt_img, lr_img, lr_max)
			lr_ssim          = compare_ssim(lr_img.reshape(h, w, 1), gt_img.reshape(h, w, 1), multichannel=True, data_range=(lr_max-lr_min))
			# append errors for each image
			lr_rMSE_arr.append(lr_rMSE); lr_psnr_arr.append(lr_psnr); lr_ssim_arr.append(lr_ssim)
			opt_rMSE_arr.append(opt_rMSE); opt_psnr_arr.append(opt_psnr); opt_ssim_arr.append(opt_ssim)
			print("IMG: %s || avg opt [rMSE: %.4f, PSNR: %.4f, SSIM: %.4f] || avg LD [rMSE: %.4f, PSNR: %.4f, SSIM: %.4f]"\
			%(img_str, opt_rMSE, opt_psnr, opt_ssim, lr_rMSE, lr_psnr, lr_ssim))
		else:
			print('img no', i)
			print('input min/max', np.min(lr_img), np.max(lr_img), lr_img.dtype)
			print('solution min/max', np.min(opt_sol), np.max(opt_sol), opt_sol.dtype)


		# ==========================================================		    
		# saving denoised images if save denoised images is true
		# ==========================================================
		if (args.save_imgs ==True):	    	
			utils.imsave_raw((opt_sol), output_folder + '/tv_' + img_no+ '.raw') if (args.input_img_type=='raw') else \
			utils.imsave((opt_sol), output_folder + '/tv_'  + img_no + '.tif', type='original')
			if gt_available:
				if not os.path.isdir(output_folder+'_diff'): os.makedirs(output_folder+'_diff', exist_ok=True)
				utils.imsave((utils.normalize_data_ab(0.0, 255.0, diff_img)).astype('uint8'), output_folder + '_diff/tv_'  + img_no + '.tif', type='original')
	print("\ndenoised solutions are stored in folder:", output_folder)
	print('')

	# print avaraged error for all image in a given folder
	if gt_available:
		print('\n----------------')
		print('folder summary')
		print('-----------------')
		print("Folder denoised: %s \navg opt (std of imgs) [rMSE: %.4f (%.4f), PSNR: %.4f (%.4f), SSIM: %.4f (%.4f)] \navg LD (std of imgs)  [rMSE: %.4f (%.4f), PSNR: %.4f (%.4f), SSIM: %.4f (%.4f)]" % \
		(folder_str, np.mean(opt_rMSE_arr), np.std(opt_rMSE_arr), np.mean(opt_psnr_arr), np.std(opt_psnr_arr), np.mean(opt_ssim_arr), np.std(opt_ssim_arr), \
					np.mean(lr_rMSE_arr), np.std(lr_rMSE_arr), np.mean(lr_psnr_arr), np.std(lr_psnr_arr), np.mean(lr_ssim_arr), np.std(lr_ssim_arr)))
		quantfile.write("%11s,%6d,%9.4f,%9.4f,%14.4f,%9.4f,%9.4f,%9.4f,%8.4f,%9.4f,%13.4f,%9.4f,%8.4f,%9.4f\n" % \
		((args.input_folder).split('/')[-1],len(in_paths), np.mean(opt_rMSE_arr), np.std(opt_rMSE_arr), np.mean(opt_psnr_arr), np.std(opt_psnr_arr), np.mean(opt_ssim_arr), np.std(opt_ssim_arr), \
					np.mean(lr_rMSE_arr), np.std(lr_rMSE_arr), np.mean(lr_psnr_arr), np.std(lr_psnr_arr), np.mean(lr_ssim_arr), np.std(lr_ssim_arr)))
		
		quantfile.close()


def variational(args, normalized_input, normalized_gt=None):
	# Defines an optimizer to update the parameters
	h, w      = normalized_input.shape
	nlr_timg  = torch.tensor(normalized_input, dtype=torch.float32, device=args.device, requires_grad=False)
	nlr_timg  = nlr_timg.view(1, 1, h, w)
	opt_sol   = torch.tensor(normalized_input.reshape(1, 1, h, w), dtype=torch.float32, device=args.device, requires_grad=True)
	optimizer = optim.Adam([opt_sol], lr = args.lr)
	reg_loss  = combinedLoss(args, reg_lambda=args.reg_lambda)
	loss_arr  = []
	prev_loss = 10
	loss_tol  = 1e-6
	ite 	  = 0
	loss_diff = 1
	if args.print_opt_errs:
		print('\n=============================================================')
		print("ite No,        loss,        rMSE,        SSIM,     rel_err")
		print('=============================================================')

	while ((loss_diff>loss_tol)):
		if ite == args.nite:
			break
		sol_old   = utils.tensor_2_img(opt_sol)
		loss_fn   = reg_loss(opt_sol, nlr_timg) 
		loss_arr.append(loss_fn.cpu().detach())
		loss_fn.backward()
		# updates opt_solver
		optimizer.step()
		# free the gradients
		optimizer.zero_grad()
		opt_sol_np = utils.tensor_2_img(opt_sol)
		if (args.print_opt_errs and ite % 1 ==0):
			if ((normalized_gt is not None)):
				ite_max, ite_min = max(np.max(normalized_gt), np.max(opt_sol_np)), min(np.min(normalized_gt), np.min(opt_sol_np))
				ssim_per_ite = compare_ssim(opt_sol_np.reshape(h, w, 1), normalized_gt.reshape(h, w, 1), multichannel=True, data_range=(ite_max-ite_min))
				print("%6d,%12.7f,%12.5f,%12.5f,%12.5f" % (ite, loss_fn.cpu().detach().numpy(), utils.relative_mse(opt_sol_np, normalized_gt),ssim_per_ite, utils.relative_error(f_old=sol_old, f_new=opt_sol_np)))
			else:
				print("%6d,%12.7f,%12s,%12s,%12.5f" % (ite, loss_fn.cpu().detach().numpy(), "NA", "NA", utils.relative_error(f_old=sol_old, f_new=opt_sol_np)))
		ite+=1
		loss_diff = prev_loss  - loss_fn.cpu().detach().numpy()
		prev_loss = loss_fn.cpu().detach().numpy()
	return(opt_sol_np)