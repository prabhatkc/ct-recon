clc; clear all;
measure_option = 'fbp_sharp';

if (strcmpi(measure_option, 'fbp_sharp'))
% ---------------------------------------------------------------------------------------------------------------------%
 	disp('LCD-CHO on simulated fbp-sharp CT scans acquisition aligned to the LDGC acquisition');
% ---------------------------------------------------------------------------------------------------------------------%
	data_folder      ='../../irt/digiNoise/results/mita';
	proc_data_folder = '';
	chkpt_string     = '';
	output_fname     ='results/matfiles/fbp';
	if ~exist(output_fname, 'dir')
		mkdir(output_fname)
	end
	mita_lcd_4r_ldgc(data_folder, proc_data_folder, chkpt_string, 1 , output_fname);
	mita_lcd_4r_ldgc(data_folder, proc_data_folder, chkpt_string, 2 , output_fname);
	mita_lcd_4r_ldgc(data_folder, proc_data_folder, chkpt_string, 3 , output_fname);
	mita_lcd_4r_ldgc(data_folder, proc_data_folder, chkpt_string, 4 , output_fname);
    plot_cho_lcd_of_sim_ldgc;
elseif (strcmpi(measure_option, 'pre-calc-bilateral'))
% ---------------------------------------------------------------------------------------------------------------------%
 	disp('LCD-CHO on simulated fbp-sharp CT scans acquisition aligned to the LDGC acquisition,');
    disp('& the CT-scans obtained by using bilateral denoising on the fbp-sharp images.');
    disp('plot generated from bilateral filtering is from pre-caluated bilateral filter applied');
    disp('to CCT-189 scans. If you want to apply bilateral filtering to your current CCT189 scans');
    disp('look in https://github.com/prabhatkc/ct-recon/tree/main/Denoising/BLTdenoise.')
% ---------------------------------------------------------------------------------------------------------------------%
	if ~exist('results/plots/blf', 'dir')
	    mkdir('results/plots/blf')
    end
    y_label_auc = 'Detectability(AUC)';
	y_label_snr = 'Detectability(SNR)';
    
	% 3mm 14HU
	title_str ='Insert 3mm & 14HU';
	load('results/matfiles/blf/_idx_1.mat');
    output_plot_name_auc='results/plots/blf/_idx_1_auc.png';
	auc_or_snr_plot_denoising(auc_all, output_plot_name_auc, y_label_auc, title_str, 2);
    output_plot_name_snr='results/plots/blf/_idx_1_snr.png';
	auc_or_snr_plot_denoising(snr_all, output_plot_name_snr, y_label_snr, title_str, 2);

    title_str ='Insert 5mm & 7HU';
	load('results/matfiles/blf/_idx_2.mat')
    output_plot_name_auc='results/plots/blf/_idx_2_auc.png';
    auc_or_snr_plot_denoising(auc_all, output_plot_name_auc, y_label_auc, title_str, 2);
    output_plot_name_snr='results/plots/blf/_idx_2_snr.png';
	auc_or_snr_plot_denoising(snr_all, output_plot_name_snr, y_label_snr, title_str, 2);
    
    title_str ='Insert 7mm & 5HU';
    load('results/matfiles/blf/_idx_3.mat')
    output_plot_name_auc='results/plots/blf/_idx_3_auc.png';
	auc_or_snr_plot_denoising(auc_all, output_plot_name_auc, y_label_auc, title_str, 2);
    output_plot_name_snr='results/plots/blf/_idx_3_snr.png';
	auc_or_snr_plot_denoising(snr_all, output_plot_name_snr, y_label_snr, title_str, 2);

    title_str ='Insert 10mm & 3HU';
    load('results/matfiles/blf/_idx_4.mat')
	output_plot_name_auc='results/plots/blf/_idx_4_auc.png';
	auc_or_snr_plot_denoising(auc_all, output_plot_name_auc, y_label_auc, title_str, 2);
    output_plot_name_snr='results/plots/blf/_idx_4_snr.png';
	auc_or_snr_plot_denoising(snr_all, output_plot_name_snr, y_label_snr, title_str, 2);
end


function [] = auc_or_snr_plot_denoising(value_all, output_plot_name, ylabel_str, title_string, denoising_num)
    tick_vec = [0, 1, 2, 3, 4, 5];
    mark_list = {'.-k','^-b','o-r','.--k','^--b','o--r'};
	figure, hold on;
	valuemean = squeeze(mean(value_all));
	valuestd = squeeze(std(value_all));
    if denoising_num==2
        for i=1:2 
		% here i=2 represent the recon type i.e. first make fbpsharp with fbp_valuemean +- fbp_std; then second line
		% plot in same figure of cnn_valuemean +- cnn_std
	        errorbar(1:4, valuemean(i,:), valuestd(i,:),mark_list{i});
        end
        h=legend('FBPsharp','bilateral', 'Location', 'southeast');
    else
        errorbar(1:4, valuemean, valuestd,mark_list{1})
        h=legend('FBPsharp', 'Location', 'southeast');
    end
	xlim([0.8,4.2]);
    title(title_string);
	set(h, 'Fontsize', 14);
	h=xlabel('dose levels');
	set(h, 'Fontsize', 14);
	h=ylabel(ylabel_str);
	set(h, 'Fontsize', 14);
	set(gca,'XTick',[tick_vec]);
	set(gca, 'xticklabel', {'', '25%', '50%', '75%', '100%', ''});
	set(gca, 'Fontsize', 14); % tick font size
	box on; grid on; grid minor;
	hold off;
	print('-dpng', output_plot_name);
    close 
end