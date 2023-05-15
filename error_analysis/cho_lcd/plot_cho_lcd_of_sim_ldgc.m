dnn_type='fbp';

if (strcmpi(dnn_type, 'fbp'))
    if ~exist('results/plots/fbp', 'dir')
	    mkdir('results/plots/fbp')
    end
	% ---------------------------------------------------------------------------------------------------------------------%
 	disp('fbp sharp plots');
	% ---------------------------------------------------------------------------------------------------------------------%
	y_label_auc = 'Detectability(AUC)';
	y_label_snr = 'Detectability(SNR)';

	% 3mm 14HU
	title_str ='Insert 3mm & 14HU';
	load('results/matfiles/fbp/_idx_1.mat');
    output_plot_name_auc='results/plots/fbp/_idx_1_auc.png';
	auc_or_snr_plot_denoising(auc_all, output_plot_name_auc, y_label_auc, title_str);

    title_str ='Insert 5mm & 7HU';
	load('results/matfiles/fbp/_idx_2.mat')
    output_plot_name_auc='results/plots/fbp/_idx_2_auc.png';
    auc_or_snr_plot_denoising(auc_all, output_plot_name_auc, y_label_auc, title_str);
    
    title_str ='Insert 7mm & 5HU';
    load('results/matfiles/fbp/_idx_3.mat')
    output_plot_name_auc='results/plots/fbp/_idx_3_auc.png';
	auc_or_snr_plot_denoising(auc_all, output_plot_name_auc, y_label_auc, title_str);

    title_str ='Insert 10mm & 3HU';
    load('results/matfiles/fbp/_idx_4.mat')
	output_plot_name_auc='results/plots/fbp/_idx_4_auc.png';
	auc_or_snr_plot_denoising(auc_all, output_plot_name_auc, y_label_auc, title_str);
end

function [] = auc_or_snr_plot_denoising(value_all, output_plot_name, ylabel_str, title_string)
    tick_vec = [0, 1, 2, 3, 4, 5]
    mark_list = {'.-k','^-b','o-r','.--k','^--b','o--r'};
	figure, hold on;
	valuemean = squeeze(mean(value_all));
	valuestd = squeeze(std(value_all));
    errorbar(1:4, valuemean, valuestd,mark_list{1})
	xlim([0.8,4.2]);
    title(title_string);
	h=legend('FBPsharp', 'Location', 'southeast');
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