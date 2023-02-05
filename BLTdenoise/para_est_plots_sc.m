%% header is
%Folder-Name, nImgs, dns rMSE, (+,-std), dns PSNR [dB], (+,-std), dns SSIM,
%(+,-std), LD rMSE, (+,-std), LD PSNR [dB], (+,-std), LD SSIM, (+,-std)
lookUpfiles    = '/ss-5.0-ws-*-sc-0.02*.txt';%'/ss-5.0-ws-7-sc-*.txt';%'/ss-*-ws-7-sc-0.01*.txt';
commonfile_str = 'ss-5-sc-0.02';%'ws-7-sc-0.01';
var_para       = 'ws';
folder_str     = ['/home/prabhat.kc/Implementations/lowdosect/denoise_opt/bilateral/results/bilateral/para_est/poisson/noisy_denoised/', var_para];
 %{'ss', 'ws', 'sc'};

if strcmp(var_para, 'ss')
    var_para_ind =2;
    outfile_str = sprintf(['ss_vary-', commonfile_str]);
    xlab_str = 'sigma-spatial';
elseif strcmp(var_para, 'ws')
    var_para_ind=4;
    outfile_str = 'ws_vary-';
    xlab_str = 'win-size';
    xticklabel_str ="{sprintf(''), sprintf(para_vals{1}), sprintf(para_vals{2}), " + ...
        "sprintf(para_vals{3}), sprintf(para_vals{4}), sprintf(para_vals{5})," + ...
        "sprintf(para_vals{6}), sprintf(para_vals{7}), sprintf('')}";

elseif strcmp(var_para, 'sc')
    var_para_ind =6;
    outfile_str = 'sc_vary-';
    xlab_str = 'sigma-color';
    xticklabel_str ="{sprintf(''), sprintf(para_vals{1}), sprintf(para_vals{2}), " + ...
        "sprintf(para_vals{3}), sprintf(para_vals{4}), sprintf(para_vals{5})," + ...
        "sprintf(para_vals{6}), sprintf(para_vals{7}), sprintf(para_vals{8}), sprintf('')}";
end

gen_file_str = dir([folder_str, lookUpfiles]);
Nfiles       = length(gen_file_str);
dn_rmse_arr  = zeros(Nfiles, 1);
dn_srmse_arr = zeros(Nfiles, 1);
ld_rmse_arr  = zeros(Nfiles, 1);
ld_srmse_arr = zeros(Nfiles, 1);

dn_ssim_arr  = zeros(Nfiles, 1);
dn_sssim_arr = zeros(Nfiles, 1);
ld_ssim_arr  = zeros(Nfiles, 1);
ld_sssim_arr = zeros(Nfiles, 1);

dn_psnr_arr  = zeros(Nfiles, 1);
dn_spsnr_arr = zeros(Nfiles, 1);
ld_psnr_arr  = zeros(Nfiles, 1);
ld_spsnr_arr = zeros(Nfiles, 1);

para_vals ={};
for i=1:Nfiles
    fid = fopen([folder_str, '/', gen_file_str(i).name]);
    instr=textscan(fid, ['%11s %6d %9.4f %9.4f %14.4f %9.4f %9.4f ' ...
        '%9.4f %8.4f %9.4f %13.4f %9.4f %8.4f %9.4f'], 'Delimiter', ',' , 'HeaderLines', 1);
    fclose(fid);
    dn_rmse_arr(i)  = instr{3}*100;
    dn_srmse_arr(i) = instr{4}*100;
    ld_rmse_arr(i)  = instr{9}*100;
    ld_srmse_arr(i) = instr{10}*100;

    dn_ssim_arr(i)  = instr{7};
    dn_sssim_arr(i) = instr{8};
    ld_ssim_arr(i)  = instr{13};
    ld_sssim_arr(i) = instr{14};

    dn_psnr_arr(i)  = instr{5};
    dn_spsnr_arr(i) = instr{6};
    ld_psnr_arr(i)  = instr{11};
    ld_spsnr_arr(i) = instr{12};

    fname_str = split(gen_file_str(i).name, '-');
    para_vals{i}=fname_str{var_para_ind};
end

%arrange in ascending order
[dummy,ac_order]=sort(str2double(para_vals));
dn_rmse_arr  = dn_rmse_arr(ac_order);
dn_srmse_arr = dn_srmse_arr(ac_order);
ld_rmse_arr  = ld_rmse_arr(ac_order);
ld_srmse_arr = ld_srmse_arr(ac_order);

dn_ssim_arr  = dn_ssim_arr(ac_order);
dn_sssim_arr = dn_sssim_arr(ac_order);
ld_ssim_arr  = ld_ssim_arr(ac_order);
ld_sssim_arr = ld_sssim_arr(ac_order);

dn_psnr_arr  = dn_psnr_arr(ac_order);
dn_spsnr_arr = dn_spsnr_arr(ac_order);
ld_psnr_arr  = ld_psnr_arr(ac_order);
ld_spsnr_arr = ld_spsnr_arr(ac_order);

para_vals    = para_vals(ac_order);

metric = 'RMSE';
outfile_name = [folder_str, '/', metric,'-', outfile_str, '.esp'];
error_bar_plot(1:Nfiles, ld_rmse_arr, ld_srmse_arr, dn_rmse_arr, dn_srmse_arr, metric, xlab_str,para_vals, xticklabel_str, outfile_name)

metric = 'PSNR';
outfile_name = [folder_str, '/', metric,'-', outfile_str, '.esp'];
error_bar_plot(1:Nfiles, ld_psnr_arr, ld_spsnr_arr, dn_psnr_arr, dn_spsnr_arr, metric, xlab_str,para_vals, xticklabel_str, outfile_name)

metric = 'SSIM';
outfile_name = [folder_str, '/', metric,'-', outfile_str, '.esp'];
error_bar_plot(1:Nfiles, ld_ssim_arr, ld_sssim_arr, dn_ssim_arr, dn_sssim_arr, metric, xlab_str, para_vals, xticklabel_str, outfile_name)

function [] = error_bar_plot(x_vals, ld_vals, ld_err, dn_vals, dn_err, metric, xlab_str, para_vals, xticklabel_str, outfile_name)
    ylim_max = max(max(ld_vals+ld_err), max(dn_vals+dn_err));
    ylim_min = min(min(ld_vals-ld_err), min(dn_vals-dn_err));
    if strcmp(metric, 'RMSE')
        ylim_min = ylim_min*0.5;
        ylim_max = ylim_max*1.5;
        ylab = [metric, '% (std)'];
    elseif strcmp(metric, 'SSIM')
        ylim_min =ylim_min - 0.03;
        ylim_max = ylim_max + 0.03;
        ylab = [metric, ' (std)'];
    else
        ylim_min = ylim_min - 2;
        ylim_max = ylim_max + 2;
        ylab = [metric, ' (std)'];
    end
    figure, hold on;
    ylim([ylim_min ylim_max]);
    errorbar(x_vals, ld_vals, ld_err,'o', 'Color','k');
    errorbar(x_vals, dn_vals, dn_err, 's', 'Color','b');
    hold off;

    legend('FBP-LD','denoised');
    set(gca, 'Fontsize', 10);
    xlim([0.0 length(x_vals)+1]);
    h=xlabel(sprintf(xlab_str));
    set(h, 'Fontsize', 14);
    h=ylabel(ylab);
    set(h, 'Fontsize', 14);
    %set(gca,'XTick',[35 120 340 990]);
    para_vals
    set(gca,'xticklabel', eval(xticklabel_str));
        %{sprintf(''), sprintf(para_vals{1}), sprintf(para_vals{2}), ...
        %sprintf(para_vals{3}), sprintf(para_vals{4}), sprintf(para_vals{5}), ...
        %sprintf(para_vals{6}), sprintf(para_vals{7}), sprintf(para_vals{8}), sprintf('')});
    box on;
    grid on;
    grid minor;
    print('-depsc', sprintf(outfile_name));
close;
end 