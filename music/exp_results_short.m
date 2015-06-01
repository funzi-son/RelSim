function exp_results_short()
SMG_RS = '/home/funzi/Documents/Experiments/Sparse.MM.Sim/MUSIC/SHORT_DAT/SMARGIN/';

RMM_RS = '/home/funzi/Documents/Experiments/Sparse.MM.Sim/MUSIC/SHORT_DAT/MM_SMARGIN/';

SPR_RS = '';

% soft-margin result
load(strcat(SMG_RS,'exp.mat'));


fprintf('Best result for SMG: %.5f\n',max(data(:,end)));
% multimodal RBM result
 fs = dir(strcat(RMM_RS,'exp_*.mat'));
 rs = [];
 for i=1:size(fs,1)
     load(strcat(RMM_RS,fs(i).name));
     %if isempty(rs)
         rs = [rs data(:,end)];
     %else         
     %end     
 end
 mmm = mean(rs'); 
 sdv = std(rs');
 [~,inx] = max(mmm);
 fprintf('Best result for RMM: %.5f+-%.5f\n',mmm(inx),sdv(inx));
% sparsity result 
end