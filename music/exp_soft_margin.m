function exp_soft_margin(exp_setting)
% Experiment with music data
% sontran 2013
eval(exp_setting);

%% load data
vars = whos('-file', feature_file);
A = load(feature_file,vars(1).name,vars(2).name,vars(3).name,vars(4).name);
raw_features = A.(vars(1).name);
indices      = A.(vars(2).name);
tst_inx      = A.(vars(3).name);
trn_inx      = A.(vars(4).name);

g_features   = raw_features(:,G_INX);
b_features   = raw_features(:,B_INX); %%
r_features   = [g_features b_features];
% Visualize modalities
%if ~isempty(g_features), figure(1); colormap(hot);imagesc(g_features); colorbar; end
%if ~isempty(b_features), figure(2); colormap(hot);imagesc(b_features); colorbar; end

log_file = strcat(exp_dir,'SMARGIN',lm,'exp.mat');
num_case = size(trn_inx,1); 
for cxx = [0.01:0.02:0.5]
    
cr_ = 0;
cr  = 0;
tic;
for f=1:num_case      
    %extract features    
    mmrbb_features = r_features;

    w = SUB_SIZE;
    [trnd_12 trnd_13] = subspace_distances(trn_inx,mmrbb_features,indices,w,0);
    [tstd_12 tstd_13] = subspace_distances(tst_inx,mmrbb_features,indices,w,0);
    % train similarity model        
    [W] = learn_soft_margin(trnd_12{f},trnd_13{f},cxx,1000,0.0001)';   
    cr_ = cr_ + sum((trnd_12{f}-trnd_13{f})*W' < 0, 1)/size(trnd_12{f},1);
    cr = cr + sum((tstd_12{f}-tstd_13{f})*W' < 0, 1)/size(tstd_12{f},1); %'    

end
cr_ = cr_/num_case;
cr  = cr/num_case;
fprintf('Accuracy (Training|Testing) %f|%f\n',cr_,cr);
logging(log_file,[cxx cr_ cr]);
toc;
end
end