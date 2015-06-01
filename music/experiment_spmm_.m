function experiment_spmm_(exp_setting)
% Experiment with music data, multimodal with sparsity
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
r_features   = raw_features(:,R_INX);
% Visualize modalities
%if ~isempty(g_features), figure(1); colormap(hot);imagesc(g_features); colorbar; end
%if ~isempty(b_features), figure(2); colormap(hot);imagesc(b_features); colorbar; end
for iii=1:TRIAL_NUM
str_date = datestr(now, 30);    
log_file = strcat(exp_dir,'SPMM_SMARGIN',lm,'exp_',str_date,'.mat'); % SP_SMARGIN is for balancing
num_case = size(trn_inx,1); 
for hidN = [50 100 200]
for lr  = [0.005 0.01 0.05 0.1 0.2] % Best [0.05 0.1 0.2]
    if hidN>100 && lr>0.05, continue; end
for ls  = [0.01 0.05 0.1 0.2 0.5]% 0.01 0.02 0.03]
for ps  = [0.001 0.01 0.1]%[0.0001 0.001 0.01 0.1]    
for cxx = [0.01:0.02:0.5]
    
conf.hidNum  = hidN;
conf.eNum    = 100;
conf.sNum    = 0;
conf.bNum    = 1;
conf.gNum    = 1;
conf.params  = [lr lr 0.01 0.00002];
conf.N       = 50;
conf.MAX_INC = 10;

conf.plot_  = 0;
conf.sigma = 1;    
conf.p = ps;
conf.lambda = ls;

conf.rel = 0;

cr_ = 0;
cr  = 0;
tic;
for f=1:num_case
    % run mmRBM on training set
    [~,f_inx] = ismember(unique(trn_inx{f}(:,1:3)),indices);    
%             [coeff,~] = princomp(g_features(f_inx,:));
%             g_features = g_features*coeff;        

    M = mean(g_features(f_inx,:));
    D = std(g_features(f_inx,:));
    g_features_ = (g_features-repmat(M,size(g_features,1),1))./repmat(D,size(g_features,1),1);            

    conf.sNum = size(f_inx,1);   
    disp(conf);        
    
    [Wg,Wb,~,hidB,ms,sigs] = train_spmm_rbm(conf,g_features_(f_inx,:),b_features(f_inx,:));            
    
    if all(all(Wg==0)) && all(all(Wb==0)), continue; end
    %extract features    
    mmrbb_features = logistic((g_features_./repmat(sigs.^2,size(g_features_,1),1))*Wg + b_features*Wb + repmat(hidB,size(g_features_,1),1));
    %mmrbb_features = [mmrbb_features > rand(size(mmrbb_features)) r_features];            
    %figure(3);imagesc(mmrbb_features); colorbar; ylabel('Samples'); xlabel('Features');        
    % Subspace is used?          
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
logging(log_file,[hidN lr ls ps conf.rel cxx cr_ cr]);
toc;
end
end
clc;
end
end
end
end
end