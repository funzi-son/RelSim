function experiment(exp_setting)
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
r_features   = raw_features(:,R_INX);
% Visualize modalities
%if ~isempty(g_features), figure(1); colormap(hot);imagesc(g_features); colorbar; end
%if ~isempty(b_features), figure(2); colormap(hot);imagesc(b_features); colorbar; end
for iii=1:TRIAL_NUM
    
    log_file = strcat(exp_dir,'exp_',num2str(conf.lambda),'_',num2str(iii),'.mat');
    %log_file = strcat(exp_dir,'exp_pca.mat');
% 10-fold cross validation
num_case = size(trn_inx,1); 
for hidN = [30 50 100]
for cxx = [0.01:0.02:0.5]    
cr_ = 0;
cr  = 0;


for f=1:num_case
    % run mmRBM on training set
    [~,f_inx] = ismember(unique(trn_inx{f}(:,1:3)),indices);
    if MMRBM_                  
%             [coeff,~] = princomp(g_features(f_inx,:));
%             g_features = g_features*coeff;        
        M = mean(g_features(f_inx,:));
        D = std(g_features(f_inx,:));
        g_features_ = (g_features-repmat(M,size(g_features,1),1))./repmat(D,size(g_features,1),1);            
        conf.sNum = size(f_inx,1);
        conf.hidNum = hidN;
        disp(conf);        
        [Wg,Wb,~,hidB,ms,sigs] = train_mm_rbm_(conf,g_features_(f_inx,:),b_features(f_inx,:));            
        %extract features    
        mmrbb_features = logistic((g_features_./repmat(sigs.^2,size(g_features_,1),1))*Wg + b_features*Wb + repmat(hidB,size(g_features_,1),1));
        %mmrbb_features = [mmrbb_features > rand(size(mmrbb_features)) r_features];        
    else
        mmrbb_features = raw_features(:,[G_INX B_INX]);
        [coeff,~] = princomp(mmrbb_features(f_inx,:));
        mmrbb_features = mmrbb_features*coeff;
    end
    %figure(3);imagesc(mmrbb_features); colorbar; ylabel('Samples'); xlabel('Features');        
    %%%%%%%%%%%%%%%%%%%%%%%%%%% SRBM Layer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if SRBM_        
%       [coeff,~] = princomp(mmrbb_features(f_inx,:));
%       mmrbb_features = mmrbb_features*coeff(:,1:20);

      M = mean(mmrbb_features(f_inx,:));      
      D = std(mmrbb_features(f_inx,:));
           
      mmrbb_features = (mmrbb_features-repmat(M,size(mmrbb_features,1),1))./repmat(D,size(mmrbb_features,1),1);
      
      mmrbb_features(isnan(mmrbb_features)) = 0 ;
      srbm_conf.sNum = size(f_inx,1);     
     
%      [W visB hidB] = training_srbm1(srbm_conf,mmrbb_features(f_inx,:));
%      mmrbb_features = logistic((mmrbb_features*W+ repmat(hidB,size(mmrbb_features,1),1))/(srbm_conf.sigma^2));        

     [W,hidB,ms,sigs] = training_srbm_(srbm_conf,mmrbb_features(f_inx,:));
     mmrbb_features = logistic(bsxfun(@rdivide,mmrbb_features,sigs.^2)*W+ repmat(hidB,size(mmrbb_features,1),1));

%      mmrbb_features  = mmrbb_features>rand(size(mmrbb_features));
     
%     figure(4);imagesc(mmrbb_features); colorbar;   
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%% RBM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if RBM_
     disp(rbm_conf);
     rbm_conf.sNum = size(f_inx,1);
     [W visB hidB] = training_rbm_(rbm_conf,[],mmrbb_features(f_inx,:));
     mmrbb_features = logistic(mmrbb_features*W+ repmat(hidB,size(mmrbb_features,1),1));
     %mmrbb_features  = mmrbb_features>rand(size(mmrbb_features));
     
%     figure(5);colormap(hot);imagesc(mmrbb_features); colorbar; ylabel('Samples'); xlabel('Features');
     pause();
    end
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Subspace is used?          
    w = SUB_SIZE;
    [trnd_12 trnd_13] = subspace_distances(trn_inx,mmrbb_features,indices,w,0);
    [tstd_12 tstd_13] = subspace_distances(tst_inx,mmrbb_features,indices,w,0);
    % train similarity model
    %W = linear_programming(trnd_12{f},trnd_13{f});
    %cr_ = cr_ + sum((trnd_12{f}-trnd_13{f})*W' < 0, 1)/size(trnd_12{f},1); %'
    %[W rr] = gradient_ascent_(trnd_12{f},trnd_13{f},0.01,0.01,1,tstd_12{f},tstd_13{f});        
%      for cxx=0.5:0.01:2
%          cr  = 0;
%          cr_ = 0;
    [W] = learn_soft_margin(trnd_12{f},trnd_13{f},cxx,1000,0.0001)';   
    cr_ = cr_ + sum((trnd_12{f}-trnd_13{f})*W' < 0, 1)/size(trnd_12{f},1);
    cr = cr + sum((tstd_12{f}-tstd_13{f})*W' < 0, 1)/size(tstd_12{f},1); %'    
%      logging(strcat(exp_dir,'test.mat'),[cxx cr_ cr]);
%     end
end
cr_ = cr_/num_case;
cr  = cr/num_case;
fprintf('Accuracy (Training|Testing) %f|%f\n',cr_,cr);
logging(log_file,[cxx cr_ cr]);
clc;
end
end
end
end