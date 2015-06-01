%if ~exist('traind','var')
    load /home/funzi/My.Academic/DATA/CHAR.REG/MNIST/mnist_train_dat_60k.mat;
%end

conf.hidNum = 1000;
conf.eNum   = 100;                                                           % number of epoch
conf.bNum   = 200;                                                           % number of batches devides the data set
conf.sNum   = 100;                                                          % number of samples per batch
conf.gNum   = 1;                                                            % number of iteration for gibb sampling
conf.params = [0.001 0.001 0.02 0.0002];                                        % [lr1 lr2 momentum cost] 
conf.N      = 50;
conf.MAX_INC = 20;

conf.vis_type = 1; % real-valued visible unit

conf.plot_ = 0;

conf.pca   = 1;

conf.sigma  = 0.4;
conf.lambda = 0.1
conf.p      = 0.000001;

%% PCA whitening
C = 0;
 if conf.pca ==1
     traind = traind(1:conf.sNum*conf.bNum,:);
     [C traind] = princomp(traind);
     traind = traind(:,1:69);
 end
 
 M = mean(traind);      
 D = std(traind);       
 traind = (traind-repmat(M,size(traind,1),1))./repmat(D,size(traind,1),1);
 traind(isnan(traind(:))) = 0;
 traind(isinf(traind(:))) = 0;
%% Training
%[W visB hidB] = training_rbm_(conf,[],traind);
[W,visB,hidB] = training_srbm1(conf,traind);
%[W,hidB,ms,sigs] = training_srbm_(conf,traind);
%save(strcat(dir,'sparseRBM_mnist',num2str(size(traind,1)),'_hid',num2str(conf.hidNum),'.mat'),'conf','W','visB','hidB','C');
%load(strcat(dir,'sparseRBM_mnist',num2str(size(traind,1)),'_hid',num2str(conf.hidNum),'.mat'),'conf','W','visB','hidB','C');

%% Visualize
if conf.pca ==1
    base = W'*C(:,1:69)';
    display_bases(base,28,'norm1');
else 
    mn = min(min(W));
    mx = max(max(W));
    show_images((W'-mn)/(mx-mn),100,28,28);
end
