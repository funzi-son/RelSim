function [W hidB ms sigs] = training_srbm_(conf,data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training Sparse RBM (with real visible, binary hidden)             %
% Visible units are encoded by Gaussian distribution with 0 mean     %
% conf: training setting                                             %
% W: weights of connections                                          %
% -*-sontran2012-*-                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

assert(~isempty(data),'[KBRBM] Data is empty'); 
%% initialization
visNum  = size(data,2);
hidNum  = conf.hidNum;
sNum  = conf.sNum;
lr    = conf.params(1);
N     = conf.N;                                                                     % Number of epoch training with lr_1                     

W     = 0.01*randn(visNum,hidNum);
hidB  = 0.01*randn(1,hidNum);

DW    = zeros(size(W));
DVB   = zeros(1,visNum);
DHB   = zeros(1,hidNum);

if conf.sigma==0
   ms   = 0.5*randn(1,visNum);   
else
   ms = zeros(1,visNum);
end
sigs = conf.sigma*ones(1,visNum);

DM = zeros(1,visNum);
DS = zeros(1,visNum);

%% Reconstruction error & evaluation error & early stopping
mse    = 0;
omse   = 0;
inc_count = 0;
MAX_INC = conf.MAX_INC;                                                                % If the error increase MAX_INC times continuously, then stop training
%% Plotting
if conf.plot_, h = plot(nan); end
%% ==================== Start training =========================== %%
for i=1:conf.eNum
    if i== N+1
        lr = conf.params(2);
    end
    omse = mse;
    mse = 0;
    for j=1:conf.bNum
       visP = data((j-1)*conf.sNum+1:j*conf.sNum,:);
       %up       
       hidI = bsxfun(@rdivide,visP,sigs.^2)*W + repmat(hidB,sNum,1)/(conf.sigma^2);              
       %hidI = visP*W + repmat(hidB,sNum,1); 
       
       hidP = logistic(hidI);
       hidPs =  1*(hidP > rand(sNum,hidNum));
       hidNs = hidPs;
       for k=1:conf.gNum
            % down
            if conf.vis_type ==1                                
                visN  = (hidNs*W' + repmat(ms,sNum,1)); % sampling from normal distribution
                r = randn(sNum,visNum);
                visNs = visN + repmat(sigs,sNum,1).*r; % sampling from normal distribution                
            else
             visN = logistic(hidNs*W' + repmat(ms,sNum,1));             
             visNs = visN>rand(sNum,visNum);             
            end
            
           hidN  = logistic(bsxfun(@rdivide,visNs,sigs.^2)*W + repmat(hidB,sNum,1));
           %hidN  = logistic(visNs*W + repmat(hidB,sNum,1));           
           hidNs = 1*(hidN>rand(sNum,hidNum));
       end
       % Compute MSE for reconstruction       
       mse = mse + mserr(visP,visNs);
       % Update W,visB,hidB       
       diff = (bsxfun(@rdivide,visP,sigs.^2)'*hidP - bsxfun(@rdivide,visNs,sigs.^2)'*hidN)/sNum;              
       %diff = (visP'*hidP - visNs'*hidN)./sNum;       
       
       DW  = lr*(diff - conf.params(4)*W) +  conf.params(3)*DW;
       W   = W + DW;       
              
       DHB  = lr*sum(hidP - hidN,1)/sNum  + conf.params(3)*DHB;
       hidB = hidB + DHB;       
       
       DM  = lr*sum(visP - visNs)./(sNum*sigs.^2) + conf.params(3)*DM;                
        %DM  = lr*sum(visP - visNs)/sNum + conf.params(3)*DM;
        ms  = ms + DM;
       
        if conf.sigma==0
            
        end
       %% Update sparse regularization
       if conf.lambda>0
           hidI = bsxfun(@rdivide,visP,sigs.^2)*W + repmat(hidB,sNum,1);
           hidN = logistic(hidI);
           hidB = hidB + lr*(2*conf.lambda)*((conf.p - sum(hidN,1)/sNum).*(sum((hidN.^2).*exp(-hidI),1)/sNum));       
       end
    end
    %% 
    if conf.plot_
        mse_plot(i) = mse;
        axis([0 (conf.eNum+1) 0 5]);
        set(h,'YData',mse_plot);
        drawnow;
    end
    
    if mse > omse
        inc_count = inc_count + 1
    else
        inc_count = 0;
    end
    if inc_count> MAX_INC, break; end;
    fprintf('Epoch %d  : MSE = %f\n',i,mse);   
end
end