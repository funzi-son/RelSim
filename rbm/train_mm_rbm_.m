function [Wg Wb visBb hidB ms sigs] = train_mm_rbm_(conf,g_dat,b_dat)
% Training multi-model rbm (no lower layers linked)
% sontran 2013
assert(~isempty(g_dat) && ~isempty(b_dat),'[KBRBM] Data is empty'); 
%% initialization
visGNum  = size(g_dat,2);
visBNum  = size(b_dat,2);
hidNum   = conf.hidNum;

sNum  = conf.sNum;
lr    = conf.params(1);
N     = conf.N;                                                                     % Number of epoch training with lr_1                     


Wg    = 0.1*randn(visGNum,hidNum);
Wb    = 0.1*randn(visBNum,hidNum);
DWG   = zeros(size(Wg));
DWB   = zeros(size(Wb));

visBg  = zeros(1,visGNum);
DVBG   = zeros(1,visGNum);
visBb  = zeros(1,visBNum);
DVBB   = zeros(1,visBNum);

hidB  = zeros(1,hidNum);
DHB   = zeros(1,hidNum);

if ~conf.sigma
   ms   = 0.5*randn(1,visGNum);   
else
   ms = zeros(1,visGNum);
end
sigs = ones(1,visGNum);

DM = zeros(1,visGNum);
DS = zeros(1,visGNum);
%% Reconstruction error & evaluation error & early stopping
mse    = 0;
omse   = 0;
inc_count = 0;
MAX_INC = conf.MAX_INC;                                                                % If the error increase MAX_INC times continuously, then stop training
%% Plotting
if conf.plot_, h = plot(nan); end
tic;
%%%%%%%%%%
%% ==================== Start training =========================== %%
for i=1:conf.eNum
    if i== N+1
        lr = conf.params(2);
    end
    omse = mse;
    mse = 0;
    for j=1:conf.bNum
       visPg = g_dat((j-1)*conf.sNum+1:j*conf.sNum,:);
       visPb = b_dat((j-1)*conf.sNum+1:j*conf.sNum,:);
       %up            
       hidI  = bsxfun(@rdivide,visPg,sigs.^2)*Wg + visPb*Wb+ repmat(hidB,sNum,1);       
       hidP  = logistic(hidI);
       hidPs =  1*(hidP > rand(size(hidP)));
       hidNs = hidPs;
       
       % for relation constraint
       if conf.rel>0
           hidIg = bsxfun(@rdivide,visPg,sigs.^2)*Wg + repmat(hidB,sNum,1);
           hidIb = visPb*Wb + repmat(hidB,sNum,1);
       end
       
       for k=1:conf.gNum
           % down
           visNg  = (hidNs*Wg' + repmat(ms,sNum,1)); % sampling from normal distribution           
           visNgs = visNg + repmat(sigs,sNum,1).*randn(sNum,visGNum); % sampling from normal distribution          
           
           visNb  = logistic(hidNs*Wb' + repmat(visBb,sNum,1));
           visNbs = 1*(visNb>rand(size(visNb)));
           % up
           hidN  = logistic(bsxfun(@rdivide,visNgs,sigs.^2)*Wg  + visNbs*Wb + repmat(hidB,sNum,1));
           hidNs = 1*(hidN>rand(size(hidN)));
       end
       % Compute MSE for reconstruction       
       mse = mse + mserr(visPg,visNgs) + mserr(visPb,visNbs);
       % Update parameters       
       diff_g = (bsxfun(@rdivide,visPg,sigs)'*hidP - bsxfun(@rdivide,visNgs,sigs)'*hidN)/sNum;
       DWG  = lr*(diff_g - conf.params(4)*Wg) +  conf.params(3)*DWG;
       Wg   = Wg + DWG;
       
       diff_b = (visPb'*hidP - visNbs'*hidN)/sNum;
       DWB  = lr*(diff_b - conf.params(4)*Wb) +  conf.params(3)*DWB;
       Wb   = Wb + DWB;
                    
       DVBB  = lr*sum(visPb - visNbs,1)/sNum + conf.params(3)*DVBB;
       visBb = visBb + DVBB;
       
       DHB  = lr*sum(hidP - hidN,1)/sNum + conf.params(3)*DHB;
       hidB = hidB + DHB;
       
       DM  = lr*sum(visPg - visNgs)./(sNum*sigs.^2) + conf.params(3)*DM;
       ms  = ms + DM;
             
       % If sparse constraint is applied
       if conf.lambda >0
           % Check this one, shoud hidI = inp(hidP);           
           %hidB = hidB + lr*(2*conf.lambda)*((conf.p - sum(hidP,1)/sNum).*(sum((hidP.^2).*exp(-hidI),1)/sNum));                  
           %WORKING HERE
           pppp = (conf.p - sum(hidP,1)/sNum);           
           
           Wg = Wg + lr*conf.lambda*(repmat(pppp,visGNum,1).*(visPg'*((hidP.^2).*exp(-hidI))));
           Wb = Wb + lr*conf.lambda*(repmat(pppp,visBNum,1).*(visPb'*((hidP.^2).*exp(-hidI))));
           %u_diff = 
           hidB = hidB + lr*conf.lambda*(pppp.*(sum((hidP.^2).*exp(-hidI),1)/sNum));
           %l_diff
       end
       % If relation constraint is on
        if conf.rel>0
            ppppG = logistic(hidIg);
            ppppB = logistic(hidIb);
            pppp  = ppppG - ppppB;
            gggg  = (ppppG.^2).*exp(-hidIg).*pppp;
            bbbb  = -(ppppB.^2).*exp(-hidIb).*pppp;
            Wg = Wg + lr*conf.rel*(visPg'*gggg)/sNum;
            Wb = Wb + lr*conf.rel*(visPb'*bbbb)/sNum;
        end    
    end
    
    fprintf('Epoch %d  : MSE = %f\n',i,mse);
    if isnan(mse) || mse > 100, Wg = 0; Wb = 0; break; end
end
toc;
end
