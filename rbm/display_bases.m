function display_bases(B,row,opt)
figure;
if exist('opt','var')    
    if strcmp(opt,'log')
        B = logistic(10*B);
    elseif strcmp(opt,'norm1')
        B = B./repmat(max(abs(B),[],2),1,size(B,2));
        B = (B+1)/2;
    elseif strcmp(opt,'norm2')
        mn  = min(min(B));
        mx  =  max(max(B));
        B = (B-mn)/(mx-mn);
    end
end
show_images(B,size(B,1),row,round(size(B,2)/row));
end