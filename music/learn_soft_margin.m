function W = learn_soft_margin(trn_12,trn_13,C,max_iter,tol)
   X =  trn_13 - trn_12;
   alpha = zeros(size(X,1),1);
   stop_f = 0;
   i = 1;
   oW = 0;
   no_change = 0;
   while ~stop_f && i<max_iter       
       L = 0;
       for s=1:size(X,1)
           W = X'*alpha;
           W(W<0) = 0;
           l_ = (1-X(s,:)*W);
           a  = (l_/sum(X(s,:).^2,2) + alpha(s));
           L = L + (l_>0);
           if  a> C
               alpha(s) = C;
           elseif a<0
               alpha(s) = 0;
           else 
               alpha(s) = a;
           end 
       end
       fprintf('L  =%f\n',L);
       
       %% check kkt & stopping condition              
       KKT = X*W;       
       kkt_1 = sum(sum(KKT(alpha==0)<1-tol));
       kkt_12  = sum(sum((KKT(alpha>0 & alpha<C)-1)>tol));
       kkt_2 = sum(sum(KKT(alpha==C)>1+tol));
       
       diff = abs(W-X'*alpha);
       kkt_3 = sum(sum(diff(W>0)>tol)); 
       
       if isequal(W,oW)
           no_change = no_change+1;
       end
       if no_change>3
           fprintf('There is no change since iteration %d\n',i-3);
           break;
       end
       if rem(i,100)==0
           fprintf('%d iteration passed, KKT(%d|%d|%d)\n',i,kkt_1,kkt_2,kkt_3);
       end
       oW = W;       
       if kkt_1==0 && kkt_2==0 && kkt_12
           %fprintf('First KKT met\n');           
           if kkt_3==0
               fprintf('Optima found after %d iterations!!\n',i);
            stop_f = 1;
           end
       end
       i = i+1;
   end
   if i>= max_iter
       fprintf('All iteration has passed. Optimization may not be found\n');
   end
end
