function [d_12 d_13] = subspace_distances(f_inx,features,indices,w,norm)
    num_case = max(size(f_inx));
    d_12 = cell(1,num_case);
    d_13 = cell(1,num_case);    
    if w == 0 % d_12 = d(a,b) , d_13= d(a,c) 
      for i = 1:num_case % over all cross-validation folds (num_case)
        [d_12{i} d_13{i}] = simple_dist(f_inx{i},features,indices);    
      end
    else
      for i = 1:num_case % for w > 1
        [d_12{i} d_13{i}] = conv_euclidean_dist(f_inx{i},features,indices,w,norm);    %% normalize is better than no normalize in almost the case
      end
    end
end
