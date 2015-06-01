function [dis_s12,dis_s13] = simple_dist(index_list,vector_list,index_map)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the distance vector from 2 vector                                     %
% sontran2013                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s_num = size(index_list,1);
s_len = size(vector_list,2);

dis_s12 = zeros(s_num,s_len);
dis_s13 = zeros(s_num,s_len);

% Convert list of assigned indices to ordered indices
[checker index_list] = ismember(index_list,index_map);

if all(all(checker(:,1:3)))
  dis_s12 = (vector_list(index_list(:,1),:)-vector_list(index_list(:,2),:)).^2;
  dis_s13 = (vector_list(index_list(:,1),:)-vector_list(index_list(:,3),:)).^2;
else
  fprintf('Some indices are not mapped\n');
end

end
% simple_dist(sontest{1},sonfeatures,sonindex)