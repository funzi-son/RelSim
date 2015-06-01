%% DIRECTORY
sys_inf = computer();
if ~isempty(findstr('WIN',sys_inf))
    exp_dir = 'C:\Pros\Experiments\Sparse.MM.Sim\MUSIC\';
    lm = '\';
elseif ~isempty(findstr('linux',sys_inf)) || ~isempty(findstr('GLNX',sys_inf))
    exp_dir = '/home/funzi/Documents/Experiments/Sparse.MM.Sim/MUSIC/';
    lm = '/';
end
%% NUMBER OF TRIALS
TRIAL_NUM = 10;
%% MODALITY
CASE = 1    
switch CASE
    case 1
        %LONG CASE
        feature_file = 'rel_music_raw_features+simdata_ISMIR12.mat';
        B_INX = [31:131 157:197];
        G_INX = [1:30 132:156];
        %R_INX = [];
        %R_INX = [1:197];
        %R_INX = [31:131 157:197];
        R_INX = [];
        
        exp_dir = strcat(exp_dir,'LONG_DAT',lm);
    case 2
        %SHORT CASE
        feature_file = 'rel_music_raw_features.mat';
        B_INX = [27:66];
        G_INX = [1:12 14:25 68:83];
        R_INX = [];%[1:12 14:25 27:66 68:83];        
        
        exp_dir = strcat(exp_dir,'SHORT_DAT',lm);
end
%% SUBSPACE
SUB_SIZE = 0;