% This is the demo script for ASRCF

clear;
clc;
close all;
setup_path();

% get seq information
base_path  = 'D:/data_seq';
%base_path='F:\集成算法_fromxiao\data_seq\';
%问题较大的序列：
%遮挡导致跟丢：skating2.1
%模糊：dragonbaby、matrix

seqname = 'skating2';
video_path = [base_path '/' seqname];
[seq, ground_truth] = load_video_info(video_path,seqname);
seq.startFrame = 1;
seq.endFrame = seq.len;
seq.ground_truth=ground_truth;

rect_anno = dlmread(['./anno/' seq.name '.txt']);

reset = [0 1 0];

%%Run ASRCF- main function
if exist(['./result/' seq.name '_' 'ASRCF' '.mat'])&& reset(1) == 0
    load(['./result/' seq.name '_' 'ASRCF' '.mat']);
elseif reset(1)
    params.adapt_learn = 0;
    params.learning_rate       = 0.02;
    params.APCE = 0;
    params.APCEplus = 0;
    results1 = run_ASRCF(seq, video_path, params);
    save(['./result/' seq.name '_' 'ASRCF' '.mat'], 'results1');
end

%Run ASRCF-APCEplus main function
if exist(['./result/' seq.name '_' 'ASRCF_APCEplus' '.mat']) && reset(2) == 0
    load(['./result/' seq.name '_' 'ASRCF_APCEplus' '.mat']);
elseif reset(2)
    params.adapt_learn = 1;
    params.learning_rate       = 0.02;
    params.APCE = 0;
    params.APCEplus = 0;
    results2 = run_ASRCF(seq, video_path, params);
    save(['./result/' seq.name '_' 'ASRCF_APCEplus' '.mat'], 'results2');
end

%%Run ASRCF-APCE main function
if exist(['./result/' seq.name '_' 'ASRCF_APCE' '.mat']) && reset(3) == 0
    load(['./result/' seq.name '_' 'ASRCF_APCE' '.mat']);
elseif reset(2)
    params.adapt_learn = 0;
    params.learning_rate       = 0.1;
    params.APCE = 1;
    params.APCEplus = 0;
    results3 = run_ASRCF(seq, video_path, params);
    save(['./result/' seq.name '_' 'ASRCF_APCE' '.mat'], 'results3');
end

figure
[score1,fps1] = calc_success_rate(results1,rect_anno);
hold on 
[score2,fps2] = calc_success_rate(results2,rect_anno);
hold on 
[score3,fps3] = calc_success_rate(results3,rect_anno);

fontSize = 16;
tmpNAME{1} = ['ASRCF''[' score1 ']''(' fps1 ')'];
tmpNAME{2} = ['ASRCF-APCEplus''[' score2 ']''(' fps2 ')'];
tmpNAME{3} = ['ASRCF-APCE''[' score3 ']''(' fps3 ')'];
legend(tmpNAME);
title('success plots of OPE','fontsize',fontSize);
xlabel('Overlap threshold','fontsize',fontSize);
ylabel('Success rate','fontsize',fontSize);
