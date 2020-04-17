function [tmp,fps] = cal_success_rate(results, rect_anno)

seq_length = size(results.res,1);

if strcmp(results.type,'rect')
    for i = 2:seq_length
        r = results.res(i,:);
        r_anno = rect_anno(i,:);
        if (isnan(r) | r(3)<=0 | r(4)<=0)&(~isnan(r_anno))
            results.res(i,:)=results.res(i-1,:);
        end
    end
end


centerGT = [rect_anno(:,1)+(rect_anno(:,3)-1)/2 rect_anno(:,2)+(rect_anno(:,4)-1)/2];

rectMat = zeros(seq_length, 4);

rectMat = results.res;
              
                
rectMat(1,:) = rect_anno(1,:);

center = [rectMat(:,1)+(rectMat(:,3)-1)/2 rectMat(:,2)+(rectMat(:,4)-1)/2];

errCenter = sqrt(sum(((center(1:seq_length,:) - centerGT(1:seq_length,:)).^2),2));

index = rect_anno>0;
idx=(sum(index,2)==4);

tmp = calcRectInt(rectMat(idx,:),rect_anno(idx,:));

errCoverage=-ones(length(idx),1);
errCoverage(idx) = tmp;
errCenter(~idx)=-1;

aveErrCoverage = sum(errCoverage(idx))/length(idx);

aveErrCenter = sum(errCenter(idx))/length(idx);

%%plot
thresholdSet = 0:0.05:1;
thresholdSetError = 0:50;
for tIdx=1:length(thresholdSet)
    successNumOverlap(tIdx) = sum(errCoverage >thresholdSet(tIdx));
end

for tIdx=1:length(thresholdSetError)
    successNumErr(tIdx) = sum(errCenter <= thresholdSetError(tIdx));
end

lenALL = size(rect_anno,1);
aveSuccessRatePlot(:) = successNumOverlap/(lenALL+eps);
aveSuccessRatePlotErr(:) = successNumErr/(lenALL+eps);

aa = aveSuccessRatePlot;
aa = aa(sum(aa,2)>eps,:);
bb = mean(aa);



plot(thresholdSet,aa);
tmp = sprintf('%.3f', bb);
fps = sprintf('%.3f',results.fps)
% tmpNAME = ['ASRCF-APCE''[' tmp ']'];
% legend(tmpNAME);
% title('success plots of OPE','fontsize',fontSize);
% xlabel('Overlap threshold','fontsize',fontSize);
% ylabel('Success rate','fontsize',fontSize);
end