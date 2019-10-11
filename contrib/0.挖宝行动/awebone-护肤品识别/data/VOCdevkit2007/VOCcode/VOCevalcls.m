function [rec,prec,ap] = VOCevalcls(VOCopts,id,cls,draw)

% load test set
[gtids,gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d');

% load results
[ids,confidence]=textread(sprintf(VOCopts.clsrespath,id,cls),'%s %f');

% map results to ground truth images
out=ones(size(gt))*-inf;
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: pr: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end
    
    % find ground truth image
    j=strmatch(ids{i},gtids,'exact');
    if isempty(j)
        error('unrecognized image "%s"',ids{i});
    elseif length(j)>1
        error('multiple image "%s"',ids{i});
    else
        out(j)=confidence(i);
    end
end

% compute precision/recall

[so,si]=sort(-out);
tp=gt(si)>0;
fp=gt(si)<0;

fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/sum(gt>0);
prec=tp./(fp+tp);

% compute average precision

ap=0;
for t=0:0.1:1
    p=max(prec(rec>=t));
    if isempty(p)
        p=0;
    end
    ap=ap+p/11;
end

if draw
    % plot precision/recall
    plot(rec,prec,'-');
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('class: %s, subset: %s, AP = %.3f',cls,VOCopts.testset,ap));
end
