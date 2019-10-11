%VOCEVALSEG Creates a confusion matrix for a set of segmentation results.
% VOCEVALSEG(VOCopts,ID); prints out the per class and overall
% segmentation accuracies.
%
% [ACCURACIES,AVACC,CONF] = VOCEVALSEG(VOCopts,ID) returns the per class
% percentage ACCURACIES, the average accuracy AVACC and the confusion
% matrix CONF.
function [accuracies,avacc,conf,rawcounts] = VOCevalseg(VOCopts,id)

% image test set
[gtids,t]=textread(sprintf(VOCopts.seg.imgsetpath,VOCopts.testset),'%s %d');

% number of labels = number of classes plus one for the background
num = VOCopts.nclasses+1; 
confcounts = zeros(num);
count=0;
tic;
for i=1:length(gtids)
    % display progress
    if toc>1
        fprintf('test confusion: %d/%d\n',i,length(gtids));
        drawnow;
        tic;
    end
        
    imname = gtids{i};
    
    % ground truth label file
    gtfile = sprintf(VOCopts.seg.clsimgpath,imname);
    [gtim,map] = imread(gtfile);    
    gtim = double(gtim);
    
    % results file
    resfile = sprintf(VOCopts.seg.clsrespath,id,VOCopts.testset,imname);
    [resim,map] = imread(resfile);
    resim = double(resim);
    
    % Check validity of results image
    maxlabel = max(resim(:));
    if (maxlabel>VOCopts.nclasses), 
        error('Results image ''%s'' has out of range value %d (the value should be <= %d)',imname,maxlabel,VOCopts.nclasses);
    end

    szgtim = size(gtim); szresim = size(resim);
    if any(szgtim~=szresim)
        error('Results image ''%s'' is the wrong size, was %d x %d, should be %d x %d.',imname,szresim(1),szresim(2),szgtim(1),szgtim(2));
    end
    
    %pixel locations to include in computation
    locs = gtim<255;
    
    % joint histogram
    sumim = 1+gtim+resim*num; 
    hs = histc(sumim(locs),1:num*num); 
    count = count + numel(find(locs));
    confcounts(:) = confcounts(:) + hs(:);
end

% confusion matrix - first index is true label, second is inferred label
conf = zeros(num);
rawcounts = confcounts;
overall_acc = 100*sum(diag(confcounts)) / sum(confcounts(:));
fprintf('Percentage of pixels correctly labelled overall: %6.3f%%\n',overall_acc);
accuracies = zeros(VOCopts.nclasses,1);
fprintf('Percentage of pixels correctly labelled for each class\n');
for j=1:num
   rowsum = sum(confcounts(j,:)); 
   if (rowsum>0), conf(j,:) = 100*confcounts(j,:)/rowsum;   end;
   accuracies(j) = conf(j,j);
   clname = 'background';
   if (j>1), clname = VOCopts.classes{j-1};end;
   fprintf('  %14s: %6.3f%%\n',clname,accuracies(j));
end
accuracies = accuracies(1:end);
avacc = mean(accuracies);
fprintf('-------------------------\n');
fprintf('Average accuracy: %6.3f%%\n',avacc);
