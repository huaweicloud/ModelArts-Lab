% Creates segmentation results from detection results. 
% CREATE_SEGMENTATIONS_FROM_DETECTIONS(ID) creates segmentations from 
% the detection results with identifier ID e.g. 'comp3'.  All detections
% will be used, no matter what their confidence level.
%
% CREATE_SEGMENTATIONS_FROM_DETECTIONS(ID, CONFIDENCE) as above, but only 
% detections above the specified confidence will be used.  
function create_segmentations_from_detections(id,confidence)

if nargin<2
    confidence = -inf;
end

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit; 

% load detection results 

tic;
imgids={};
for clsnum = 1:VOCopts.nclasses
    resultsfile = sprintf(VOCopts.detrespath,id,VOCopts.classes{clsnum});
    if ~exist(resultsfile,'file')
        error('Could not find detection results file to use to create segmentations (%s not found)',resultsfile);
    end
    [ids,confs,b1,b2,b3,b4]=textread(resultsfile,'%s %f %f %f %f %f');
    BBOXS=[b1 b2 b3 b4];
    previd='';
    for j=1:numel(ids)
        % display progress
        if toc>1
            fprintf('class %d/%d: load detections: %d/%d\n',clsnum,VOCopts.nclasses,j,numel(ids));
            drawnow;
            tic;
        end
        
        imgid = ids{j};
        conf = confs(j);
        
        if ~strcmp(imgid,previd)
            ind = strmatch(imgid,imgids,'exact');
        end
        
        detinfo.clsnum = clsnum;
        detinfo.conf = conf;
        detinfo.bbox = BBOXS(j,:);        
        if isempty(ind)
            imgids{end+1}=imgid;
            ind = numel(imgids);
            detnum=1;
        else
            detnum = numel(im(ind).det)+1;
        end
        im(ind).det(detnum) = detinfo;        
    end
end

% Write out the segmentations
resultsdir = sprintf(VOCopts.seg.clsresdir,id,VOCopts.testset);
resultsdirinst = sprintf(VOCopts.seg.instresdir,id,VOCopts.testset);

if ~exist(resultsdir,'dir')
    mkdir(resultsdir);
end

if ~exist(resultsdirinst,'dir')
    mkdir(resultsdirinst);
end

cmap = VOClabelcolormap(255);
tic;
for j=1:numel(imgids)
    % display progress
    if toc>1
        fprintf('make segmentation: %d/%d\n',j,numel(imgids));
        drawnow;
        tic;
    end
    imname = imgids{j};

    classlabelfile = sprintf(VOCopts.seg.clsrespath,id,VOCopts.testset,imname);
    instlabelfile = sprintf(VOCopts.seg.instrespath,id,VOCopts.testset,imname);

    imgfile = sprintf(VOCopts.imgpath,imname);
    imginfo = imfinfo(imgfile);

    [instim,classim]= convert_dets_to_image(imginfo.Width, imginfo.Height,im(j).det,confidence);
    imwrite(instim,cmap,instlabelfile);
    imwrite(classim,cmap,classlabelfile);    
    
% Copy in ground truth - uncomment to copy ground truth segmentations in
% for comparison
%    gtlabelfile = [VOCopts.root '/Segmentations(class)/' imname '.png'];
%    gtclasslabelfile = sprintf('%s/%d_gt.png',resultsdir,imnums(j));        
%    copyfile(gtlabelfile,gtclasslabelfile);
end

% Converts a set of detected bounding boxes into an instance-labelled image
% and a class-labelled image
function [instim,classim]=convert_dets_to_image(W,H,dets,confidence)

    instim = uint8(zeros([H W]));
    classim = uint8(zeros([H W]));  
    for j=1:numel(dets)
        detinfo = dets(j);
        if detinfo.conf<confidence
            continue
        end
        bbox = round(detinfo.bbox); 
        % restrict to fit within image
        bbox([1 3]) = min(max(bbox([1 3]),1),W);
        bbox([2 4]) = min(max(bbox([2 4]),1),H);      
        instim(bbox(2):bbox(4),bbox(1):bbox(3)) = j;
        classim(bbox(2):bbox(4),bbox(1):bbox(3)) = detinfo.clsnum;      
    end


