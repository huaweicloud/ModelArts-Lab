function example_layout

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

% train and test detector

cls='person';
detector=train(VOCopts,cls);                                % train detector
test(VOCopts,cls,detector);                                 % test detector
[recall,prec,ap]=VOCevallayout(VOCopts,'comp6',cls,true);   % compute and display PR    

% train detector
function detector = train(VOCopts,cls)

% load 'train' image set
ids=textread(sprintf(VOCopts.layout.imgsetpath,'train'),'%s');

% extract features and objects
n=0;
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: train: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end
    
    % read annotation
    rec=PASreadrecord(sprintf(VOCopts.annopath,ids{i}));
    
    % find objects of class and extract difficult flags for these objects
    clsinds=strmatch(cls,{rec.objects(:).class},'exact');
    diff=[rec.objects(clsinds).difficult];
    hasparts=[rec.objects(clsinds).hasparts];
    
    % assign ground truth class to image
    if isempty(clsinds)
        gt=-1;          % no objects of class
    elseif any(~diff&hasparts)
        gt=1;           % at least one non-difficult object with parts
    else
        gt=0;           % only difficult/objects without parts
    end

    if gt
        % extract features for image
        try
            % try to load features
            load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        catch
            % compute and save features
            I=imread(sprintf(VOCopts.imgpath,ids{i}));
            fd=extractfd(VOCopts,I);
            save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        end
        
        n=n+1;
        
        detector(n).fd=fd;
        
        % extract non-difficult objects with parts

        detector(n).object=rec.objects(clsinds(~diff&hasparts));
        
        % mark image as positive or negative
        
        detector(n).gt=gt;
    end
end    

% run detector on test images
function out = test(VOCopts,cls,detector)

% load test set ('val' for development kit)
[ids,gt]=textread(sprintf(VOCopts.layout.imgsetpath,VOCopts.testset),'%s %d');

% apply detector to each image
rec.results.layout=[];
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: test: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end
    
    try
        % try to load features
        load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    catch
        % compute and save features
        I=imread(sprintf(VOCopts.imgpath,ids{i}));
        fd=extractfd(VOCopts,I);
        save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
    end

    % compute confidence of positive classification and layout
    
    l=detect(VOCopts,detector,fd,ids{i});
    if isempty(rec.results.layout)
        rec.results.layout=l;
    else
        rec.results.layout=[rec.results.layout l];
    end
end

% write results file

fprintf('saving results...\n');
VOCwritexml(rec,sprintf(VOCopts.layout.respath,'comp6',cls));

% trivial feature extractor: compute mean RGB
function fd = extractfd(VOCopts,I)

fd=squeeze(sum(sum(double(I)))/(size(I,1)*size(I,2)));

% trivial detector: confidence is computed as in example_classifier, and
% bounding boxes of nearest positive training image are output
function layout = detect(VOCopts,detector,fd,imgid)

FD=[detector.fd];

% compute confidence
d=sum(fd.*fd)+sum(FD.*FD)-2*fd'*FD;
dp=min(d([detector.gt]>0));
dn=min(d([detector.gt]<0));
c=dn/(dp+eps);

% copy objects and layout from nearest positive image

pinds=find([detector.gt]>0);
[dp,di]=min(d(pinds));
pind=pinds(di);

BB=[];
for i=1:length(detector(pind).object)
    o=detector(pind).object(i);
        
    layout(i).image=imgid;
    layout(i).confidence=c;
    layout(i).bndbox.xmin=o.bbox(1);
    layout(i).bndbox.ymin=o.bbox(2);
    layout(i).bndbox.xmax=o.bbox(3);
    layout(i).bndbox.ymax=o.bbox(4);
        
    for j=1:length(o.part)    
        layout(i).part(j).class=o.part(j).class;
        layout(i).part(j).bndbox.xmin=o.part(j).bbox(1);
        layout(i).part(j).bndbox.ymin=o.part(j).bbox(2);
        layout(i).part(j).bndbox.xmax=o.part(j).bbox(3);
        layout(i).part(j).bndbox.ymax=o.part(j).bbox(4);
    end
end
