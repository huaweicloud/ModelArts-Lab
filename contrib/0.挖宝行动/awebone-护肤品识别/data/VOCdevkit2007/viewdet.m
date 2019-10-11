function viewdet(cls,onlytp)

if nargin<1
    error(['usage: viewdet(class,onlytp) e.g. viewdet(' 39 'car' 39 ') or ' ...
            'viewdet(' 39 'car' 39 ',true) to show true positives']);
end

if nargin<2
    onlytp=false;
end

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

% load test set
[gtids,t]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');

% load ground truth objects
tic;
npos=0;
for i=1:length(gtids)
    % display progress
    if toc>1
        fprintf('%s: viewdet: load: %d/%d\n',cls,i,length(gtids));
        drawnow;
        tic;
    end
    
    % read annotation
    rec=PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));
    
    % extract objects of class
    clsinds=strmatch(cls,{rec.objects(:).class},'exact');
    gt(i).BB=cat(1,rec.objects(clsinds).bbox)';
    gt(i).diff=[rec.objects(clsinds).difficult];
    gt(i).det=false(length(clsinds),1);
    npos=npos+sum(~gt(i).diff);
end

% load results
[ids,confidence,b1,b2,b3,b4]=textread(sprintf(VOCopts.detrespath,'comp3',cls),'%s %f %f %f %f %f');
BB=[b1 b2 b3 b4]';

% sort detections by decreasing confidence
[sc,si]=sort(-confidence);
ids=ids(si);
BB=BB(:,si);

% view detections
nd=length(confidence);
tic;
for d=1:nd
    % display progress
    if onlytp&toc>1
        fprintf('%s: viewdet: find true pos: %d/%d\n',cls,i,length(gtids));
        drawnow;
        tic;
    end
    
    % find ground truth image
    i=strmatch(ids{d},gtids,'exact');
    if isempty(i)
        error('unrecognized image "%s"',ids{d});
    elseif length(i)>1
        error('multiple image "%s"',ids{d});
    end

    % assign detection to ground truth object if any
    bb=BB(:,d);
    ovmax=-inf;
    for j=1:size(gt(i).BB,2)
        bbgt=gt(i).BB(:,j);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 & ih>0                
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end

    % skip false positives
    if onlytp&ovmax<VOCopts.minoverlap
        continue
    end
    
    % read image
    I=imread(sprintf(VOCopts.imgpath,gtids{i}));

    % draw detection bounding box and ground truth bounding box (if any)
    imagesc(I);
    hold on;
    if ovmax>=VOCopts.minoverlap
        bbgt=gt(i).BB(:,jmax);
        plot(bbgt([1 3 3 1 1]),bbgt([2 2 4 4 2]),'y-','linewidth',2);
        plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),'g:','linewidth',2);
    else
        plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),'r-','linewidth',2);
    end    
    hold off;
    axis image;
    axis off;
    title(sprintf('det %d/%d: image: "%s" (green=true pos,red=false pos,yellow=ground truth',...
            d,nd,gtids{i}));
    
    fprintf('press any key to continue with next image\n');
    pause;
end
