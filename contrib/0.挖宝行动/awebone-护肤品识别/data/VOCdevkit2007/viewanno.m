function viewanno(imgset)

if nargin<1
    error(['usage: viewanno(imgset) e.g. viewanno(' 39 'Main/train' 39 ') ' ...
            'or viewanno(' 39 'Main/car_train' 39 ')']);
end

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

% load image set
[ids,gt]=textread(sprintf(VOCopts.imgsetpath,['../' imgset]),'%s %d');

for i=1:length(ids)
    
    % read annotation
    rec=PASreadrecord(sprintf(VOCopts.annopath,ids{i}));
    
    % read image
    I=imread(sprintf(VOCopts.imgpath,ids{i}));

    if rec.segmented

        % read segmentations
        
        [Sclass,CMclass]=imread(sprintf(VOCopts.seg.clsimgpath,ids{i}));
        [Sobj,CMobj]=imread(sprintf(VOCopts.seg.instimgpath,ids{i}));
    end
    
    % display annotation
    
    if rec.segmented
        subplot(131);
    else
        clf;
    end
    
    imshow(I);
    hold on;
    for j=1:length(rec.objects)
        bb=rec.objects(j).bbox;
        if rec.objects(j).difficult
            ls='y'; % "difficult": yellow
        else
            ls='g'; % not "difficult": green
        end
        if rec.objects(j).truncated
            ls=[ls ':'];    % truncated: dotted
        else
            ls=[ls '-'];    % not truncated: solid
        end
        plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),ls,'linewidth',2);
        text(bb(1),bb(2),rec.objects(j).class,'color','k','backgroundcolor',ls(1),...
            'verticalalignment','top','horizontalalignment','left','fontsize',8);
        
        for k=1:length(rec.objects(j).part)
            bb=rec.objects(j).part(k).bbox;
            plot(bb([1 3 3 1 1]),bb([2 2 4 4 2]),ls,'linewidth',2);
            text(bb(1),bb(2),rec.objects(j).part(k).class,'color','k','backgroundcolor',ls(1),...
                'verticalalignment','top','horizontalalignment','left','fontsize',8);
        end
    end
    hold off;
    axis image;
    axis off;
    title(sprintf('image: %d/%d: "%s" (dotted=truncated, yellow=difficult)',...
            i,length(ids),ids{i}));
    
    if rec.segmented
        subplot(132);
        imshow(Sclass,CMclass);
        axis image;
        axis off;
        title('segmentation by class');
        
        subplot(133);
        imshow(Sobj,CMobj);
        axis image;
        axis off;
        title('segmentation by object');
    end
        
    fprintf('press any key to continue with next image\n');
    pause;
end
