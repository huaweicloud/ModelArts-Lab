function [rec,prec,ap] = VOCevallayout(VOCopts,id,cls,draw)

% load test set
[gtids,t]=textread(sprintf(VOCopts.layout.imgsetpath,VOCopts.testset),'%s %d');

npos=0;
tic;
for i=1:length(gtids)
    % display progress
    if toc>1
        fprintf('%s: layout pr: load: %d/%d\n',cls,i,length(gtids));
        drawnow;
        tic;
    end
    rec=PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));
    rec.objects=rec.objects(strmatch(cls,{rec.objects(:).class},'exact'));
    if ~isempty(rec.objects)
        [rec.objects.detected]=deal(false);
    end
    gt(i)=rec;
    npos=npos+sum([rec.objects.hasparts]&~[rec.objects.difficult]);
end

% load results

fprintf('layout pr: loading results...\n');
res=VOCreadxml(sprintf(VOCopts.layout.respath,id,cls));

% sort detections by decreasing confidence
[t,si]=sort(-str2double({res.results.layout.confidence}));

% assign detections to ground truth objects
nd=length(si);
tp=zeros(nd,1);
fp=zeros(nd,1);
tic;
for di=1:nd
    % display progress
    if toc>1
        fprintf('%s: layout pr: compute: %d/%d\n',cls,di,nd);
        drawnow;
        tic;
    end
    
    d=res.results.layout(si(di));    
    id=d.image;
    
    % find ground truth image
    i=strmatch(id,gtids,'exact');
    if isempty(i)
        error('unrecognized image "%s"',ids{d});
    elseif length(i)>1
        error('multiple image "%s"',ids{d});
    end

    % assign detection to ground truth object if any
    bb=str2double({d.bndbox.xmin d.bndbox.ymin d.bndbox.xmax d.bndbox.ymax});
    ovmax=-inf;
    for j=1:length(gt(i).objects)
        bbgt=gt(i).objects(j).bbox;
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
        
    % assign detection as true/false positive
    if ovmax>=VOCopts.minoverlap        
        o=gt(i).objects(jmax);
        if ~o.detected
            gt(i).objects(jmax).detected=true;
            
            % num detected parts = num gt parts?
            if length(o.part)==length(d.part)            
                op=zeros(size(o.part));
                dp=zeros(size(d.part));
                for k=1:VOCopts.nparts
                    op(strmatch(VOCopts.parts{k},{o.part.class},'exact'))=k;
                    dp(strmatch(VOCopts.parts{k},{d.part.class},'exact'))=k;
                end                       
                % bag of detected parts = bag of gt parts?
                if all(sort(op)==sort(dp))
                    % find possible matches (same type + overlap)
                    M=zeros(length(op));
                    for k=1:length(dp)
                        bb=str2double({d.part(k).bndbox.xmin d.part(k).bndbox.ymin ...
                                        d.part(k).bndbox.xmax d.part(k).bndbox.ymax});
                        for l=find(op==dp(k))
                            bbgt=o.part(l).bbox;
                            bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
                            iw=bi(3)-bi(1)+1;
                            ih=bi(4)-bi(2)+1;
                            if iw>0 & ih>0                
                                % compute overlap as area of intersection / area of union
                                ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
                                   (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
                                   iw*ih;
                                ov=iw*ih/ua;
                                M(k,l)=ov>=VOCopts.minoverlap;                            
                            end
                        end
                    end   
                    % valid assignments for all part types?
                    tp(di)=1;
                    for k=1:VOCopts.nparts
                        v=(op==k);
                        % each part matchable and sufficient matches?
                        if ~(all(any(M(:,v)))&&sum(any(M(:,v),2))>=sum(v))
                            tp(di)=0;
                            fp(di)=1;
                            break
                        end
                    end

                else
                    fp(di)=1; % wrong bag of parts
                end
            else
                fp(di)=1; % wrong number of parts
            end
        else
            fp(di)=1;   % multiple detection
        end
    else
        fp(di)=1;   % no overlapping object
    end
end

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
v=tp+fp>0;
rec=tp(v)/npos;
prec=tp(v)./(fp(v)+tp(v));

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
