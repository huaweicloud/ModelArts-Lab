function record=PASreadrectxt(filename)
  [fd,syserrmsg]=fopen(filename,'rt');
  if (fd==-1),
    PASmsg=sprintf('Could not open %s for reading',filename);
    PASerrmsg(PASmsg,syserrmsg); 
  end;
  
  matchstrs=initstrings;
  record=PASemptyrecord;
  notEOF=1;
  while (notEOF),
    line=fgetl(fd);
    notEOF=ischar(line);
    if (notEOF),
      matchnum=match(line,matchstrs);
      switch matchnum,
    case 1, [imgname]=strread(line,matchstrs(matchnum).str);
	        record.imgname=char(imgname);
	case 2, [x,y,c]=strread(line,matchstrs(matchnum).str);
	        record.imgsize=[x y c];
	case 3, [database]=strread(line,matchstrs(matchnum).str);
	        record.database=char(database);
	case 4, [obj,lbl,xmin,ymin,xmax,ymax]=strread(line,matchstrs(matchnum).str);
	        record.objects(obj).label=char(lbl);
		record.objects(obj).bbox=[min(xmin,xmax),min(ymin,ymax),max(xmin,xmax),max(ymin,ymax)];
	case 5, tmp=findstr(line,' : ');
	        [obj,lbl]=strread(line(1:tmp),matchstrs(matchnum).str);
	        record.objects(obj).label=char(lbl);
    		record.objects(obj).polygon=sscanf(line(tmp+3:end),'(%d, %d) ')';
	case 6, [obj,lbl,mask]=strread(line,matchstrs(matchnum).str);
	        record.objects(obj).label=char(lbl);
    		record.objects(obj).mask=char(mask);
	case 7, [obj,lbl,orglbl]=strread(line,matchstrs(matchnum).str);
            lbl=char(lbl);
	        record.objects(obj).label=lbl;
    		record.objects(obj).orglabel=char(orglbl);
            if strcmp(lbl(max(end-8,1):end),'Difficult')
                record.objects(obj).difficult=true;
                lbl(end-8:end)=[];
            else
                record.objects(obj).difficult=false;
            end
            if strcmp(lbl(max(end-4,1):end),'Trunc')
                record.objects(obj).truncated=true;
                lbl(end-4:end)=[];
            else
                record.objects(obj).truncated=false;
            end
            t=find(lbl>='A'&lbl<='Z');
            t=t(t>=4);
            if ~isempty(t)
                record.objects(obj).view=lbl(t(1):end);
                lbl(t(1):end)=[];
            else
                record.objects(obj).view='';                
            end
            record.objects(obj).class=lbl(4:end);
        
	otherwise, %fprintf('Skipping: %s\n',line);
      end;
    end;
  end;
  fclose(fd);
return

function matchnum=match(line,matchstrs)
  for i=1:length(matchstrs),
    matched(i)=strncmp(line,matchstrs(i).str,matchstrs(i).matchlen);
  end;
  matchnum=find(matched);
  if isempty(matchnum), matchnum=0; end;
  if (length(matchnum)~=1), 
    PASerrmsg('Multiple matches while parsing','');
  end;
return

function s=initstrings
  s(1).matchlen=14;
  s(1).str='Image filename : %q';
  
  s(2).matchlen=10;
  s(2).str='Image size (X x Y x C) : %d x %d x %d';
  
  s(3).matchlen=8;
  s(3).str='Database : %q';
  
  s(4).matchlen=8;
  s(4).str='Bounding box for object %d %q (Xmin, Ymin) - (Xmax, Ymax) : (%d, %d) - (%d, %d)';
  
  s(5).matchlen=7;
  s(5).str='Polygon for object %d %q (X, Y)';
  
  s(6).matchlen=5;
  s(6).str='Pixel mask for object %d %q : %q';

  s(7).matchlen=8;
  s(7).str='Original label for object %d %q : %q';

return