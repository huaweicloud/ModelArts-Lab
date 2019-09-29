function VOCwritexml(rec, path)

fid=fopen(path,'w');
writexml(fid,rec,0);
fclose(fid);

function xml = writexml(fid,rec,depth)

fn=fieldnames(rec);
for i=1:length(fn)
    f=rec.(fn{i});
    if ~isempty(f)
        if isstruct(f)
            for j=1:length(f)            
                fprintf(fid,'%s',repmat(char(9),1,depth));
                fprintf(fid,'<%s>\n',fn{i});
                writexml(fid,rec.(fn{i})(j),depth+1);
                fprintf(fid,'%s',repmat(char(9),1,depth));
                fprintf(fid,'</%s>\n',fn{i});
            end
        else
            if ~iscell(f)
                f={f};
            end       
            for j=1:length(f)
                fprintf(fid,'%s',repmat(char(9),1,depth));
                fprintf(fid,'<%s>',fn{i});
                if ischar(f{j})
                    fprintf(fid,'%s',f{j});
                elseif isnumeric(f{j})&&numel(f{j})==1
                    fprintf(fid,'%s',num2str(f{j}));
                else
                    error('unsupported type');
                end
                fprintf(fid,'</%s>\n',fn{i});
            end
        end
    end
end

