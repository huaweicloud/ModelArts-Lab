function rec = PASreadrecord(path)

if length(path)<4
    error('unable to determine format: %s',path);
end

if strcmp(path(end-3:end),'.txt')
    rec=PASreadrectxt(path);
else
    rec=VOCreadrecxml(path);
end
