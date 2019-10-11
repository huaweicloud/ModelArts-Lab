function PASerrmsg(PASerr,SYSerr)
  fprintf('Pascal Error Message: %s\n',PASerr);
  fprintf('System Error Message: %s\n',SYSerr);
  k=input('Enter K for keyboard, any other key to continue or ^C to quit ...','s');
  if (~isempty(k)), if (lower(k)=='k'), keyboard; end; end;
  fprintf('\n');
return