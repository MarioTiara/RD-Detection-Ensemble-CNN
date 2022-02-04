% Local functions called from getPropertySet()
%
function toggle(this,zmode)
% If we are toggling the current mode, then turn it off.  Otherwise, set
% the current mode to what we are toggling.
if strcmpi( zmode, this.ZoomMode )
    this.ZoomMode = 'off';
else
    this.ZoomMode = zmode;
end

end
