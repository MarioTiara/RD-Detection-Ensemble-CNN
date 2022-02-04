% Local functions called from getPropertySet()
%
function toggleAutoscale(this,newMode)
if strcmpi( getPropertyValue( this, 'AutoscaleMode' ), newMode )
    newValue = 'Manual';
else
    newValue = newMode;
end
setPropertyValue( this, 'AutoscaleMode', newValue );

end
