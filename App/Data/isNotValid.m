% Local functions called from getPropertySet()
%
function b = isNotValid(displayRange)
b = ~isscalar( displayRange ) || ~isa( displayRange, 'double' ) || isnan( displayRange ) || ~isreal( displayRange ) || displayRange>100 || displayRange<1;

end
