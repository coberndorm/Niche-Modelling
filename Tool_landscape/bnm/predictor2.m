function [map, response]=predictor2(Tdata, Z, indicators, vars, coeff)

stot = length(indicators);
tot = stot + length(vars);

if isempty(coeff)
    coeff = ones(1,tot);
end

reps = size(Z);
map = ones(reps(1), reps(2));
caps = reps(3);
template = Z(:, :, 1);
data = NaN(length(template(:)), caps);

for i = 1 : caps
    template = Z(:, :, i);
    data(:, i) = template(:);
end

nanDetector = sum(data, 2);
pointer = ~isnan(nanDetector);
response = NaN(length(template(:)), tot);

i = 1;

for j = indicators
    xdata = Tdata{:, i}(:, 1);
    ydata = Tdata{:, i}(:, 2);
    response(pointer, i) = interp1(xdata, ydata.^(coeff(i)), data(pointer, j), 'pchip', 0);
    i = i + 1;
end


for j = vars
    xdata = Tdata{:, i}(:, 1);
    ydata = Tdata{:, i}(:, 2);
    
    response(pointer,i) = interp1(xdata, ydata.^(coeff(i)), ...
        eval(strcat("Tdata.Properties.CustomProperties.m", num2str(j), "(data(pointer,:));")), 'pchip', 0);
    i = i + 1;
end

final = (prod(response, 2));
final(final < 0) = 0;
final = final.^(1/sum(coeff(1 : stot)));

% final=(abs(prod(response,2))).^(1/sum(coeff(1:stot)));
map(:) = final(:);

end
    

