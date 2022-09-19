function indMap = bnm_convexH(doc,layerfolder)

if istable(doc)
    T = doc;
else
    T = readtable(doc);
end

if isstring(layerfolder) || ischar(layerfolder)
    temp = dir(layerfolder);
    layers = {temp.name};
    comp = strncmp('bio',layers,3);
    layers = layers(comp);
    disp('----Reading layers----')

    [Z,R] = arcgridread(strcat(layerfolder,layers{1}));
    
    N = length(layers);
    Z = zeros(size(Z,1),size(Z,2),N);

    for i = 1:N
%        progressbar(i/N)
       Z(:, :, i) = arcgridread(strcat(layerfolder,layers{i}));
       eval(strcat('T.bio',num2str(i),'=ltln2val(Z(:,:,i),R,[T.LAT],[T.LONG]);'))
    end
    toc
else
    
    Z = layerfolder.Z;
    R = layerfolder.R;
    N = size(Z,3);
    
    for i = 1:N
%        progressbar(i/N)
       eval(strcat('T.bio', num2str(i), '=ltln2val(Z(:,:,i),R,[T.LAT],[T.LONG]);'))
    end
end

%T_Or = T;
T = T{:,4:end};
T=T';

    indMap = Z(:,:,1);
    index = ~isnan(indMap(:));
    dim = size(Z);
    tsize = sum(index);
    N = round(tsize);
    fmap = zeros(dim(1)*dim(2),dim(3));
    index = find(index>0);
    for i=1:dim(3)
        aux = Z(:,:,i);
        fmap(:,i)=aux(:);
    end
    warning('off')
    for i = 1:N
        point = fmap(index(i),:);
        b = regress(point',T);
        flag = max(abs(b))>1;
        if flag
            indMap(index(i)) = 0;
        else
            indMap(index(i)) = 1;
        end
    end
    warning('on')
    geoshow(indMap, R, 'DisplayType', 'surface')

end