function inFinal = bnm_modeling(in, layerfolder, show, method, parallel,outland)
    tic
        if nargin < 4
            method = 4;
        end

        if nargin < 5
            parallel = false;
        end

         if nargin < 6
            outland = [];
        end

        if parallel
            parforArg = inf;
        else
            parforArg = 0;
        end

    if ~isempty(layerfolder)

        disp('----Reading layers----')
        temp = dir(layerfolder);
        layers = {temp.name};
        comp = strncmp('bio', layers, 3);
        layers = layers(comp);

        [Z,R] = readgeoraster(strcat(layerfolder, layers{1}),'CoordinateSystemType','geographic');

        N = length(layers);
        dimensions=size(Z);
        Z = zeros(dimensions(1), dimensions(1), N);

        parfor (i = 1:N, parforArg)
           Z(:, :, i) = readgeoraster(strcat(layerfolder, layers{i}),'CoordinateSystemType','geographic');
        end
        toc
    else
        Z = in{1}.Z;
        R = in{1}.R;
    end
    dimensions=size(Z);
    try
        K = in{1}.T2.Properties.CustomProperties.NumClusters;
    catch
        K = 1;
    end
    maps=zeros(dimensions(1),dimensions(2),K);
    disp('----Modeling----')

    for ij = 1 : K
        ij
        disp("-Model "+num2str(ij))
        Tdata = in{ij}.Tdata;
        indicators = in{ij}.Indicators;
        vars = in{ij}.Vars;
        T2 = in{ij}.T2;
        maps(:,:,ij) = predictor2(Tdata, Z, indicators, vars, []);
    end
     map = max(maps,[],3);
     
     if ~isempty(outland)
         outmap = predictor2(outland.Tdata,Z,outland.Indicators,outland.Vars,[]);
         param = 0.01;
         map2 = (1+param)*map(:)./(param+outmap(:));
%          nanmap = isnan(map2);
%          cleanMap = rmoutliers(map2(~nanmap),'percentiles',[0,99.99]);
%          indMax = max(cleanMap);
%          map2(map2>indMax) = indMax;
         map(:) = normalize(map2,'range',[0,max(map(:))]);
     end
     
    [minimize, roc] = curverock(map, R, T2, show, method);
    
    inFinal.Map = map;
    inFinal.Minimize = minimize;
    inFinal.Method = method;
    inFinal.Roc = roc;
    inFinal.Z = Z;
    inFinal.R = R;
    toc
    
    if show
    
    figure
    clf
    geoshow(map, R, 'DisplayType', 'surface');
    axis off
    contourcmap('jet', 0 : 0.05 : 1, 'colorbar', 'on', 'location', 'vertical')
    for ij = 1:K
        geoshow(in{ij}.T2.LAT, in{ij}.T2.LONG, 'DisplayType', 'Point', 'Marker', 'o', ...
            'MarkerSize', 5, 'MarkerFaceColor', [.95 .9 .8], 'MarkerEdgeColor', ...
            'black','Zdata', 2*ones(length(in{ij}.T2.LONG),1));
    end
    axis off
    
    figure
    clf
    geoshow(map, R, 'DisplayType','surface');
    contourcmap('jet', 0 : 0.05 : 1, 'colorbar', 'on', 'location', 'vertical')
    axis off
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     
%     D1 = 2; % Number of rows of subplot
%     D2 = 2;
%     figure('Name','Map gradient')
%     clf
%     count = 1;
%     
%         for i = 0.2:0.2:0.8
% 
%             map2 = map;
%             map2(map2 < i) = 0;
% 
%             cont = map2(:);
%             cont = cont(~isnan(cont));
% 
%             indice = sum(cont > 0)/length(cont)*100;
% 
%             subplot(D1, D2, count)
%             geoshow(map2, R, 'DisplayType', 'surface')
%             contourcmap('jet', 0:0.05:1, 'colorbar', 'on', 'location', 'vertical')
%             title(strcat('Score: ', num2str(round(indice, 2)), '%'))
%             axis off
%             count = count + 1;
% 
%         end
    end
%     


end