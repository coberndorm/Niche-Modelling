function T = bnm_clustering(T, opt)

    %Species classification using spectral clustering
    
    %Setting the clustering options
    if nargin < 2        
        DiffMag = 10;
        %Distance = "mahalanobis";
        Distance = "spearman";
        dim = size(T,1);
        InitialCluster = ceil(10*dim/(200+dim));
        ClassVar = 4:22;
        
    else        
        DiffMag = opt.DiffMag;
        Distance = opt.Distance;
        InitialCluster = opt.InitialCluster;
        ClassVar = opt.ClassVar;   
    end

    X = normalize(T{:, ClassVar});
    dims = size(X);
    if dims(1)>dims(2)
        try
            [~, ~, D_temp] = spectralcluster(X, InitialCluster, 'Distance', Distance,'ClusterMethod','kmedoids');
            % find order of magnitude
            n = floor(log(abs(D_temp))./log(10));
            INF = isinf(n);
            n(INF) = 0;

            % find optmial number of clusters
            cond1=(n <= min(n) + DiffMag);
            cond2=D_temp<=1e-3;
            cond=cond1.*cond2;
            k = find(cond);
            k = length(k) + sum(INF);
        catch
            k=1;
        end

        try
            %Clasiffy each point into a cluster
            [ClusterIndex, ~, ~] = spectralcluster(X, k,'Distance', Distance,'ClusterMethod','kmedoids');
            numElem = sum(ClusterIndex==1:k);
            if any(numElem<10)
                lessers=find(numElem<10);
                numElemS=sort(numElem);
                absorv = numElemS(end);
                absorv = find(numElem==absorv);
                for ik=lessers
                    ClusterIndex(ClusterIndex==ik)=absorv;
                end
                finalindex=unique(ClusterIndex);
                aux = 1;
                for ik = finalindex'
                    ClusterIndex(ClusterIndex==ik)=aux;
                    aux = aux+1;
                end
                k=length(finalindex);
            end

        catch        
           k = 1;
           ClusterIndex = ones(1, length(T.LAT));
           disp("Warning: check the spectralcluster options")
        end
    else
         k = 1;
         ClusterIndex = ones(1, length(T.LAT));
    end
    
    %Save the classification index for each point
    %T = addvars(T, ClusterIndex,'Before','LAT');
    T = addprop(T, {'ClusterIndex', 'NumClusters'}, {'table', 'table'});
    T.Properties.CustomProperties.ClusterIndex = ClusterIndex;
    T.Properties.CustomProperties.NumClusters = k;
    
end