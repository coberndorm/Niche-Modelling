classdef nicheData

    properties (Hidden)
        normax
        normin
        normMethod
        funDat
        normaxE
        norminE
        geomModel
        landscapeModel
    end
    properties
        orData
        nData
        nObs
        nVars
        inds
        varNames
        clusters
        agents
        RAgents
        Graph
        pMink
    end

    methods

        function obj = nicheData(Data,method,inds) %create basic Niche
            wData=Data{:,inds};
            cleaner = isnan(sum(wData,2));
            wData(cleaner,:)=[];
            [obj.nObs,obj.nVars]=size(wData);     
            obj.normMethod = method;
            switch method
                case 1
                    obj.normax=max(wData,[],1);
                    obj.normin=min(wData,[],1);
                    dataCandidate = normalizer(obj,wData);
                    [~,~,RD,chi_crt]=DetectMultVarOutliers(dataCandidate);
                    id_in=RD<chi_crt(1);
                    disp(num2str(sum(id_in)+" outlier observations"))
                    obj.normax=max(wData(id_in,:),[],1);
                    obj.normin=min(wData(id_in,:),[],1);
                    obj.nData=normalizer(obj,wData);
                    
                case 2
                    obj.normax=mean(wData,1);
                    obj.normin=std(wData,1);
                    obj.nData=normalizer(obj,wData);
                case 3
                    obj.normax=median(wData,1);
                    obj.normin=mad(wData,1);
                    obj.normMethod = 2;
                    obj.nData=normalizer(obj,wData);
                otherwise
                    disp("Select an implemented method")
                    return
            end
            obj.orData=wData;
            obj.varNames = Data.Properties.VariableNames(inds);   
            obj.inds = inds;
        end

        %Data preprocessing functions

        function obj = aClus(obj,thr)
            [obj.clusters,obj.agents,obj.RAgents] = corClus(obj,thr);
        end

        function nData = normalizer(obj,Data)
            method = obj.normMethod;            
            switch method
                case 1
                    nData = (Data-obj.normin)./(obj.normax-obj.normin);
                case 2
                    nData = (Data-obj.normax)./obj.normin;
                otherwise
            end
        end

        function oData = inormalizer(obj,Data)
            method = obj.normMethod;
             switch method
                case 1
                    oData = Data.*(obj.normax-obj.normin) + obj.normin;
                case 2
                    oData = Data.*(obj.normin) + obj.normax;
                otherwise
            end
        end

        function [indRelated,bestReg,realAgents] = corClus(obj,thr)
            Data = obj.nData;
            correlation = corr(Data);
            absCorrelation = abs(correlation);
            corRelated = absCorrelation>thr;
            vars = 1:obj.nVars;
            bestReg = [];

            while ~isempty(vars)
                pointer = vars(1);
                actRel = corRelated(pointer,:);
                related = find(actRel);
                
                if sum(actRel)>1
                    warning('off','all')

                    try
                        covM = cov(Data(related,related));
                        covMi = inv(covM);
                        D1 = diag(covM);
                        D2 = diag(covMi);
                        bestRegCand = 1-1./(D1.*D2);
                        [~,index] = sort(bestRegCand,'descend');
                        bestReg = union(bestReg,related(index(1)));
                    catch
                        bestReg = union(bestReg,pointer);
                    end

                    warning('on','all')
                else
                    bestReg = union(bestReg,pointer);
                end

                vars = setdiff(vars,related);
            end

            indRelated = corRelated(bestReg,:);
            realAgents = sum(indRelated,2)>1;
        end

        function obj = regressor(obj,show)
            if nargin<2
                show=true;
            end
            Data = obj.nData;
            agent = obj.agents;
            cluster = obj.clusters;
            obs = obj.nObs;
            lAgents=length(agent);
            lCLuster=sum(cluster,2);
            nClusters = sum(obj.RAgents);
            D1 = floor(sqrt(lAgents)); % Number of rows of subplot
            D2 = D1+ceil((lAgents-D1^2)/D1); % Number of columns of subplot
            intercept = ones(obs,1);   
            funData = zeros(obj.nVars+1,lAgents+nClusters);
            funCluster = logical([ones(lAgents,1),cluster]);
            clusCounter = 1;
            
            if show
                figure(1)
                clf
            end
            for i=1:lAgents
                nagent =agent(i);
                switch lCLuster(i)
                    case 1 
                        ydat = Data(:,nagent);
                        if show
                            subplot(D1,D2,i)
                            histfit(ydat);
                            xlabel('Observed Response');
                            ylabel('Fitted Response');
                            legend({'Data'},  ...
	                            'location','NW');
                            title("Agent "+num2str(nagent))
                        end
                        funData(nagent+1,i) = 1;
                    case 2
                        ydat = Data(:,agent(i));
                        cluster(i,agent(i))=0;
                        funCluster(i,agent(i)+1)=0;
                        xdat = Data(:,cluster(i,:));
                        mdl=fitlm(ydat,xdat);
                        beta=mdl.Coefficients.Estimate;
                        yfit=[intercept xdat]*beta;
                        if show
                            subplot(D1,D2,i)
                            plot(ydat,yfit,'r^');
                            xlabel('Observed Response');
                            ylabel('Fitted Response');
                            legend({'Linear Regression'},  ...
	                            'location','NW');
                            title("Cluster "+num2str(agent(i)))
                        end
                        funData(nagent+1,[i,clusCounter+lAgents]) = 1;
                        funData(funCluster(i,:),clusCounter+lAgents) = -beta;
                        clusCounter = clusCounter+1;

                    otherwise
                        ydat = Data(:,agent(i));
                        cluster(i,agent(i))=0;
                        funCluster(i,agent(i)+1)=0;
                        xdat = Data(:,cluster(i,:));
                        [PCALoadings,PCAScores] = pca(xdat);
                        betaPCR = regress(ydat-mean(ydat), PCAScores(:,1:2));
                        betaPCR = PCALoadings(:,1:2)*betaPCR;
                        beta = [mean(ydat) - mean(xdat)*betaPCR; betaPCR];
                        yfitPCR = [intercept xdat]*beta;
                        if show
                            subplot(D1,D2,i)
                            plot(ydat,yfitPCR,'r^');
                            xlabel('Observed Response');
                            ylabel('Fitted Response');
                            legend({'PCR with 2 Components'},  ...
	                            'location','NW');
                            title("Cluster "+num2str(agent(i)))
                        end
                        funData(nagent+1,[i,clusCounter+lAgents]) = 1;
                        funData(funCluster(i,:),clusCounter+lAgents) = -beta;
                        clusCounter = clusCounter+1;

                end
            end
            obj.funDat = funData;
            nameClusters(obj)

        end

        function out = DataWithErrors(obj,Data)
            sData = size(Data,1);
            Data=[ones(sData,1),Data];
            out = Data*obj.funDat;
        end

        function obj = setProcFun(obj)
            inData = DataWithErrors(obj,obj.nData);
            nAgents = length(obj.agents);
            method = obj.normMethod;
            errData = inData(:,nAgents+1:end);
            switch method
                case 1
                    obj.normaxE=max(errData,[],1);
                    obj.norminE=min(errData,[],1);
   
                case 2
                    obj.normaxE=mean(errData,1);
                    obj.norminE=std(errData,1);
                    %obj.nData=normalizer(obj,wData);
                otherwise
            end
        end

        function out = preProcFun(obj,Data)
            method = obj.normMethod;
            switch method
                case 1
                    out = (Data-obj.norminE)./(obj.normaxE-obj.norminE);
                case 2
                    out = (Data-obj.normaxE)./obj.norminE;
                otherwise
            end
        end

        function out = ipreProcFun(obj,Data)
            method = obj.normMethod;
            switch method
                case 1
                    out = Data.*(obj.normaxE-obj.norminE) + obj.norminE;
                case 2
                    out = Data.*(obj.norminE) + obj.normaxE;
                otherwise
            end
        end

        function Data = procFun(obj,Data)
            lAgents = length(obj.agents);
            Data=normalizer(obj,Data);
            Data=DataWithErrors(obj,Data);
            Data(:,lAgents+1:end)=preProcFun(obj,Data(:,lAgents+1:end));
        end

        %modeling functions
        %basic modeling

        function obj = nicheModel(obj,method)
            switch method
                case 1
                    agent = obj.agents;
                    obs = obj.nObs;
                    lAgents=length(agent);
                    oData = obj.nData;
                    Data=DataWithErrors(obj,oData);
                    Data(:,lAgents+1:end)=preProcFun(obj,Data(:,lAgents+1:end));
                    cols = size(Data,2);   
                    if obs>5e3
                        npoints = ceil(5e3 + sqrt(obs));
                    else
                        npoints = obs; 
                    end
                    model = zeros(npoints,cols,2);
                    for i = 1:cols
                        points = datasample(Data(:,i),npoints);
                        model(:,i,1) = points;
                        funcKernel = ksdensity(Data(:,i),points);
                        model(:,i,2) = normalize(funcKernel, 'range');
                    end
                    obj.geomModel = model;
                case 2
                    
                otherwise
            end
        end

        function out = predictModel(obj,method,Data,land)
            if nargin <4
                land = false;
            end
            switch method
                case 1
                    oData = Data;
                    Data = obj.procFun(Data);
                    model = obj.geomModel;
                    nanind = ~isnan(sum(Data,2));
                    wData = Data(nanind,:);
                    cols = size(wData,2);
                    response = Data;
                    agent = obj.agents;
                    lAgents=length(agent);
                    for i = 1:cols
                        [~,xind] = unique(model(:,i,1));
                        response(nanind,i) = interp1(model(xind,i,1), model(xind,i,2), wData(:,i),'linear',0);
                    end
                    out = prod(response,2).^(1/lAgents);

                    if land
                        nicheL = obj.landscapeModel;
                        DataL = nicheL.procFun(oData);
                        modelL = nicheL.geomModel;
                        nanindL = ~isnan(sum(DataL,2));
                        wDataL = DataL(nanindL,:);
                        colsL = size(wDataL,2);
                        responseL = DataL;
                        agentL = nicheL.agents;
                        lAgentsL=length(agentL);
                        for i = 1:colsL
                            [~,xind] = unique(modelL(:,i,1));
                            responseL(nanind,i) = interp1(modelL(xind,i,1), modelL(xind,i,2), wDataL(:,i),'linear',0);
                        end
                        alpha = 1e-8;
                        outL=(prod(responseL,2).^(1/lAgentsL))+alpha;
                        auxOut = out./outL;
                        while any(auxOut(nanind)>1)
                            [~,mInd]=max(auxOut);
                            alpha = out(mInd)-outL(mInd);
                            outLAux = outL+alpha;
                            auxOut = out./outLAux;
                        end
                        out = out./(outL+alpha);
                    end

                otherwise
            end
        end

        function auxmap = predictMap(obj,method,Layers,show,land)
            if nargin<5
                land = false;
            end
            map = Layers.Z;
            [rows,cols,layers] = size(map);
            vecmap = nan(rows*cols,layers);
            for i=1:layers
                auxmap = map(:,:,i);
                vecmap(:,i) = auxmap(:);
            end
            switch method
                case 1 
                    out = predictModel(obj,method,vecmap,land);
                    auxmap(:) = out;
                case 2
                    out = heatClassifier(obj,vecmap);
                    auxmap(:) = out;
                otherwise
            end
            if show
                figure(2)
                clf
                geoshow(auxmap, Layers.R, 'DisplayType', 'surface');
                contourcmap('jet',0 : 0.05 : 1, 'colorbar', 'on', 'location', 'vertical')
            end
        end

        function [vecmap,nanind] = map2vec(~,map)
            [rows,cols,layers] = size(map);
            vecmap = nan(rows*cols,layers);
            for i=1:layers
                auxmap = map(:,:,i);
                vecmap(:,i) = auxmap(:);
            end
            nanind = ~isnan(sum(vecmap,2));
            vecmap = vecmap(nanind,:);
        end
        function nameClusters(obj)
            agent = obj.agents;
            cluster = obj.clusters;
            lAgents=length(agent);
            for i=1:lAgents
                disp("Cluster for Agent "+num2str(agent(i))+ " is:")
                disp(find(cluster(i,:)))
            end
        end

        function obj = insertLand(obj,Data,method,inds,thr,modelM)
            inMap = map2vec(obj,Data);
            Data=array2table(inMap);
            nicheLand = nicheData(Data,method,inds);
            nicheLand=nicheLand.aClus(thr);
            nicheLand=nicheLand.regressor;
            nicheLand=nicheLand.setProcFun();
            nicheLand=nicheLand.nicheModel(modelM);
            obj.landscapeModel = nicheLand;
        end

        %diffusion modeling

        %functions for graphs
        function obj=createGraph(obj,full,pMink)
            agent=obj.agents;
            Data = obj.nData;
            if ~full
                Distances = squareform(pdist(Data(:,agent),'minkowski',pMink));
            else
                lAgents = length(agent);
                Data=DataWithErrors(obj,Data);
                Data(:,lAgents+1:end)=preProcFun(obj,Data(:,lAgents+1:end));
                Distances = squareform(pdist(Data,'minkowski',pMink));
            end
            G = graph(Distances);
            figure(2)
            clf
            p = plot(G,'Marker','o','NodeColor','r','MarkerSize',3,'Layout','force');
            layout(p,'force','WeightEffect','direct');
%             L = laplacian(G);
%             [v, d] = eigs(L,4,'sa');
%             gplot3(Distances,v(:,2:4))
            obj.Graph=G;
            obj.pMink = pMink;
        end

        function [Kur,G] = ricciOllivierC(obj,alpha,method,parforArg)
            G = obj.Graph;
            switch method
                case 'sinkhorn'
                    nNodes = height(G.Nodes);
                    G.Edges.Weight = G.Edges.Weight/sum(G.Edges.Weight)*height(G.Edges.Weight);
                    Adj = full(adjacency(invertGraph(obj,G),'weighted'));
                    %Adj = full(adjacency(G,'weighted'));
                    reg=ones(1,nNodes)*Adj;
                    Ad = alpha*eye(nNodes)+ (1-alpha)*Adj./reg;
                    M = distances(G);
                    lambda=100;
                    % the matrix to be scaled.
                    Kur = zeros(size(Ad));
                    disp(' ');
                    disp(['Computing ',num2str(nNodes),' curvatures']);
                    parfor (i=1:(nNodes-1),parforArg)
                        Kur(i,:) = ParSinkhornTransport(obj,Adj,Ad,Kur(i,:),M,lambda,i);
                        if mod(i,20)==0
                            disp(num2str(round(i/(nNodes-1)*100,1))+"%")
                        end

                    end
                    disp('Done computing curvatures');
                    disp(' ');
                case 'isinkhorn'
                    nNodes = height(G.Nodes);
                    G.Edges.Weight = G.Edges.Weight/sum(G.Edges.Weight)*height(G.Edges.Weight);
                    Adj = full(adjacency(invertGraph(obj,G),'weighted'));
                    %Adj = full(adjacency(G,'weighted'));
                    reg=ones(1,nNodes)*Adj;
                    Ad = alpha*eye(nNodes)+ (1-alpha)*Adj./reg;
                    M = distances(G);
                    lambda=100;
                    % the matrix to be scaled.
                    Kur = zeros(size(Ad));
                    disp(' ');
                    disp(['Computing ',num2str(nNodes),' curvatures']);
                    parfor (i=1:(nNodes-1),parforArg)
                    %for i=1:(nNodes-1)
                        [~,~,~,Kur(i,:)] = ParSinkhornTransport(obj,Adj,Ad,Kur(i,:),M,lambda,i);
                        if mod(i,20)==0
                            disp(num2str(round(i/(nNodes-1)*100,1))+"%")
                        end

                    end
                    disp('Done computing curvatures');
                    disp(' ');
                otherwise
            end
        end

        function [Kur,lBound,uBound,Curv] = ParSinkhornTransport(~,Adj,Ad,Kur,M,lambda,i)
            adIndex = Adj(:,i)>0;
            adIndex2 = Ad(:,i)>0;
            if i>2
                adIndex(1:(i-1))=false;
            end
            B = Ad(:,adIndex2);
            adIndex3 = sum(B,2)>0;
            A = Ad(adIndex3,i);
            B = Ad(adIndex3,adIndex);

            if sum(adIndex)==0
                lBound = Kur;
                uBound = Kur;
                Curv = Kur;
                return
            end
            K=exp(-lambda*M(adIndex3,adIndex3));
            % in practical situations it might be a good idea to do the following:
            K(K<1e-100)=1e-100;
            % pre-compute matrix U, the Schur product of K and M.
            U=K.*M(adIndex3,adIndex3);
            [D,lBound,uBound,~]=sinkhornTransport(A,B,K,U,lambda,[],[],[],[],0); % running with VERBOSE
            Kur(adIndex) = 1-D./M(i,adIndex);
            if nargout>3
                Curv = Kur;
                Curv(adIndex) = 1 - M(i,adIndex)./D;
            end

        end

        

        function [obj,surG] = ricciFlow(obj,alpha,method,epsilon,flowType,iters,steps,percent,parforArg,show)
            G = obj.Graph;
            or_G = G;
            if strcmp(method,'iGraph')
                G = invertGraph(obj,G);
                obj.Graph = G;
                method = 'sinkhorn';
            end
            if show
                figure(1)
                clf
                subplot(1,2,1)
                p = plot(G,'Marker','o','NodeColor','r','MarkerSize',3,'Layout','force');
                layout(p,'force','WeightEffect','direct');
                title("Iteration 0")
                subplot(1,2,2)
                histogram(G.Edges.Weight)
                drawnow
            end
            
            for i = 1:iters
                if flowType ~= 4
                    [Kur,G] = ricciOllivierC(obj,alpha,method,parforArg);
                    auxG = G;
                    for j = 1:height(auxG)
                        auxG.Edges.Weight(j) = Kur(auxG.Edges.EndNodes(j,1),auxG.Edges.EndNodes(j,2));
                    end
                %auxG = graph(Kur,'upper');
                end
                if epsilon == 1
                    switch flowType
                        case 1
                            normSum = abs(sum(auxG.Edges.Weight.*G.Edges.Weight)/(sum(G.Edges.Weight))*max(abs(auxG.Edges.Weight)));
                        case 2
                            normSum = abs(sum(auxG.Edges.Weight./G.Edges.Weight)/sum(G.Edges.Weight.^(-1)));
                        case 3
                            normSum = sum(auxG.Edges.Weight./G.Edges.Weight);
                        case 5
                            normSum = abs(sum(auxG.Edges.Weight./G.Edges.Weight)/(sum(G.Edges.Weight.^(-1))*max(abs(auxG.Edges.Weight))));
                        otherwise
                    end
                else
                    normSum=0;
                end
                switch flowType
                    case 1
                        G.Edges.Weight = G.Edges.Weight.*(1-0.1*(auxG.Edges.Weight+normSum))+eps;
                        %G.Edges.Weight - epsilon*(auxG.Edges.Weight).*G.Edges.Weight;
                    case {2, 3, 5}
                        G.Edges.Weight = abs(G.Edges.Weight./(1-0.1*(auxG.Edges.Weight+normSum)))+eps;
                    case 4
                        %G=rmedge(G,find((auxG.Edges.Weight>0.3)+(auxG.Edges.Weight<-0.3)));
                    otherwise
                end
                if mod(i,steps) == 0
                    G = ricciSurgery(obj,G,percent);
                end
                obj.Graph = G;
                if show
                    figure(1)
                    clf
                    subplot(1,2,1)
                    colormap jet
                    p = plot(G,'Marker','o','NodeColor','r','MarkerSize',3,'Layout','force','EdgeCData',G.Edges.Weight);
                    try
                        layout(p,'force','WeightEffect','direct');
                    catch
                    end
                
                    % Ad = adjacency(G,"weighted");
                    % sizG = size(Ad,1);
                    % wd = Ad*ones(sizG,1);
                    % D = diag(wd);
                    % L = D - Ad;
                    % [vecs,~]=eigs(L);
                    % plot(G,'XData',vecs(:,1),'YData',vecs(:,2),'MarkerSize',3,'EdgeCData',G.Edges.Weight)
    
                    colorbar
                    title("Iteration "+num2str(i))
                    subplot(1,2,2)
                    histogram(G.Edges.Weight)
                    drawnow
                end
            end
            %Se regresa el grafo original con cirugia
            surG = G;
            or_Ad = adjacency(or_G,"weighted");
            sur_Ad = adjacency(G);
            f_Ad = or_Ad.*sur_Ad;
            G = graph(f_Ad);
            obj.Graph = G;
        end

        function G = invertGraph(~,G)
            Ad = adjacency(G,"weighted");
            %Ad(Ad~=0) = Ad(Ad~=0).^(-1);
            Ad(Ad~=0) = exp(-(Ad(Ad~=0)).^2);
            G = graph(Ad);
        end


        function G3 = ricciSurgery(~,G,percent)
            edTable = G.Edges;
            edTable = sortrows(edTable,"Weight","descend");
            pvalue = prctile(edTable{:,"Weight"},percent);
            pindex = find(edTable{:,"Weight"}<pvalue,1);
            G2=G;
            for i=1:pindex
                G3 = rmedge(G2,edTable{i,"EndNodes"}(1,1),edTable{i,"EndNodes"}(1,2));
                bins = conncomp(G3);
                if sum(bins)~=numnodes(G)
                    G3 =  G2;
                    % isolatedNodes = degree(G3) == 0;
                    % G2 = rmnode(G, find(isolatedNodes));
                else
                    G2 = G3;
                end
            end
        end

        function [stableHeat,vecs] = heatFlow(obj,show)
            G = obj.Graph;
            Ad = adjacency(G,"weighted");
            oAd = Ad;
            sizG = size(Ad,1);
            Ad(Ad~=0) = Ad(Ad~=0).^(-1);
            wd = Ad*ones(sizG,1);
            %owd = oAd*ones(sizG,1);
            stableHeat = wd/sum(wd)*sizG;
            D = diag(wd);
            L = D - oAd;
            [vecs,~]=eigs(L);
            if show
                figure(2)
                clf
                plot(G,'XData',vecs(:,1),'YData',vecs(:,2),'MarkerSize',10,'NodeCData',stableHeat)
                colormap jet
                colorbar
            end
            
        end

        function map = heatClassifier(obj,Data)
            map=Data(:,1);
            oData = obj.orData;
            nanind = ~isnan(sum(Data,2));
            wData = Data(nanind,:);
            wData = obj.procFun(wData);
            GData = obj.procFun(oData);
            MDistances = pdist2(GData,wData,'minkowski',obj.pMink);
            G = obj.Graph;
            Ad = full(adjacency(G,"weighted"));
            MDistances(MDistances>max(Ad,[],2))=0;
            Ad(Ad~=0) = Ad(Ad~=0).^(-1);
            MDistances(MDistances~=0) = MDistances(MDistances~=0).^(-1);
            TMDistances = MDistances+repmat(sum(Ad,2),1,size(MDistances,2));
            TMDistances(end+1,:) = sum(MDistances);
            TMDistances = TMDistances./sum(TMDistances);
            normalizers = [max(TMDistances(1:end-1,:));min(TMDistances(1:end-1,:))];
            cmap=(TMDistances(end,:)-normalizers(2,:))./(normalizers(1,:)-normalizers(2,:));
            cmap(cmap<0)=0;
            map(nanind)=cmap;
        end

        function cmap = isolHeatClass(obj,Data,oData)
            GData=oData;
            wData=Data;
            MDistances = pdist2(GData,wData,'minkowski',obj.pMink);
            G = obj.Graph;
            Ad = full(adjacency(G,"weighted"));
            MDistances(MDistances>max(Ad,[],2))=0;
            Ad(Ad~=0) = Ad(Ad~=0).^(-1);
            MDistances(MDistances~=0) = MDistances(MDistances~=0).^(-1);
            TMDistances = MDistances+repmat(sum(Ad,2),1,size(MDistances,2));
            TMDistances(end+1,:) = sum(MDistances);
            TMDistances = TMDistances./sum(TMDistances);
            normalizers = [max(TMDistances(1:end-1,:));min(TMDistances(1:end-1,:))];
            cmap=(TMDistances(end,:)-normalizers(2,:))./(normalizers(1,:)-normalizers(2,:));
            cmap(cmap<0)=0;
        end

        function metric = mapMetric(~,real,predicted)
            vreal = real(:);
            vpredicted = predicted(:);
            nanind = ~isnan(real);
            vreal = vreal(nanind);
            vpredicted = vpredicted(nanind);

            mutualNicheInd = (vreal + vpredicted)>0;
            vrealM = vreal(mutualNicheInd);
            vpredictedM = vpredicted(mutualNicheInd);
            metric = 1-norm(vrealM-vpredictedM,1)/length(vrealM);
        end

        function dmap = plotMaps(obj,real,Layers,predicted,T)
            
            if nargin<3
                figure("Map")
                clf
                geoshow(auxmap, Layers.R, 'DisplayType', 'surface');
                contourcmap('jet',0 : 0.05 : 1, 'colorbar', 'on', 'location', 'vertical')
            else
                figure('Name',"Maps")
                clf
                subplot(1,3,1)
                geoshow(real, Layers.R, 'DisplayType', 'surface');
                contourcmap('jet',0 : 0.05 : 1, 'colorbar', 'on', 'location', 'vertical')
                title("Real")
                if nargin>4
                    geoshow(T.LAT, T.LONG, 'DisplayType', 'Point', 'Marker', 'o', ...
                    'MarkerSize', 5, 'MarkerFaceColor', [.95 .9 .8], 'MarkerEdgeColor', ...
                    'black', 'Zdata', 2 * ones(length(T.LONG), 1));  
                end
                subplot(1,3,2)
                geoshow(predicted, Layers.R, 'DisplayType', 'surface');
                contourcmap('jet',0 : 0.05 : 1, 'colorbar', 'on', 'location', 'vertical')
                metric = obj.mapMetric(real,predicted);
                title("Predicted "+string(round(metric*100,1))+"%")
                if nargin>4
                    geoshow(T.LAT, T.LONG, 'DisplayType', 'Point', 'Marker', 'o', ...
                    'MarkerSize', 5, 'MarkerFaceColor', [.95 .9 .8], 'MarkerEdgeColor', ...
                    'black', 'Zdata', 2 * ones(length(T.LONG), 1));  
                end
                subplot(1,3,3)
                dmap = real-predicted;
                geoshow(abs(dmap), Layers.R, 'DisplayType', 'surface');
                contourcmap('jet',0 : 0.05 : 1, 'colorbar', 'on', 'location', 'vertical')
                title("|Real - Predicted|")
                if nargin>4
                    geoshow(T.LAT, T.LONG, 'DisplayType', 'Point', 'Marker', 'o', ...
                    'MarkerSize', 5, 'MarkerFaceColor', [.95 .9 .8], 'MarkerEdgeColor', ...
                    'black', 'Zdata', 2 * ones(length(T.LONG), 1));  
                end
            end


        end

        function obj = progressMaps(obj,alpha,method,epsilon,flowType,iters,steps,percent,parforArg,real,Layers,T,kind,filename)
            
            G = obj.Graph;
            predicted = obj.predictMap(2,Layers,false);
            fhandle = progressGraphs(obj,G,predicted,real,Layers,T,kind,0);
            frame = getframe(fhandle);
            im{1} = frame2im(frame);

            for i=1:floor(iters/steps)
                selfIters = steps;
                [obj,surG] = ricciFlow(obj,alpha,method,epsilon,flowType,selfIters,steps,percent,parforArg,false);
                predicted = obj.predictMap(2,Layers,false);
                if kind==1
                    fhandle = progressGraphs(obj,surG,predicted,real,Layers,T,kind,i);
                else
                    fhandle = progressGraphs(obj,obj.Graph,predicted,real,Layers,T,kind,i);
                end
                frame = getframe(fhandle);
                im{i+1} = frame2im(frame);
            end

            for idx = 1:i+1
                [A,map] = rgb2ind(im{idx},256);
                if idx == 1
                    imwrite(A,map,filename,"gif","LoopCount",Inf,"DelayTime",2);
                else
                    imwrite(A,map,filename,"gif","WriteMode","append","DelayTime",2);
                end
            end
    

        end

        function fhandle = progressGraphs(obj,G,predicted,real,Layers,T,kind,i)

            fhandle = figure(1);
            clf

            subplot(1,3,1)
            if kind==1
                colormap jet
                p = plot(G,'Marker','o','NodeColor','r','MarkerSize',3,'Layout','force','EdgeCData',G.Edges.Weight);
                try
                    layout(p,'force','WeightEffect','direct');
                catch
                end
                % Ad = adjacency(G,"weighted");
                % oAd = Ad;
                % sizG = size(Ad,1);
                % Ad(Ad~=0) = Ad(Ad~=0).^(-1);
                % wd = Ad*ones(sizG,1);
                % D = diag(wd);
                % L = D - oAd;
                % [vecs,~]=eigs(L);
                % plot(G,'XData',vecs(:,1),'YData',vecs(:,2),'NodeColor','r','MarkerSize',3,'EdgeCData',G.Edges.Weight)
                % colormap jet
                % colorbar
            else
                [stableHeat,vecs] = obj.heatFlow(false);
                plot(G,'XData',vecs(:,1),'YData',vecs(:,2),'MarkerSize',10,'NodeCData',stableHeat)
                colormap jet
                colorbar
            end
            title('Observations graph')

            subplot(1,3,2)
            geoshow(predicted, Layers.R, 'DisplayType', 'surface');
            contourcmap('jet',0 : 0.05 : 1, 'colorbar', 'on', 'location', 'vertical')
            title("Predicted: "+string(round(obj.mapMetric(predicted,real),2))+" similar")
            geoshow(T.LAT, T.LONG, 'DisplayType', 'Point', 'Marker', 'o', ...
            'MarkerSize', 5, 'MarkerFaceColor', [.95 .9 .8], 'MarkerEdgeColor', ...
            'black', 'Zdata', 2 * ones(length(T.LONG), 1));

            subplot(1,3,3)
            geoshow(real, Layers.R, 'DisplayType', 'surface');
            contourcmap('jet',0 : 0.05 : 1, 'colorbar', 'on', 'location', 'vertical')
            title("Real")
            geoshow(T.LAT, T.LONG, 'DisplayType', 'Point', 'Marker', 'o', ...
            'MarkerSize', 5, 'MarkerFaceColor', [.95 .9 .8], 'MarkerEdgeColor', ...
            'black', 'Zdata', 2 * ones(length(T.LONG), 1));
            sgtitle("Iteration: "+string(i))
            drawnow
        end

        function [mmap,nicheF,roc] = aucModel(obj,alpha,method,epsilon,flowType,iters,steps,percent,parforArg,show,T,Layers,op)
            nicheF=obj;
            pastRoc=-inf;
            newRoc=0;
            while pastRoc<newRoc
                nicheF = nicheF.ricciFlow(alpha,method,epsilon,flowType,iters,steps,percent,parforArg,false);
                mmap = nicheF.predictMap(2,Layers,false);
                [~, roc] = nicheF.aucMetric(mmap,T,Layers,op,false);
                pastRoc=newRoc;
                newRoc=roc;
            end


        end

        function [score, roc] = aucMetric(~,map,T,Layers,op,show)
            R = Layers.R;
            switch op
                case {1,2,4,5}%estos son los casos que merecen mayor atención
                    metric=zeros(1,20);
                    area=metric;
                    counter=0;
                    map = real(map);
                    records=geointerp(map, R, [T.LAT], [T.LONG]);
                    records=records(~isnan(records));
                    runs=0.05:0.05:1;
                    tot=sum(~isnan(map(:)));
                    for i=runs
                        counter=counter+1;
                        vrecords=sum(records>=1-i);
                        metric(counter)=vrecords/length(records);
                        area(counter)=sum(map(:)>1-i)/tot;
                        
                    end
                    switch op
                        case 1
                            score=sum((ones(1,counter)-metric).^2+(area.^2));%área 
                            %entre cada curva y su tope priorizando los valores más
                            %altos
                        case 2
                            score=sum((ones(1,counter)-metric)+area);%área entre cada
                            %curva y su tope
                        case 5
                            score=1-trapz([0,area,1],[0,metric,1]);%score clásico
                        otherwise
                            score=1-0.05*(trapz(metric)-trapz(area));%área entre ambas curvas
                    end
                case 3
                    runs=0.5;
                    map(map<runs)=0;
                    records=geointerp(map, R, [A.latitude], [A.longitude]);
                    records=records(~isnan(records));
                    frecords=sum(records==0);
                    vrecords=length(records)-frecords;
                    metric=vrecords/length(records);
                    area=sum(map(:)>0)/sum(~isnan(map(:)));
                    score=sum((1-metric)+area);
                otherwise
                        disp('Wrong metric selected')
                        return
            end
            
            roc = trapz([0,area,1],[0,metric,1]);
            
            if show && length(runs)>1
                figure('Name','Rock')
                plot([0,area,1],[0,metric,1],[0 1],[0 1],'LineWidth',2)
                xlabel('Map area (%)')
                ylabel('Predicted records (%)')
                legend({'Actual model','Null model'},'Location','best')
                title(strcat('Score: ',num2str(round(trapz([0,area,1],[0,metric,1])*100)),'%'))
                figure('Name','Objetive')
                plot(runs,metric,runs,area,'LineWidth',2)
                xlabel('1-Probability of prescence')
                ylabel('%')
                legend({'Records','Area'},'Location','best')
                figure('Name','Map')
                geoshow(map, Layers.R, 'DisplayType', 'surface');
                contourcmap('jet',0 : 0.05 : 1, 'colorbar', 'on', 'location', 'vertical')
                title("Real")
                geoshow(T.LAT, T.LONG, 'DisplayType', 'Point', 'Marker', 'o', ...
                'MarkerSize', 5, 'MarkerFaceColor', [.95 .9 .8], 'MarkerEdgeColor', ...
                'black', 'Zdata', 2 * ones(length(T.LONG), 1));
         
            end
        end

    end
 end