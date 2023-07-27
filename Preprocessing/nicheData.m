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
        nData
        nObs
        nVars
        inds
        varNames
        clusters
        agents
        RAgents
        Graph
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

        function auxmap = predictMap(obj,method,Layers,land)
            if nargin<4
                land = false;
            end
            map = Layers.Z;
            switch method
                case 1
                    [rows,cols,layers] = size(map);
                    vecmap = nan(rows*cols,layers);
                    for i=1:layers
                        auxmap = map(:,:,i);
                        vecmap(:,i) = auxmap(:);
                    end
                    out = predictModel(obj,method,vecmap,land);
                    auxmap(:) = out;
                otherwise
            end
            figure(1)
            clf
            geoshow(auxmap, Layers.R, 'DisplayType', 'surface');
            contourcmap('jet',0 : 0.05 : 1, 'colorbar', 'on', 'location', 'vertical')
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
        function obj=createGraph(obj,full)
            agent=obj.agents;
            Data = obj.nData;
            if ~full
                Distances = squareform(pdist(Data(:,agent),'minkowski',1));
            else
                lAgents = length(agent);
                Data=DataWithErrors(obj,Data);
                Data(:,lAgents+1:end)=preProcFun(obj,Data(:,lAgents+1:end));
                Distances = squareform(pdist(Data,'minkowski',1));
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
        end

        function [Kur,G] = ricciOllivierC(obj,alpha,method)
            G = obj.Graph;
            switch method
                case 'sinkhorn'
                    nNodes = height(G.Nodes);
                    G.Edges.Weight = G.Edges.Weight/sum(G.Edges.Weight)*height(G.Edges.Weight);
                    Adj = full(adjacency(G,'weighted'));
                    reg=ones(1,nNodes)*Adj;
                    Ad = alpha*eye(nNodes)+ (1-alpha)*Adj./reg;
                    M = distances(G);
                    lambda=100;
                    % the matrix to be scaled.
                    Kur = zeros(size(Ad));
                    disp(' ');
                    disp(['Computing ',num2str(nNodes),' curvatures']);
                    for i=1:(nNodes-1)
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
                            continue
                        end
                        K=exp(-lambda*M(adIndex3,adIndex3));
                        % in practical situations it might be a good idea to do the following:
                        K(K<1e-100)=1e-100;
                        % pre-compute matrix U, the Schur product of K and M.
                        U=K.*M(adIndex3,adIndex3);
                        [D,lBound,~,~]=sinkhornTransport(A,B,K,U,lambda,[],[],[],[],0); % running with VERBOSE
                        Kur(i,adIndex) = 1-D./M(i,adIndex);
                        if mod(i,20)==0
                            disp(num2str(round(i/(nNodes-1)*100,1))+"%")
                        end

                    end
                    disp('Done computing curvatures');
                    disp(' ');
                    
                otherwise
            end
        end

        function obj = ricciFlow(obj,alpha,method,epsilon,flowType,iters)
            G = obj.Graph;
            figure(1)
            clf
            p = plot(G,'Marker','o','NodeColor','r','MarkerSize',3,'Layout','force');
            layout(p,'force','WeightEffect','direct');
            title('Iteration 0')
            drawnow
            for i = 1:iters
                [Kur,G] = ricciOllivierC(obj,alpha,method);
                auxG = graph(Kur,'upper');
                switch flowType
                    case 1
                        G.Edges.Weight = G.Edges.Weight - epsilon*(auxG.Edges.Weight).*G.Edges.Weight;
                    case 2
                        G.Edges.Weight = G.Edges.Weight + (auxG.Edges.Weight./(abs(auxG.Edges.Weight)+epsilon)).*G.Edges.Weight;
                    case 3
                        G=rmedge(G,find((auxG.Edges.Weight>0.3)+(auxG.Edges.Weight<-0.3)));
                    otherwise
                end
                obj.Graph = G;
                figure(1)
                clf
                p = plot(G,'Marker','o','NodeColor','r','MarkerSize',3,'Layout','force');
                layout(p,'force','WeightEffect','direct');
                title("Iteration "+num2str(i))
                drawnow
            end

        end

    end
 end