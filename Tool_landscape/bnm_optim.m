function in = bnm_optim(in,show,parallel,solver,reps)

Tdata=in.Tdata;
Z=in.Z;
R=in.R;
indicators=in.Indicators;
vars=in.Vars;
T2=in.T2;
N=length(indicators)+length(vars);
method=in.Method;

if solver==1
    if nargin<5
        reps=100;
    end
    x=zeros(reps,N);
    res=zeros(1,reps);
    limit=10;
    parfor j=1:reps
        init=rand(1,N)*limit;
        opt=optimoptions('fmincon','Display','iter','UseParallel',parallel);
        [x(j,:),res(j)] = fmincon(@(coeff) optimizer(Tdata,Z,R,indicators,vars,T2,false,method,coeff),init,[],[],[],[],zeros(1,N),ones(1,N)*limit,[],opt);
    end
    [~,idx]=sort(res,'ascend');
    x=x(idx(1),:);
    res=res(idx(1));
else
opt=optimoptions('particleswarm','Display','iter','UseParallel',parallel);
[x,res] = particleswarm(@(coeff) optimizer(Tdata,Z,R,indicators,vars,T2,false,method,coeff),N,zeros(1,N),ones(1,N)*10,opt);
end

in.Coeff=x;
in.Optim=res;

if show
    predictor2(Tdata,Z,R,indicators,vars,T2,true,method,x);
    figure('Name','OptimizedIndicators')
    clf
    siz=size(indicators,2);
    D1 = floor(sqrt(siz)); % Number of rows of subplot
    D2 = D1+ceil((siz-D1^2)/D1);
    for i=1:siz
        subplot(D1,D2,i)
        plot(Tdata{:,i}(:,1),Tdata{:,i}(:,2).^x(i),'LineWidth',2)
        hold on
        plot(Tdata{:,i}(:,1),Tdata{:,i}(:,2),'LineWidth',2)
        legend({'Optimized','Initial'})
        title(strcat('bio',num2str(indicators(i))))
    end
    figure('Name','OptimizedVars')
    clf
    siz=size(vars,2);
    D1 = floor(sqrt(siz)); % Number of rows of subplot
    D2 = D1+ceil((siz-D1^2)/D1);
    for j=1:siz
        subplot(D1,D2,j)
        plot(Tdata{:,i+j}(:,1),Tdata{:,i+j}(:,2).^x(i+j),'LineWidth',2)
        hold on
        plot(Tdata{:,i+j}(:,1),Tdata{:,i+j}(:,2),'LineWidth',2)
        legend({'Optimized','Initial'})
        title(strcat('var',num2str(vars(j))))
    end
end

end