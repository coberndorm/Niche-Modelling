function in = bnm_modeling2(in,show,outlier,outlier2)
if nargin <3
    outlier=false;
    outlier2=false;
end

T2=in.T2;
indicators=in.Indicators;
Tdata=in.Tdata;
Z=in.Z;
R=in.R;
vars=in.Vars;

%T=T2(:,4:end);
T=array2table(unique(T2{:,4:end},'rows'));
Tpca=table();

for i=indicators
    eval(strcat("Tpca.bio",num2str(i),"=T{:,i};"))
end
for i=vars
    eval(strcat("Tpca.var",num2str(i),"=Tdata.Properties.CustomProperties.m",num2str(i),"(T{:,:});"))
end
Tpca.Properties.CustomProperties=Tdata.Properties.CustomProperties;

[map3,score]=predictpca(Tpca,Z,R,indicators,vars,show,outlier,outlier2);

in.MapPca=map3;
in.ScorePca=score;

end