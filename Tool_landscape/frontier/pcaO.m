function [pin, Tclas, coeff]=pcaO(T,Z,R,indicators,vars,show,outlier,outlier2)
reps=size(Z);
caps=reps(3);
template=Z(:,:,1);
data=NaN(length(template(:)),caps);
out1=[];
out2=[];
for i=1:caps
    template=Z(:,:,i);
    data(:,i)=template(:);
end
temp = T{:,4:end};
viable=~isnan(data(:,i));
normalizers=[max(temp(:,:));min(temp(:,:))];
temp(:,:)=(temp(:,:)-normalizers(2,:))./(normalizers(1,:)-normalizers(2,:));

if outlier
    [~,~,RD,chi_crt]=DetectMultVarOutliers(temp(:,:));
    id_out=RD>chi_crt(4);
    out1=temp{id_out,:};
    temp=temp(~id_out,:);
end
[coeff,~,~,~,explained]=pca(temp(:,:));
pin=data{:,:}*coeff(:,1:3);

if ~isempty(out1)
    out1=out1*coeff(:,1:3);
end


if outlier2
    %siz=round(size(pin,1)*0.3);
    [~,~,RD,chi_crt]=DetectMultVarOutliers(pin);
    id_out=RD>chi_crt(4);
    out2=pin(id_out,:);
    pin=pin(~id_out,:);
end

nodes = boundary(pin(:,1),pin(:,2),pin(:,3),1);
boundPoints = unique(nodes)

outT=[out1;out2];
if isempty(outT)
    grap=false;
else
    grap=true;
end