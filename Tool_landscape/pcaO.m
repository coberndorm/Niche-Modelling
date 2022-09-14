function [bounds, coeff]=pcaO(in,show,outlier,outlier2)
if nargin <3
    outlier=false;
    outlier2=false;
end
out1=[];
out2=[];
T2 = in.T2;
temp = T2{100:end,4:end};
normalizers=[max(temp(:,:));min(temp(:,:))];
temp(:,:)=(temp(:,:)-normalizers(2,:))./(normalizers(1,:)-normalizers(2,:));

if outlier
    [~,~,RD,chi_crt]=DetectMultVarOutliers(temp(:,:));
    id_out=RD>chi_crt(4);
    out1=temp{id_out,:};
    temp=temp(~id_out,:);
end

[coeff,~,~,~,explained]=pca(temp(:,:));
pin=temp(:,:)*coeff(:,1:3);

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

nodes = boundary(pin(:,1),pin(:,2),pin(:,3),0);

if show
    trisurf(nodes,pin(:,1),pin(:,2),pin(:,3), 'Facecolor','cyan','FaceAlpha',0.8); axis equal;
    hold on
    plot3(pin(:,1),pin(:,2),pin(:,3),'.r')
    hold off
end

boundPointsIndex = unique(nodes)
boundPoints = temp(boundPointsIndex,:)
points = length(boundPointsIndex);
samples = length(temp);
radius = zeros(points,samples);

for i=1:points
    for j=1:samples
        radius(i,j)=norm(temp(boundPointsIndex(i),:)-temp(j,:));
    end
end
radius
classifiers = min(radius)

temp2 = T2{1:100,4:end};
normalizers=[max(temp2(:,:));min(temp2(:,:))];
temp2(:,:)=(temp2(:,:)-normalizers(2,:))./(normalizers(1,:)-normalizers(2,:));
radius2 = zeros(points,length(temp2));
for i=1:points
    for j=1:length(temp2)
        radius2(i,j)=norm(temp(boundPointsIndex(i),:)-temp2(j,:));
    end
end

newpoints = min(radius2)
inBounds = zeros(1,length(temp2));

for i=1:length(newpoints)
    inBounds(i) = sum(newpoints(i)<=classifiers);
end
inBounds./length(classifiers)
outT=[out1;out2];
if isempty(outT)
    grap=false;
else
    grap=true;
end