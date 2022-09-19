function classifiers=pcaO(in,show,outlier,outlier2)
if nargin <3
    outlier=false;
    outlier2=false;
end
out1=[];
out2=[];
T2 = in{1}.T2;
Z = in{1}.Z;
R = in{1}.R;
points = T2{:,4:end};
normalizers=[max(points(:,:));min(points(:,:))];
points(:,:)=(points(:,:)-normalizers(2,:))./(normalizers(1,:)-normalizers(2,:));

if outlier
    [~,~,RD,chi_crt]=DetectMultVarOutliers(points(:,:));
    id_out=RD>chi_crt(4);
    out1=points{id_out,:};
    points=points(~id_out,:);
end

[coeff,~,~,~,explained]=pca(points(:,:));
pin=points(:,:)*coeff(:,1:3);

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

boundPointsIndex = unique(nodes)'
boundPoints = points(boundPointsIndex,:);
pointsSize = length(boundPointsIndex);
samples = length(points);
radius = zeros(pointsSize,samples);

for i=1:pointsSize
    for j=1:samples
        radius(i,j)=norm(points(boundPointsIndex(i),:)-points(j,:));
    end
end
radius = radius(:,setdiff(1:end,boundPointsIndex))
radiusClass = min(radius)

classifiers.nodes = boundPoints;
classifiers.index = boundPointsIndex;
classifiers.radius = radiusClass;

outT=[out1;out2];

if isempty(outT)
    grap=false;
else
    grap=true;
end