function [outputArg1,outputArg2] = newPoints(in,classifiers)
%NEWPOINTS Summary of this function goes here
%   Detailed explanation goes here
T2 = in.T2;
nodes = classifiers.nodes;
classIndex = classifiers.index;
radius = classifiers.radius;
classSize = length(classIndex);
points = T2{:,4:end};

normalizers=[max(points(:,:));min(points(:,:))]; %duda sobre esta normalizacion
points(:,:)=(points(:,:)-normalizers(2,:))./(normalizers(1,:)-normalizers(2,:));
dist = zeros(classSize,length(points));

for i=1:classSize
    for j=1:length(points)
        dist(i,j)=norm(nodes(i,:)-points(j,:));
    end
end

newpoints = min(dist);
inBounds = zeros(1,length(newpoints));

for i=1:length(newpoints)
    inBounds(i) = sum(newpoints(i)<=radius);
end

inBounds
a = inBounds./(length(radius))
hist(a)

end

