function [matrixPreprocessing,matrixNoPreprocessing] = ExperimentalMatrixCode(Layers, mapsAmount,nicheOccupations,samples,correlationPercentages)

warning off
matrixPreprocessing = NaN(mapsAmount, length(nicheOccupations),length(samples),length(correlationPercentages),2);
matrixNoPreprocessing = NaN(mapsAmount, length(nicheOccupations),length(samples),2);


for map=1:mapsAmount
InfoInitialPoint = InitialPoint(Layers,'harmonic', false);

for i=1:length(nicheOccupations)
nicheOccupation = nicheOccupations(i);
MapInfo = NicheGeneration(Layers, InfoInitialPoint, nicheOccupation, false);

for j=1:length(samples)
sample = samples(j);
T = samplingVS(Layers, InfoInitialPoint, MapInfo, sample, -1, false, 'GenSP', true, true);

for k=1:length(correlationPercentages)
corr = correlationPercentages(k);
frontierDepthPreprocessing = FrontierDepthPreprocessing(T, Layers, 0, 1, false, false, false, corr);

%Acurracy of the estimated preprocessed niche
met1 = MapMetric(MapInfo.Map,frontierDepthPreprocessing.map,false);
matrixPreprocessing(map,i,j,k,:)=met1;

end
%Accuracy of the estimated niche
frontierDepth = FrontierDepth(T, Layers, 0, 1, false, false, false);
met2 = MapMetric(MapInfo.Map,frontierDepth.map,false);
matrixNoPreprocessing(map,i,j,:)=met2;

end
end
end

end

