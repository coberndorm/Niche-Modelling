function [bestMap, averageMap, medianMap, worstMap, metricAlllMaps] = results(Layers, mapsAmount)

warning off
bestMap = cell(5,2);
bestMap{1,1} = 0; bestMap{1,2} = 1;
averageMap = cell(5,1);
averageMap{1} = 1;
medianMap = cell(5,1);
medianMap{1} = 1;
worstMap = cell(5,2);
worstMap{1,1} = 0; worstMap{1,2} = 1;
metricAlllMaps =  NaN (mapsAmount,1);

for map=1:mapsAmount
map
InfoInitialPoint = InitialPoint(Layers,'harmonic', false);

MapInfoWorst = NicheGeneration(Layers, InfoInitialPoint, 0.9, false);
MapInfoAverage = NicheGeneration(Layers, InfoInitialPoint, 0.6, false);
MapInfoBest = NicheGeneration(Layers, InfoInitialPoint, 0.3, false);

TWorst = samplingVS(Layers, InfoInitialPoint, MapInfoWorst, 20, -1, false, 'GenSP', true, true);
TAverage = samplingVS(Layers, InfoInitialPoint, MapInfoAverage, 50, -1, false, 'GenSP', true, true);
TBest = samplingVS(Layers, InfoInitialPoint, MapInfoBest, 500, -1, false, 'GenSP', true, true);

worst = FrontierDepthPreprocessing(TWorst, Layers, 0, 1, false, false, false, 0.9);
average = FrontierDepthPreprocessing(TAverage, Layers, 0, 1, false, false, false, 0.9);
best = FrontierDepthPreprocessing(TBest, Layers, 0, 1, false, false, false, 0.9);

worstMetric = MapMetric(MapInfoWorst.Map,worst.map,false);
averageMetric = MapMetric(MapInfoAverage.Map,average.map,false);
bestMetric = MapMetric(MapInfoBest.Map,best.map,false);

metricAlllMaps(map) = averageMetric(2);

if worstMetric(2) > worstMap{1,1}
    worstMap{1,1} = worstMetric(2); worstMap{2,1} = worst; worstMap{3,1} = TWorst;
    worstMap{4,1} = MapInfoWorst; worstMap{5,1} = InfoInitialPoint;
end
if worstMetric(2) < worstMap{1,2}
    worstMap{1,2} = worstMetric(2); worstMap{2,2} = worst; worstMap{3,2} = TWorst;
    worstMap{4,2} = MapInfoWorst; worstMap{5,2} = InfoInitialPoint;
end

meanVal = mean(metricAlllMaps(~isnan(metricAlllMaps)));

if abs(meanVal - averageMetric(2)) < abs(meanVal - averageMap{1})
    averageMap{1} = averageMetric(2); averageMap{2} = average; averageMap{3} = TAverage; averageMap{4} = MapInfoAverage; averageMap{5} = InfoInitialPoint;
end

medianVal = median(metricAlllMaps(~isnan(metricAlllMaps)));

if abs(medianVal - averageMetric(2)) < abs(meanVal - medianMap{1})
    medianMap{1} = averageMetric(2); medianMap{2} = average; medianMap{3} = TAverage; medianMap{4} = MapInfoAverage; medianMap{5} = InfoInitialPoint;
end

if bestMetric(2) > bestMap{1,1}
    bestMap{1,1} = bestMetric(2); bestMap{2,1} = best; bestMap{3,1} = TBest; bestMap{4,1} = MapInfoBest; bestMap{5,1} = InfoInitialPoint;
end
if bestMetric(2) < bestMap{1,2}
    bestMap{1,2} = bestMetric(2); bestMap{2,2} = best; bestMap{3,2} = TBest; bestMap{4,2} = MapInfoBest; bestMap{5,2} = InfoInitialPoint;
end
end
