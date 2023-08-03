function sol = gsua_connect(pars,domain,in)

Dimensions = in.Dimensions;
map = ceil(pars(1));%elección de mapa
occ = pars(2);
nsamples = round(pars(3));
factor = ceil(pars(4));
alpha = pars(5);
percentile = floor(pars(6));

InfoInitialPoint = in.InfoInitialPoint{map}; %Generate a initial point to generate a niche
MapInfo = NicheGeneration(Dimensions, InfoInitialPoint, occ, false); %Generate the niche based on the initial point


T = samplingVS(Dimensions, InfoInitialPoint, MapInfo, nsamples, factor, false, 'GenSP', true, false); %Generate samples based on the generated niche
show=false;%show graphics
outlier=false;%remove outliers before PCA with method in https://github.com/AntonSemechko/Multivariate-Outliers
outlier2=false;%remove outliers after PCA with method in https://github.com/AntonSemechko/Multivariate-Outliers

%ColoringBorder generates classifier object with atributes nodes = environmental variables
%for each frontier point, 
%T = table with samples and information
%readInfo = dimensions of the map
%alpha = shrinking factor

classA = ColoringBorder(T,Dimensions,alpha,show,outlier,outlier2); 
classB = ColoringRadius(T,Dimensions,alpha,percentile,show,outlier,outlier2); 
data = bnm_prep(T, Dimensions, false, 0.7, false, false);
dataf = bnm_modeling(data, '', false, 4, false);



M1 = MapMetric(MapInfo.Map,classA.map,false);
M2 = MapMetric(MapInfo.Map,classB.map,false);
M3 = MapMetric(MapInfo.Map,dataf.Map,false);

sol = struct();
sol.x=0:domain(2);
sol.y=[M1' M2' M3'];


end