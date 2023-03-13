
clear;clc;close all

% Reading the map to study (has to be a folder with .asc type files)
Layer_Folder='../data/Capas_Colombia_30S/';
Dimensions = ReadLayers(Layer_Folder);

% Defining lists of options for experimentation
Virtual_Species_Methods = ["harmonic", "beta", "coeff"];
Samples = [50, 100, 300, 500, 1000];
Outlier_Handling = [false, true];
Number_Of_Maps = 50;

for idx = 1:3
    close all, clf

    Acc_Results_Closest_Point_Method = zeros(length(Samples), Number_Of_Maps);
    Acc_Results_Percentile_Point_Method = zeros(length(Samples), Number_Of_Maps);
    

    % Initializing outlier handling and if generated maps show
    Outlier_Before_PCA = false;
    Outlier_After_PCA = true;
    Show_Graphs=false;

    % Choosing virtual species niche generation method
    Virtual_Species_Method = Virtual_Species_Methods(idx)

    for idx1 = 1:50

        % Choosing an initial point
        Info_Initial_Point = InitialPoint(Dimensions, ...
            Virtual_Species_Method); 
        
        % Generating niche based on distribution generation method and
        % initialPoint chosen
        Map_Info = NicheGeneration(Dimensions, Info_Initial_Point, 0.8, ...
            Show_Graphs);

        for idx2 = 1:5
            % Choosing amount of samples to generate on vritual niche
            Number_Samples = Samples(idx2)

            % Generating samples
            T = samplingVS(Dimensions, Info_Initial_Point, Map_Info, ...
                Number_Samples, -1, Show_Graphs, 'GenSP', true, true);

            close all, clf
    
            % Aproximating niche with closest frontier point method
            classA1 = ColoringBorder(T,Dimensions,1,Show_Graphs, ...
                Outlier_Before_PCA,Outlier_After_PCA); 
            Accuracy_Closest_Point_Method = MapMetric(Map_Info.Map,classA1.map,false);
            Acc_Results_Closest_Point_Method(idx2, idx1) = Accuracy_Closest_Point_Method(1);
        
            % Aproximating niche with 25 percentile closest frontier points
            % average
            classB1 = ColoringRadius(T,Dimensions,1,25,Show_Graphs, ...
                Outlier_Before_PCA,Outlier_After_PCA); 
            Accuracy_Percentile_Point_Method = MapMetric(Map_Info.Map,classB1.map,false);
            Acc_Results_Percentile_Point_Method(idx2, idx1) = Accuracy_Percentile_Point_Method(1);
        end
    end
    Result_Closest_Point_Method = mean(Acc_Results_Closest_Point_Method')
    Result_Percentile_Point_Method = mean(Acc_Results_Percentile_Point_Method')
end
