function Results = Experimentation(Dimensions, Virtual_Species_Methods, Samples, Outlier_Handling, Number_Of_Maps)
% This function runs an experimentation to evaluate the accuracy of two methods
% for approximating a niche. The function generates several virtual species
% niches and for each one, it generates a number of samples and approximates
% the niche using two different methods. The accuracy results of the two methods
% are stored in a struct and returned by the function.

% INPUTS:
%   Dimensions: The number of dimensions of the niches to be generated.
%   Virtual_Species_Methods: A vector of three strings indicating the methods
%   to be used for generating the virtual species niches.
%   Samples: A vector of three integers indicating the number of samples to be
%   generated on each virtual species niche.
%   Outlier_Handling: A boolean value indicating whether outlier handling should
%   be used or not. (no se esta usando)
%   Number_Of_Maps: The number of virtual species niches to be generated.

% OUTPUTS:
% - Results: A struct containing the accuracy results of the two methods for
%   approximating the niches.

    for idx = 1:3
        close all, clf
    
        Acc_Results_Closest_Point_Method = zeros(length(Samples), Number_Of_Maps);
        Acc_Results_Percentile_Point_Method = zeros(length(Samples), Number_Of_Maps);
        
    
        % Initializing outlier handling and if generated maps show
        Outlier_Before_PCA = false;
        Outlier_After_PCA = false;
        Show_Graphs=false;
    
        % Choosing virtual species niche generation method
        Virtual_Species_Method = Virtual_Species_Methods(idx)
    
        for idx1 = 1:Number_Of_Maps
    
            % Choosing an initial point
            Info_Initial_Point = InitialPoint(Dimensions, ...
                Virtual_Species_Method); 
            
            % Generating niche based on distribution generation method and
            % initialPoint chosen
            Map_Info = NicheGeneration(Dimensions, Info_Initial_Point, 0.8, ...
                Show_Graphs);
    
            for idx2 = 1:3
                % Choosing amount of samples to generate on vritual niche
                Number_Samples = Samples(idx2);
    
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

        % Computing mean accuracy results for the two methods
        Result_Closest_Point_Method = mean(Acc_Results_Closest_Point_Method')
        Result_Percentile_Point_Method = mean(Acc_Results_Percentile_Point_Method')
    end

    % Saving results for current virtual species method
    Results.Result_Closest_Point_Method = Result_Closest_Point_Method;
    Results.Result_Percentile_Point_Method = Result_Percentile_Point_Method;
end