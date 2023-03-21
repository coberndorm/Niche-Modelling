function [score, roc] = curverock(map,R,A,show,op)
%a=mean(map2(~isnan(map2)));
%según veo, este script fue pensado para un proceso de optimización y
%visualización de la curva rock
switch op
    case {1,2,4,5}%estos son los casos que merecen mayor atención
        metric=zeros(1,20);
        area=metric;
        counter=0;
        map = real(map);
        records=geointerp(map, R, [A.LAT], [A.LONG]);
        records=records(~isnan(records));
        runs=0.05:0.05:1;
        tot=sum(~isnan(map(:)));
        for i=runs
            counter=counter+1;
            vrecords=sum(records>=1-i);
            metric(counter)=vrecords/length(records);
            area(counter)=sum(map(:)>1-i)/tot;
            
        end
        switch op
            case 1
                score=sum((ones(1,counter)-metric).^2+(area.^2));%área 
                %entre cada curva y su tope priorizando los valores más
                %altos
            case 2
                score=sum((ones(1,counter)-metric)+area);%área entre cada
                %curva y su tope
            case 5
                score=1-trapz([0,area,1],[0,metric,1]);%score clásico
            otherwise
                score=1-0.05*(trapz(metric)-trapz(area));%área entre ambas curvas
        end
    case 3
        runs=0.5;
        map(map<runs)=0;
        records=geointerp(map, R, [A.latitude], [A.longitude]);
        records=records(~isnan(records));
        frecords=sum(records==0);
        vrecords=length(records)-frecords;
        metric=vrecords/length(records);
        area=sum(map(:)>0)/sum(~isnan(map(:)));
        score=sum((1-metric)+area);
    otherwise
            disp('Wrong metric selected')
            return
end

roc = trapz([0,area,1],[0,metric,1]);

if show && length(runs)>1
    figure('Name','Rock')
    plot([0,area,1],[0,metric,1],[0 1],[0 1],'LineWidth',2)
    xlabel('Map area (%)')
    ylabel('Predicted records (%)')
    legend({'Actual model','Null model'},'Location','best')
    title(strcat('Score: ',num2str(round(trapz([0,area,1],[0,metric,1])*100)),'%'))
    figure('Name','Objetive')
    plot(runs,metric,runs,area,'LineWidth',2)
    xlabel('1-Probability of prescence')
    ylabel('%')
    legend({'Records','Area'},'Location','best')
end
end