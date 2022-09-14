    function [template,score]=predictpca(Tin,Z,R,indicators,vars,show,outlier,outlier2)
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
viable=~isnan(data(:,i));
Tclas=table();
for i=indicators
    eval(strcat("Tclas.bio",num2str(i),"=data(viable,i);"))
end
for i=vars
    eval(strcat("Tclas.var",num2str(i),"=Tin.Properties.CustomProperties.m",num2str(i),"(data(viable,:));"))
end
normalizers=[max(Tclas{:,:});min(Tclas{:,:})];
Tin{:,:}=(Tin{:,:}-normalizers(2,:))./(normalizers(1,:)-normalizers(2,:));
Tclas{:,:}=(Tclas{:,:}-normalizers(2,:))./(normalizers(1,:)-normalizers(2,:));

if outlier
    [~,~,RD,chi_crt]=DetectMultVarOutliers(Tin{:,:});
    id_out=RD>chi_crt(4);
    out1=Tin{id_out,:};
    Tin=Tin(~id_out,:);
end
[coeff,~,~,~,explained]=pca(Tin{:,:});
pin=Tin{:,:}*coeff(:,1:3);

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

outT=[out1;out2];
if isempty(outT)
    grap=false;
else
    grap=true;
end

% if outlier
%     [~,~,~,~,W] = robust_by_skew_v2(Tin{:,:},1);
%     Tin=array2table(W);
% end
% [coeff,~,~,~,explained]=pca(Tin{:,:});
% pin=Tin{:,:}*coeff(:,1:3);
% if outlier2
%     [~,~,~,~,W] = robust_by_skew_v2(pin,1);
%     pin=W;
% end

polygon=boundary(pin(:,1),pin(:,2),pin(:,3),1);
shp=alphaShape(pin(polygon,1),pin(polygon,2),pin(polygon,3));
clas=Tclas{:,:}*coeff(:,1:3);
in = inShape(shp,clas(:,1),clas(:,2),clas(:,3));
if show
    figure('Name','Data')
    plot3(pin(:,1),pin(:,2),pin(:,3),'g.')
    if grap
        hold on
        plot3(outT(:,1),outT(:,2),outT(:,3),'r.')
        legend({'Inliers','outliers'})
    end
    xlabel('PCA1')
    ylabel('PCA2')
    zlabel('PCA3')
    figure('Name','Manifold of data')
    subplot(1,2,1)
    plot(shp)
    xlabel('PCA1')
    ylabel('PCA2')
    zlabel('PCA3')
    subplot(1,2,2)
    plot3(clas(~in,1),clas(~in,2),clas(~in,3),'b.')
    hold on
    plot(shp)
    %trisurf(polygon,pin(:,1),pin(:,2),pin(:,3),'Facecolor','red','FaceAlpha',0.1)
    plot3(clas(in,1),clas(in,2),clas(in,3),'r.')
    xlabel('PCA1')
    ylabel('PCA2')
    zlabel('PCA3')
    suptitle(strcat(num2str(round(sum(explained(1:3))),2),'% explained'))
end
map=template(viable);
map(in)=1;
map(~in)=0.3;
template(viable)=map;
score=sum(in)/sum(viable)*100;
if show
    figure
    clf
    geoshow(template,R,'DisplayType','surface')
    contourcmap('jet',0:0.05:1,'colorbar','on','location','vertical')
    drawnow
    title(strcat('Score: ',num2str(round(score,2)),'%'))
end
end