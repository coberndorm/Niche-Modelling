function points=frontierFinder(pin,Tclas,coeff,show)
polygon=boundary(pin(:,1),pin(:,2),pin(:,3),1);
shp=alphaShape(pin(polygon,1),pin(polygon,2),pin(polygon,3));
clas=Tclas{:,:}*coeff(:,1:3);
in = inShape(shp,clas(:,1),clas(:,2),clas(:,3))
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