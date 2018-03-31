library(glmnet)
library(gdata)
library(MASS)
library(GetoptLong)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
df <- read.table("./Geographical Original of Music/default_plus_chromatic_features_1059_tracks.txt",sep=',')


y_lon<-as.data.frame(df[ ,c(117)])
y_lat<-as.data.frame(df[ ,c(118)])
#as.matrix(lapply(df[,ncol(df)], as.numeric))
feature_lat<-as.data.frame(df[ ,-c(117)])
feature_lon<-as.data.frame(df[ ,-c(118)])

lm_lon = lm('V118~.',data=feature_lat)
lm_lat = lm('V117~.',data=feature_lon)
summary(lm_lon)$r.squared
summary(lm_lat)$r.squared
plot ( predict(lm_lon, feature_lon) , lm_lon$residuals , pch =19, col='red',
       xlab='fitted', ylab='residuals',main='LM Longitude' )
#abline(lm_lon)

dev.copy(jpeg,filename="lm_lon.jpg");
dev.off ();

plot ( predict(lm_lat, feature_lat) , lm_lat$residuals , pch =19, col='red',
       xlab='fitted', ylab='residuals',main='LM Latitude'  )
#abline(lm_lat)

dev.copy(jpeg,filename="lm_lat.jpg");
dev.off ();




feature_lat['V118'] = feature_lat['V118'] +90
feature_lon['V117'] = feature_lon['V117'] +180



bc_lat <- boxcox(lm_lon,plotit = TRUE)
bc_lon <- boxcox(lm_lat,plotit = TRUE)

bc_lat_lambda = bc_lat$x[which.max(bc_lat$y)]
bc_lon_lambda = bc_lon$x[which.max(bc_lon$y)]

feature_lat['V118']=(feature_lat['V118']^bc_lat_lambda-1)/bc_lat_lambda
feature_lon['V117']=(feature_lon['V117']^bc_lon_lambda-1)/bc_lon_lambda

lm_lon = lm('V118~.',data=feature_lat)
lm_lat = lm('V117~.',data=feature_lon)
summary(lm_lon)$r.squared
summary(lm_lat)$r.squared

feature_lat_mat=data.matrix(feature_lat)
y_lat=as.matrix(as.numeric(unlist(feature_lat['V118'])))
feature_lon_mat=data.matrix(feature_lon)
y_lon=as.matrix(as.numeric(unlist(feature_lon['V117'])))

alpha <-0
lm_lat0<-cv.glmnet(feature_lat_mat, 
                   y_lat,alpha=alpha,
                  nfolds=10)
plot(lm_lat0)
legend( 'top' , legend=qq('Latitude alpha=@{alpha}'))
dev.copy(jpeg,filename=qq("latitide-alpha@{alpha}.jpg"))
dev.off ();
mean_error_1se_0 <-mean(predict(lm_lat0, feature_lat_mat, s=lm_lat0$lambda.1se, type="class") != y_lat)
mean_error_min_0 <-mean(predict(lm_lat0, feature_lat_mat, s=lm_lat0$lambda.min, type="class") != y_lat)
print(qq('Latitude mean_error_1se alpha @{alpha}  :  @{mean_error_1se_0}'))
print(qq('Latitude mean_error_min alpha @{alpha}  :  @{mean_error_min_0}'))

lm_lon0<-cv.glmnet(feature_lon_mat, 
                   y_lon,alpha=alpha,
                   nfolds=10)
plot(lm_lat0)
legend( 'top' , legend=qq('longitude alpha=@{alpha}'))
dev.copy(jpeg,filename=qq("longitude-alpha@{alpha}.jpg"))
dev.off ();
mean_error_1se_0 <-mean(predict(lm_lon0, feature_lon_mat, s=lm_lon0$lambda.1se, type="class") != y_lon)
mean_error_min_0 <-mean(predict(lm_lon0, feature_lon_mat, s=lm_lon0$lambda.min, type="class") != y_lon)
print(qq('longitude mean_error_1se alpha @{alpha}  :  @{mean_error_1se_0}'))
print(qq('longitude mean_error_min alpha @{alpha}  :  @{mean_error_min_0}'))









