library(glmnet)
library(gdata)
library(MASS)
library(car)
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

feature_lat[,117] = feature_lat[,117] +90
feature_lon[,117] = feature_lon[,117] +180

lm_lat_temp = lm('V118~.',data=feature_lat)
lm_lon_temp = lm('V117~.',data=feature_lon)


bc_lat <- boxcox(lm_lat_temp,plotit = TRUE)
bc_lon <- boxcox(lm_lon_temp,plotit = TRUE)

bc_lat_lambda = bc_lat$x[which.max(bc_lat$y)]
bc_lon_lambda = bc_lon$x[which.max(bc_lon$y)]



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