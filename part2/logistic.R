library(glmnet)
library(gdata)
require(methods)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
df <- read.xls("./default of credit card clients.xls")

y<-as.matrix(as.numeric(unlist(df[ 2:3001,c("Y")])))
#as.matrix(lapply(df[,ncol(df)], as.numeric))
feature<-data.matrix(df[2:3001 ,-c(1,25)])


model0<-cv.glmnet(feature, y,alpha=0,nfolds=10,family="binomial")
plot(model0)

mean_error_1se_0 <-mean(predict(model0, feature, s=model0$lambda.1se, type="class") != y)
mean_error_min_0 <-mean(predict(model0, feature, s=model0$lambda.min, type="class") != y)

legend( 'top' , legend='alpha=0')
 dev.copy(jpeg,filename="alpha0.jpg");
 dev.off ();

model1<-cv.glmnet(feature, y,alpha=1,nfolds=10,family="binomial")
 mean_error_1se_1 <-mean(predict(model1, feature, s=model1$lambda.1se, type="class") != y)
 mean_error_min_1 <-mean(predict(model1, feature, s=model1$lambda.min, type="class") != y)
plot(model1)
legend( 'top' , legend='alpha=1')
dev.copy(jpeg,filename="alpha1.jpg")
dev.off ()

model_dot1<-cv.glmnet(feature, y, alpha=0.1,nfolds=10,family="binomial")
mean_error_1se_.1 <-mean(predict(model_dot1, feature, s=model_dot1$lambda.1se, type="class") != y)
mean_error_min_.1 <-mean(predict(model_dot1, feature, s=model_dot1$lambda.min, type="class") != y)
plot(model_dot1)
legend( 'top' , legend='alpha=0.1')

dev.copy(jpeg,filename="alpha.1.jpg")
dev.off ()

model_dot5<-cv.glmnet(feature, y,alpha=0.5,nfolds=10,family="binomial")
mean_error_1se_.5 <-mean(predict(model_dot5, feature, s=model_dot5$lambda.1se, type="class") != y)
mean_error_min_.5 <-mean(predict(model_dot5, feature, s=model_dot5$lambda.min, type="class") != y)
plot(model_dot5)
legend( 'top' , legend='alpha=0.5')
dev.copy(jpeg,filename="alpha.5.jpg")
dev.off ()

model_dot8<-cv.glmnet(feature, y, alpha=0.8,nfolds=10,family="binomial")
mean_error_1se_.8 <-mean(predict(model_dot8, feature, s=model_dot8$lambda.1se, type="class") != y)
mean_error_min_.8 <-mean(predict(model_dot8, feature, s=model_dot8$lambda.min, type="class") != y)
plot(model_dot8)
legend( 'top' , legend='alpha=0.8')
dev.copy(jpeg,filename="alpha.8.jpg")
dev.off ()

plot ( log( model0$lambda ) , model0$cvm , pch =19, col='red',
        xlab='l o g (Lambda ) ', ylab=model0$name )
points ( log( model1$lambda ) , model1$cvm , pch =19, col='grey')
points ( log( model_dot1$lambda ) , model_dot1$cvm , pch=19, col='blue')
points ( log( model_dot5$lambda ) , model_dot5$cvm , pch=19, col='pink')
points ( log( model_dot8$lambda ) , model_dot8$cvm , pch=19, col='orange')
legend( 'bottomright',
       legend=c ( 'alpha=0', 'alpha =1 ', 'alpha=0.1','alpha=0.5','alpha=0.8') ,
       pch =19, col=c( 'red', 'grey', 'blue','pink','orange') )
dev.copy(jpeg,filename="all.jpg")
dev.off ()

