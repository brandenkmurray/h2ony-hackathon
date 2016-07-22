library(data.table)
library(h2o)
library(xgboost)
library(lubridate)

# ts1 <- fread("./Task15_W_Zone1.csv")
t1 <- fread("./Train_O4UPEyW.csv")
TARGETVAR <- t1[,TARGETVAR]
t1[,TARGETVAR:=NULL] # Moving TARGETVAR TO END TO EASILY MERGE WITH TEST SET
t1[,TARGETVAR:=TARGETVAR]
s1 <- fread("./Test_uP7dymh.csv")
s1[,TARGETVAR:=-1]
ts1 <- rbind(t1, s1)
# set(ts1, i=which(is.na(ts1[,TARGETVAR])), j='TARGETVAR', value=0)
ts1[,TIMESTAMP:=as.POSIXct(TIMESTAMP, format="%Y%m%d %H:%M")]
ts1[,":="(id=1:.N, # just to help maintin row order if I need to
          filter=ifelse(TIMESTAMP<"2013-06-01 00:00:00",0,ifelse(TIMESTAMP<"2013-08-01 00:00:00",1,2)))]

ts1[,":="(month=lubridate::month(TIMESTAMP),
          dayOfWeek=lubridate::wday(TIMESTAMP),
          dayOfYear=lubridate::yday(TIMESTAMP),
          diff_U=U10-U100,
          diff_V=V10-V100,
          ratio_U=U10/U100,
          ratio_V=V10/V100)]
ts1[,c("U10_lag4","U10_lag3","U10_lag2","U10_lag1","V10_lag4","V10_lag3","V10_lag2","V10_lag1","U100_lag4","U100_lag3","U100_lag2","U100_lag1","V100_lag4","V100_lag3","V100_lag2","V100_lag1"):=shift(.SD, n=1:4,type="lag",fill=-999), by="ZONEID", .SDcols=c("U10","V10","U100","V100")]
ts1[,c("U10_lead4","U10_lead3","U10_lead2","U10_lead1","V10_lead4","V10_lead3","V10_lead2","V10_lead1","U100_lead4","U100_lead3","U100_lead2","U100_lead1","V100_lead4","V100_lead3","V100_lead2","V100_lead1"):=shift(.SD, n=1:4,type="lead",fill=-999), by="ZONEID", .SDcols=c("U10","V10","U100","V100")]
# Means of lag and lead variables
ts1[,":="(mean_U10_lag=colMeans(ts1[,list(U10_lag4, U10_lag3, U10_lag2, U10_lag1)]), 
          mean_U100_lag=colMeans(ts1[,list(U100_lag4, U100_lag3, U100_lag2, U100_lag1)]),
          mean_V10_lag=colMeans(ts1[,list(V10_lag4, V10_lag3, V10_lag2, V10_lag1)]),
          mean_V100_lag=colMeans(ts1[,list(V100_lag4, V100_lag3, V100_lag2, V100_lag1)]),
          mean_U10_lead=colMeans(ts1[,list(U10_lead4, U10_lead3, U10_lead2, U10_lead1)]),
          mean_U100_lead=colMeans(ts1[,list(U100_lead4, U100_lead3, U100_lead2, U100_lead1)]),
          mean_V10_lead=colMeans(ts1[,list(V10_lead4, V10_lead3, V10_lead2, V10_lead1)]),
          mean_V100_lead=colMeans(ts1[,list(V100_lead4, V100_lead3, V100_lead2, V100_lead1)]))]

ts1[,":="(sd_U10_lag=apply(ts1[,list(U10_lag4, U10_lag3, U10_lag2, U10_lag1)], 2, sd),
          sd_U100_lag=apply(ts1[,list(U10_lag4, U10_lag3, U10_lag2, U10_lag1)], 2, sd),
          sd_V10_lag=apply(ts1[,list(U10_lag4, U10_lag3, U10_lag2, U10_lag1)], 2, sd),
          sd_V100_lag=apply(ts1[,list(U10_lag4, U10_lag3, U10_lag2, U10_lag1)], 2, sd))]

# Add original features from each zone as another variable. e.g. the forecasts for ZONE 10 might be able to help predict ZONES 1 through 9
ts1 <- cbind(ts1, ts1[ZONEID==1, c(4:7), with=F])
ts1 <- cbind(ts1, ts1[ZONEID==2, c(4:7), with=F])
ts1 <- cbind(ts1, ts1[ZONEID==3, c(4:7), with=F])
ts1 <- cbind(ts1, ts1[ZONEID==4, c(4:7), with=F])
ts1 <- cbind(ts1, ts1[ZONEID==5, c(4:7), with=F])
ts1 <- cbind(ts1, ts1[ZONEID==6, c(4:7), with=F])
ts1 <- cbind(ts1, ts1[ZONEID==7, c(4:7), with=F])
ts1 <- cbind(ts1, ts1[ZONEID==8, c(4:7), with=F])
ts1 <- cbind(ts1, ts1[ZONEID==9, c(4:7), with=F])
ts1 <- cbind(ts1, ts1[ZONEID==10, c(4:7), with=F])

ts1 <- ts1[order(ID)] # reorder to match order of zonecast
# dummy variables for ZONES
zonecast <- dcast.data.table(ts1, ID ~ ZONEID, fun.aggregate = length, value.var="ID")
ts1 <- cbind(ts1, zonecast)


colnames(ts1) <- make.unique(colnames(ts1))
excludeCols <- c("ID","ID.1","TIMESTAMP","TARGETVAR","filter","id","dayOfWeek","month","ZONEID")
varnames <- setdiff(colnames(ts1), excludeCols)

# varnames <- xgbImp$Feature[order(xgbImp$Gain)][1:87]

# Create XGB matrices
dtrain <- xgb.DMatrix(data=data.matrix(ts1[filter==0, varnames, with=F]), label=data.matrix(ts1[filter==0,TARGETVAR]))
dval <- xgb.DMatrix(data=data.matrix(ts1[filter==1, varnames, with=F]), label=data.matrix(ts1[filter==1,TARGETVAR]))
dtest <- xgb.DMatrix(data=data.matrix(ts1[filter==2, varnames, with=F]))
watchlist <- list(val=dval, train=dtrain)

param <- list(objective="reg:linear",
              eta = .01,
              max_depth=9,
              min_child_weight=1,
              subsample=.4,
              colsample_bytree=.55,
              eval_metric="rmse"
)

set.seed(201607)
tme <- Sys.time()
xgb1val <- xgb.train(data=dtrain,
                  watchlist=watchlist,
                  nrounds = 10000,
                  maximize = F,
                  print.every.n=100,
                  params=param,
                  early.stop.round = 100)
Sys.time() - tme

dtrain <- xgb.DMatrix(data=data.matrix(ts1[filter %in% c(0,1), varnames, with=F]), label=data.matrix(ts1[filter %in% c(0,1), TARGETVAR]))

set.seed(201607)
tme <- Sys.time()
xgb1 <- xgb.train(data=dtrain,
                  nrounds = xgb1val$bestInd,
                  maximize = F,
                  print.every.n=100,
                  params=param)
Sys.time() - tme

preds_xgb1 <- predict(xgb1, dtest)
preds_xgb1 <- pmax(preds_xgb1, 0)
preds_xgb1 <- pmin(preds_xgb1, 1)
sub <- data.table(ID=ts1[filter==2,ID], TARGETVAR=preds_xgb1)
fwrite(sub, "./sub4_test.csv")
