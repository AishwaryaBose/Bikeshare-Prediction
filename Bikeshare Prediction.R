#install and load necessary packages 

library(RGtk2)
library(readr)
library(dplyr)
library(plyr)
library(zoo)
library("randomForest")
library("lubridate")
library("pacman")
library(rattle)
library(magrittr)
library(dygraphs)
library(zoo)
library("forecast")
library(caret)
library(caretEnsemble)
library(imputeTS)
library("ggplot2")
library("dplyr")
library(colortools)
library(gridExtra)
library(RColorBrewer)

#===========================================================================================================
#Import Data
#===========================================================================================================

day <- read_csv("day.csv", 
                col_types = cols(dteday = col_date(format = "%d/%m/%Y"), 
                                 holiday = col_logical(), season = col_factor(levels = c("1", "2", "3", "4")), 
                                 weathersit = col_factor(levels = c("1", "2", "3")), 
                                 weekday = col_factor(levels = c("0", "1", "2", "3", "4", "5", "6")), 
                                 workingday = col_logical()))

View(day)

#===========================================================================================================
#Data Exploration 
#===========================================================================================================

#Scatter Plot to show the relationship between count and temp
ggplot(data = day, aes(temp,cnt)) + geom_point(alpha = 0.9, shape=1, aes(color = temp)) + geom_smooth (method=lm) + 
  xlab("Actual Temp")+ylab("Bike Rentals") + theme_bw()

#Histogram - Check for Distribution
h_plot <- hist(day$cnt, breaks = 25, ylab = 'Frequency of Rental', 
               xlab = 'Total Bike Rental Count', main = 'Distribution of Total Bike Rental Count', col = 'light blue')

xfit <- seq(min(day$cnt),max(day$cnt), length = 50)
yfit <- dnorm(xfit, mean =mean(day$cnt),sd=sd(day$cnt))
yfit <- yfit*diff(h_plot$mids[1:2])*length(day$cnt)
lines(xfit,yfit, col='red', lwd= 3)

par(mfcol=c(2,2))

boxplot(day$cnt ~ day$season,
        data = day,
        main = "Total Bike Rentals Vs Season",
        xlab = "Season",
        ylab = "Total Bike Rentals",
        col = c("coral", "coral1", "coral2", "coral3")) 

boxplot(day$cnt ~ day$holiday,
        data = day,
        main = "Total Bike Rentals Vs Holiday/Working Day",
        xlab = "Holiday/Working Day",
        ylab = "Total Bike Rentals",
        col = c("red", "red1", "red2", "red3")) 

boxplot(day$cnt ~ day$weathersit,
        data = day,
        main = "Total Bike Rentals Vs Weather Situation",
        xlab = "Weather Situation",
        ylab = "Total Bike Rentals",
        col = c("purple", "purple1", "purple2", "purple3")) 

plot(day$dteday, day$cnt,type = "p",
     main = "Total Bike Rentals Vs DateDay",
     xlab = "Year",
     ylab = "Total Bike Rentals",
     col  = "orange",
     pch  = 19)

#===========================================================================================================
#Outlier Analysis
#===========================================================================================================

plot(hum~mnth,day[which(day$yr==0),], type="l")#Humidity 0
day$hum[day$hum==0 && day$yr==0]<-mean(day$hum)#imputed with average of humidity for 2011

plot(temp~mnth,day[which(day$yr==0),], type="l") #No outliers
plot(windspeed~mnth,day[which(day$yr==0),], type="l") #No outliers

plot(cnt~mnth,day[which(day$yr==0),], type="l") #7outliers
day$cnt[day$cnt<750 && day$yr==0]<-NA
na.ma(day$cnt,k=7, weighting = "simple")

#===========================================================================================================
#Target Variable Generation to forecast Demand 2 days Later
#===========================================================================================================

#generating new variable to act as target variable. Where we try to predict the bicycle demand 2 days later based on current conditions.   
day$x2d_later_cnt = lead(day$cnt,2)

#===========================================================================================================
#Feature Engineering
#===========================================================================================================
#Adding cetrain features to predict 2 days later demand
day$x2d_later_wrkngDay = lead(day$workingday,2) #whether 2 days later it's a working day or not
wk_tempRise = day$temp-lag(day$temp,7) #change in temperature since past week
wk_tempRise[is.na(wk_tempRise)] = 0
day = cbind(day,wk_tempRise)

#Avg temp for the past 1 wk
day$wk_avg_temp = lag(rollmean(day$temp,7,fill=NA),4) 
day$wk_avg_temp[is.na(day$wk_avg_temp)] = day$temp[is.na(day$wk_avg_temp)]

#Avg humidity for the past 1 wk
day$wk_avg_hum = lag(rollmean(day$hum,7,fill=NA),4) 
day$wk_avg_hum[is.na(day$wk_avg_hum)] = day$hum[is.na(day$wk_avg_hum)]

#Avg windspeed for the past 1 wk
day$wk_avg_wndSpd = lag(rollmean(day$windspeed,7,fill=NA),4)
day$wk_avg_wndSpd[is.na(day$wk_avg_wndSpd)] = day$windspeed[is.na(day$wk_avg_wndSpd)]

#to account for annual pattern
day$day_of_yr = day(day$dteday)

#Avg demand for the past 1 week
day$wk_avg_cnt = lag(rollmean(day$cnt,7,fill=NA),4)
day$wk_avg_cnt[is.na(day$wk_avg_cnt)] = day$cnt[is.na(day$wk_avg_cnt)]

#Avg demand for the past 2 wks (15 days)
day$bimnth_avg_cnt = lag(rollmean(day$cnt,15,fill=NA),4) 
day$bimnth_avg_cnt[is.na(day$bimnth_avg_cnt)] = day$cnt[is.na(day$bimnth_avg_cnt)]

day$wk_inc_cnt = day$cnt - lag(day$wk_avg_cnt,7) #change in demand since last week's average
day$wk_inc_cnt[is.na(day$wk_inc_cnt)] = head(lead(day$wk_inc_cnt,8),7)
View(day)


#===========================================================================================================
#Training & Test Split
#===========================================================================================================
#converting to timeseries object
ts = zoo(day,day$dteday) 
head(ts,30) 

sta=as.Date("01/01/2011","%d/%m/%Y")
mid1=as.Date("31/12/2011","%d/%m/%Y")
mid2=as.Date("01/01/2012","%d/%m/%Y")
last=as.Date("31/12/2012","%d/%m/%Y")
traindata = window(ts, start=sta, end=mid1)
testdata  = window(ts, start=mid2, end=last)
traindata = as.data.frame(traindata)
testdata  = as.data.frame(testdata)

#Fixing datatypes of cols as the split caused all columns to be reset to factor
traindata$dteday = as.Date(traindata$dteday)
traindata$temp = as.numeric(as.character(traindata$temp))
traindata$atemp = as.numeric(as.character(traindata$atemp))
traindata$hum = as.numeric(as.character(traindata$hum))
traindata$windspeed = as.numeric(as.character(traindata$windspeed))
traindata$casual = as.integer(as.character(traindata$casual))
traindata$registered = as.integer(as.character(traindata$registered))
traindata$cnt = as.integer(as.character(traindata$cnt))
traindata$x2d_later_cnt = as.integer(as.character(traindata$x2d_later_cnt))
traindata$wk_tempRise = as.numeric(as.character(traindata$wk_tempRise))
traindata$wk_avg_temp = as.numeric(as.character(traindata$wk_avg_temp))
traindata$wk_avg_hum = as.numeric(as.character(traindata$wk_avg_hum))
traindata$wk_avg_wndSpd = as.numeric(as.character(traindata$wk_avg_wndSpd))
traindata$day_of_yr = as.integer(as.character(traindata$day_of_yr))
traindata$wk_avg_cnt = as.numeric(as.character(traindata$wk_avg_cnt))
traindata$bimnth_avg_cnt = as.numeric(as.character(traindata$bimnth_avg_cnt))
traindata$wk_inc_cnt = as.integer(as.character(traindata$wk_inc_cnt))


testdata$dteday = as.Date(testdata$dteday)
testdata$temp = as.numeric(as.character(testdata$temp))
testdata$atemp = as.numeric(as.character(testdata$atemp))
testdata$hum = as.numeric(as.character(testdata$hum))
testdata$windspeed = as.numeric(as.character(testdata$windspeed))
testdata$casual = as.integer(as.character(testdata$casual))
testdata$registered = as.integer(as.character(testdata$registered))
testdata$cnt = as.integer(as.character(testdata$cnt))
testdata$x2d_later_cnt = as.integer(as.character(testdata$x2d_later_cnt))
testdata$wk_tempRise = as.numeric(as.character(testdata$wk_tempRise))
testdata$wk_avg_temp = as.numeric(as.character(testdata$wk_avg_temp))
testdata$wk_avg_hum = as.numeric(as.character(testdata$wk_avg_hum))
testdata$wk_avg_wndSpd = as.numeric(as.character(testdata$wk_avg_wndSpd))
testdata$day_of_yr = as.integer(as.character(testdata$day_of_yr))
testdata$wk_avg_cnt = as.numeric(as.character(testdata$wk_avg_cnt))
testdata$bimnth_avg_cnt = as.numeric(as.character(testdata$bimnth_avg_cnt))
testdata$wk_inc_cnt = as.integer(as.character(testdata$wk_inc_cnt))
#testdata = subset(testdata,!is.na(testdata$x2d_later_cnt))



#################################################################################################
#NEURAL NETWORKS
#################################################################################################

rattle()

#============================================================================
# Iteration 1
#============================================================================

preds <- predict(crs$nnet,newdata = testdata[,crs$input])
predpairs = cbind(testdata[crs$target],preds)
plot(predpairs)

#Calculating Mean Absolute Error for a best guess of "2days later demand"= "today's demand"
bst_gs_error = testdata[crs$target]-testdata$cnt
bst_gs_error = apply(bst_gs_error,1,function(row)abs(row[1]))
cat(sprintf("Best Guess MAE=%f\n",mean(na.omit(bst_gs_error))))

AE<-function(row){abs(row[1]-row[2])}
errors = apply(predpairs,1,AE)
cat(sprintf("MAE=%f\n",mean(na.omit(errors))))

#Calculating MAPE = Avg(|Actual-forecast|/Actual * 100)
APE<-function(row){abs(row[1]-row[2])/row[1]*100}
percentage_err = apply(predpairs,1,APE)
cat(sprintf("MAPE=%f\n",mean(na.omit(percentage_err))))

#============================================================================
# Iteration 2 with past 1 Week Avg of Count + past 15 day Avg of Count as input
#============================================================================
preds <- predict(crs$nnet,newdata = testdata[,crs$input])
predpairs = cbind(testdata[crs$target],preds)
plot(predpairs)

errors = apply(predpairs,1,AE)
cat(sprintf("MAE=%f\n",mean(na.omit(errors))))

percentage_err = apply(predpairs,1,APE)
cat(sprintf("MAPE=%f\n",mean(na.omit(percentage_err))))

#============================================================================
# Iteration 3 with increase in Count since past 1 week as input
#============================================================================

preds <- predict(crs$nnet,newdata = testdata[,crs$input])
predpairs = cbind(testdata[crs$target],preds)
plot(predpairs)

errors = apply(predpairs,1,AE)
cat(sprintf("MAE=%f\n",mean(na.omit(errors))))

percentage_err = apply(predpairs,1,APE)
cat(sprintf("MAPE=%f\n",mean(na.omit(percentage_err))))

#============================================================================
# Iteration 4 after removing certain variables thru trial & error
#============================================================================

preds <- predict(crs$nnet,newdata = testdata[,crs$input])
predpairs = cbind(testdata[crs$target],preds)
#plot(predpairs)

errors = apply(predpairs,1,AE)
cat(sprintf("MAE=%f\n",mean(na.omit(errors))))

percentage_err = apply(predpairs,1,APE)
cat(sprintf("MAPE=%f\n",mean(na.omit(percentage_err))))

#################################################################################################
#RANDOM FOREST
#################################################################################################

#============================================================================
# Iteration 1 with All Variables
#============================================================================
model1 = randomForest(traindata$x2d_later_cnt~season+mnth+holiday+weekday+workingday+weathersit+temp+hum+windspeed+cnt
                      +x2d_later_wrkngDay+wk_avg_temp+wk_avg_hum+wk_avg_wndSpd+day_of_yr+wk_avg_cnt+bimnth_avg_cnt+wk_inc_cnt, data=traindata, importance=TRUE)
RndmFrst_output = predict(model1,newdata = testdata, type="response")

predpairs = cbind(testdata$x2d_later_cnt,RndmFrst_output)
plot(predpairs)

errors = apply(predpairs,1,AE)
cat(sprintf("MAE=%f\n",mean(na.omit(errors))))

percentage_err = apply(predpairs,1,APE)
cat(sprintf("MAPE=%f\n",mean(na.omit(percentage_err))))

varImpPlot(model1)

#============================================================================
# Iteration 2 after removing variables through trial and error
#============================================================================
model2 =randomForest(traindata$x2d_later_cnt~temp+cnt+wk_avg_hum+wk_avg_wndSpd
                     +wk_avg_cnt+bimnth_avg_cnt+wk_inc_cnt, data=traindata, importance=TRUE)
RndmFrst_output2 = predict(model2,newdata = testdata, type="response")

predpairs = cbind(testdata$x2d_later_cnt,RndmFrst_output2)
#plot(predpairs)

errors = apply(predpairs,1,AE)
cat(sprintf("MAE=%f\n",mean(na.omit(errors))))

percentage_err = apply(predpairs,1,APE)
cat(sprintf("MAPE=%f\n",mean(na.omit(percentage_err))))

#################################################################################################
#GLM
#################################################################################################

glmMdl<- glm(data=traindata,x2d_later_cnt~season+mnth+holiday+weekday+workingday+weathersit+temp+hum+windspeed+cnt
             +x2d_later_wrkngDay+wk_avg_temp+wk_avg_hum+wk_avg_wndSpd+day_of_yr+wk_avg_cnt+bimnth_avg_cnt+wk_inc_cnt)
glmMdl.step <- step(glmMdl)

summary(glmMdl.step)

glmMdl_output = predict(glmMdl.step,newdata = testdata, type="response")

plot(testdata$x2d_later_cnt, main = "Generalized Linear Model", ylab = "Test 2day later demand", pch = 20)
points(glmMdl_output, col = "red", pch = 20)

predpairs = cbind(testdata$x2d_later_cnt,glmMdl_output)

errors = apply(predpairs,1,AE)
cat(sprintf("MAE=%f\n",mean(na.omit(errors))))

percentage_err = apply(predpairs,1,APE)
cat(sprintf("MAPE=%f\n",mean(na.omit(percentage_err))))

#################################################################################################
#Poisson
#################################################################################################

poissonMdl <- glm(data = traindata, x2d_later_cnt~season+mnth+holiday+weekday+workingday+weathersit+temp+hum+windspeed+cnt
                  +x2d_later_wrkngDay+wk_avg_temp+wk_avg_hum+wk_avg_wndSpd+day_of_yr+wk_avg_cnt+bimnth_avg_cnt+wk_inc_cnt, family=poisson)
poissonMdl.step <- step(poissonMdl)

summary(poissonMdl.step)

poissonMdl_output = predict(poissonMdl.step,newdata = testdata, type="response")

plot(testdata$x2d_later_cnt, main = "Poisson Model", ylab = "Test 2day later demand", pch = 20)
points(poissonMdl_output, col = "red", pch = 20)

predpairs = cbind(testdata$x2d_later_cnt,poissonMdl_output)

errors = apply(predpairs,1,AE)
cat(sprintf("MAE=%f\n",mean(na.omit(errors))))

percentage_err = apply(predpairs,1,APE)
cat(sprintf("MAPE=%f\n",mean(na.omit(percentage_err))))

#################################################################################################
#Linear Model
#################################################################################################

linearMdl <- lm(formula = x2d_later_cnt~season+mnth+holiday+weekday+workingday+weathersit+temp+hum+windspeed+cnt
                +x2d_later_wrkngDay+wk_avg_temp+wk_avg_hum+wk_avg_wndSpd+day_of_yr+wk_avg_cnt+bimnth_avg_cnt+wk_inc_cnt, data=traindata)
linearMdl.step <- step(linearMdl)

summary(linearMdl.step)

linearMdl_output = predict(linearMdl.step,newdata = testdata, type="response")

plot(testdata$x2d_later_cnt, main = "Linear Model", ylab = "Test 2day later demand", pch = 20)
points(poissonMdl_output, col = "red", pch = 20)

predpairs = cbind(testdata$x2d_later_cnt,linearMdl_output)

errors = apply(predpairs,1,AE)
cat(sprintf("MAE=%f\n",mean(na.omit(errors))))

percentage_err = apply(predpairs,1,APE)
cat(sprintf("MAPE=%f\n",mean(na.omit(percentage_err))))

#################################################################################################
#Ensemble Model
#################################################################################################

#Define Training Sequence & Control + List of Models to run
trainingSequence <- trainControl(method="repeatedcv", number=20, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c("glm","rpart","nnet","knn")
set.seed(30)


#============================================================================
# Iteration 1 with All Variables
#============================================================================

#create models
models <- caretList(x2d_later_cnt~season+mnth+holiday+weekday+workingday+weathersit+temp+hum+windspeed+cnt
                    +x2d_later_wrkngDay+wk_avg_temp+wk_avg_hum+wk_avg_wndSpd+day_of_yr+wk_avg_cnt+bimnth_avg_cnt+wk_inc_cnt, data = traindata, trControl = trainingSequence,  methodList = algorithmList)

#Stack the models
stackControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE)
stack <- caretStack(models, method = "gbm", trControl = stackControl)

# Predict
ensembleOutput <- predict(stack, testdata, type = "raw")

predpairs = cbind(testdata$x2d_later_cnt,ensembleOutput)

errors = apply(predpairs,1,AE)
cat(sprintf("MAE=%f\n",mean(na.omit(errors))))

percentage_err = apply(predpairs,1,APE)
cat(sprintf("MAPE=%f\n",mean(na.omit(percentage_err))))

#============================================================================
# Iteration 2 with All Variables using GLM for Stacking
#============================================================================

#create models
models <- caretList(x2d_later_cnt~season+mnth+holiday+weekday+workingday+weathersit+temp+hum+windspeed+cnt
                    +x2d_later_wrkngDay+wk_avg_temp+wk_avg_hum+wk_avg_wndSpd+day_of_yr+wk_avg_cnt+bimnth_avg_cnt+wk_inc_cnt, data = traindata, trControl = trainingSequence,  methodList = algorithmList)

#Stack the models
stackControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE)
stack <- caretStack(models, method = "glm", trControl = stackControl)

# Predict
ensembleOutput <- predict(stack, testdata, type = "raw")

predpairs = cbind(testdata$x2d_later_cnt,ensembleOutput)

errors = apply(predpairs,1,AE)
cat(sprintf("MAE=%f\n",mean(na.omit(errors))))

percentage_err = apply(predpairs,1,APE)
cat(sprintf("MAPE=%f\n",mean(na.omit(percentage_err))))

#============================================================================
# Iteration 3 with best performing variables from Neural Nets
#============================================================================

#create models
models <- caretList(x2d_later_cnt~season+mnth+temp+hum+cnt+wk_avg_temp+wk_avg_hum+wk_avg_wndSpd+wk_avg_cnt+bimnth_avg_cnt+wk_inc_cnt,
                    data = traindata, trControl = trainingSequence,  methodList = algorithmList)

#Stack the models
stackControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE)
stack <- caretStack(models, method = "glm", trControl = stackControl)

# Predict
ensembleOutput <- predict(stack, testdata, type = "raw")

predpairs = cbind(testdata$x2d_later_cnt,ensembleOutput)

errors = apply(predpairs,1,AE)
cat(sprintf("MAE=%f\n",mean(na.omit(errors))))

percentage_err = apply(predpairs,1,APE)
cat(sprintf("MAPE=%f\n",mean(na.omit(percentage_err))))

#============================================================================
# Iteration 4 after removing variables through trial and error
#============================================================================

#create models
models <- caretList(x2d_later_cnt~season+mnth+temp+hum+cnt+wk_avg_temp+wk_avg_cnt+bimnth_avg_cnt+wk_inc_cnt,
                    data = traindata, trControl = trainingSequence,  methodList = algorithmList)

#Stack the models
stackControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE)
stack <- caretEnsemble(models, trControl = stackControl)

# Predict
ensembleOutput <- predict(stack, testdata, type = "raw")

predpairs = cbind(testdata$x2d_later_cnt,ensembleOutput)

errors = apply(predpairs,1,AE)
cat(sprintf("MAE=%f\n",mean(na.omit(errors))))

percentage_err = apply(predpairs,1,APE)
cat(sprintf("MAPE=%f\n",mean(na.omit(percentage_err))))

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#MODEL PROFIT
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#--------DEFAULT PREDICTION-------------#
#Placing order for 2 days later based on today's demand
profitDF = cbind.data.frame(testdata$x2d_later_cnt,testdata$cnt)
colnames(profitDF)[1]="actual"
colnames(profitDF)[2]="preds"
profitDF$revenue = 3*pmin(profitDF$actual,profitDF$preds)
profitDF$cost = 2*profitDF$preds
profitDF$profit = profitDF$revenue-profitDF$cost

cat(sprintf("Default Prediction Performance\nTotal Cost = $%f\nTotal Revenue = $%f\nTotal Profit = $%f\nProfit (as perc of exp.) = %f",
            sum(profitDF$cost),sum(na.omit(profitDF$revenue)),sum(na.omit(profitDF$profit)),
            sum(na.omit(profitDF$profit))*100/sum(profitDF$cost)))


#--------NEURAL NET-------------#
profitDF = cbind.data.frame(testdata$x2d_later_cnt,preds)
colnames(profitDF)[1]="actual"
profitDF$revenue = 3*pmin(profitDF$actual,profitDF$preds)
profitDF$cost = 2*profitDF$preds
profitDF$profit = profitDF$revenue-profitDF$cost

cat(sprintf("Neural Net Model Performance\nTotal Cost = $%f\nTotal Revenue = $%f\nTotal Profit = $%f\nProfit (as perc of exp.) = %f",
            sum(profitDF$cost),sum(na.omit(profitDF$revenue)),sum(na.omit(profitDF$profit)),
            sum(na.omit(profitDF$profit))*100/sum(profitDF$cost)))


#--------RANDOM FOREST------------#
profitDF$preds = RndmFrst_output2
profitDF$revenue = 3*pmin(profitDF$actual,profitDF$preds)
profitDF$cost = 2*profitDF$preds
profitDF$profit = profitDF$revenue-profitDF$cost

cat(sprintf("Random Forest Model Performance\nTotal Cost = $%f\nTotal Revenue = $%f\nTotal Profit = $%f\nProfit (as perc of exp.) = %f",
            sum(profitDF$cost),sum(na.omit(profitDF$revenue)),sum(na.omit(profitDF$profit)),
            sum(na.omit(profitDF$profit))*100/sum(profitDF$cost)))


#--------------GLM----------------#
profitDF$preds = glmMdl_output
profitDF$revenue = 3*pmin(profitDF$actual,profitDF$preds)
profitDF$cost = 2*profitDF$preds
profitDF$profit = profitDF$revenue-profitDF$cost

cat(sprintf("GLM Performance\nTotal Cost = $%f\nTotal Revenue = $%f\nTotal Profit = $%f\nProfit (as perc of exp.) = %f",
            sum(profitDF$cost),sum(na.omit(profitDF$revenue)),sum(na.omit(profitDF$profit)),
            sum(na.omit(profitDF$profit))*100/sum(profitDF$cost)))


#------------POISSON-------------#
profitDF$preds = poissonMdl_output
profitDF$revenue = 3*pmin(profitDF$actual,profitDF$preds)
profitDF$cost = 2*profitDF$preds
profitDF$profit = profitDF$revenue-profitDF$cost

cat(sprintf("Poisson Model Performance\nTotal Cost = $%f\nTotal Revenue = $%f\nTotal Profit = $%f\nProfit (as perc of exp.) = %f",
            sum(na.omit(profitDF$cost)),sum(na.omit(profitDF$revenue)),sum(na.omit(profitDF$profit)),
            sum(na.omit(profitDF$profit))*100/sum(na.omit(profitDF$cost))))


#---------LINEAR MODEL----------#
profitDF$preds = linearMdl_output
profitDF$revenue = 3*pmin(profitDF$actual,profitDF$preds)
profitDF$cost = 2*profitDF$preds
profitDF$profit = profitDF$revenue-profitDF$cost

cat(sprintf("Linear Model Performance\nTotal Cost = $%f\nTotal Revenue = $%f\nTotal Profit = $%f\nProfit (as perc of exp.) = %f",
            sum(profitDF$cost),sum(na.omit(profitDF$revenue)),sum(na.omit(profitDF$profit)),
            sum(na.omit(profitDF$profit))*100/sum(profitDF$cost)))


#-----------ENSEMBLE-------------#
profitDF$preds = ensembleOutput
profitDF$revenue = 3*pmin(profitDF$actual,profitDF$preds)
profitDF$cost = 2*profitDF$preds
profitDF$profit = profitDF$revenue-profitDF$cost

cat(sprintf("Ensemble Model Performance\nTotal Cost = $%f\nTotal Revenue = $%f\nTotal Profit = $%f\nProfit (as perc of exp.) = %f",
            sum(profitDF$cost),sum(na.omit(profitDF$revenue)),sum(na.omit(profitDF$profit)),
            sum(na.omit(profitDF$profit))*100/sum(profitDF$cost)))



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#CONDITIONS FOR [MODEL PERFORMANCE > DEFAULT PREDICTION PERFORMANCE]
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#testing for a very small revenue of $2.1 compared to cost of $2

#--------ENSEMBLE MODEL------------#
profitDF$revenue = 2.1*pmin(profitDF$actual,profitDF$preds)
profitDF$profit = profitDF$revenue-profitDF$cost

cat(sprintf("Ensemble Performance\nTotal Cost = $%f\nTotal Revenue = $%f\nTotal Profit = $%f\nProfit (as perc of exp.) = %f",
            sum(profitDF$cost),sum(na.omit(profitDF$revenue)),sum(na.omit(profitDF$profit)),
            sum(na.omit(profitDF$profit))*100/sum(profitDF$cost)))

#--------DEFAULT PREDICTION-------------#
#Placing order for 2 days later based on today's demand
profitDF = cbind.data.frame(testdata$x2d_later_cnt,testdata$cnt)
colnames(profitDF)[1]="actual"
colnames(profitDF)[2]="preds"
profitDF$revenue = 2.1*pmin(profitDF$actual,profitDF$preds)
profitDF$cost = 2*profitDF$preds
profitDF$profit = profitDF$revenue-profitDF$cost

cat(sprintf("Default Prediction Performance\nTotal Cost = $%f\nTotal Revenue = $%f\nTotal Profit = $%f\nProfit (as perc of exp.) = %f",
            sum(profitDF$cost),sum(na.omit(profitDF$revenue)),sum(na.omit(profitDF$profit)),
            sum(na.omit(profitDF$profit))*100/sum(profitDF$cost)))



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#MODEL PERFORMANCE VS SEASON & OTHER FACTORS
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

testdata$pred_cnt = ensembleOutput
testdata$error = testdata$pred_cnt - testdata$x2d_later_cnt

seasonalErr = aggregate(na.aggregate.default(abs(testdata$error)),list(testdata$season),mean)
barplot(seasonalErr$x,names.arg=seasonalErr$Group.1,xlab="Seasons",ylab="MAE")
#Season 1 has lower errors

mnthlyAE = aggregate(na.aggregate.default(abs(testdata$error)),list(testdata$mnth),mean)
# Months 1 & 2 have lower errors

WeatherErr = aggregate(na.aggregate.default(abs(testdata$error)),list(testdata$weathersit),mean)
barplot(WeatherErr$x,names.arg=WeatherErr$Group.1,xlab="Weather Situation",ylab="MAE")
# All 3 weather situations have similar errors

HolidayErr = aggregate(na.aggregate.default(abs(testdata$error)),list(testdata$holiday),mean)
barplot(HolidayErr$x,names.arg=HolidayErr$Group.1, xlab="Holiday",ylab="MAE")
# Both holiday True & False values have similar errors

WeekdyErr = aggregate(na.aggregate.default(abs(testdata$error)),list(testdata$weekday),mean)
barplot(WeekdyErr$x,names.arg=WeekdyErr$Group.1, xlab="Weekdays",ylab="MAE")
# All weekdays have similar errors

wrkngDyErr = aggregate(na.aggregate.default(abs(testdata$error)),list(testdata$workingday),mean)
barplot(wrkngDyErr$x,names.arg=wrkngDyErr$Group.1, xlab="Working day",ylab="MAE")
# Both workingDay True & False values have similar errors.



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#MODEL PERFORMANCE WITH AGE
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

mnthlyMAPE = aggregate(na.omit(abs(testdata$error)*100/testdata$x2d_later_cnt),list(testdata$mnth[!is.na(testdata$error)]),mean)

plot(testdata$dteday,abs(testdata$error),type="l", main="Absolute Error of Model over time")

barplot(mnthlyAE$x,names.arg=mnthlyAE$Group.1, xlab="Test Period Months",ylab="Mean Absolute Error",main="Absolute Error of Model over time")
