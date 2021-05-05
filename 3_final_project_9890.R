rm(list = ls())    #delete objects
cat("\014")        #clear console
library(readr)
library(dplyr)
library(tidyr)
library(glmnet)
library(glmnetUtils)
library(randomForest)
library(gridExtra)
library(ggplot2)
library(ggpubr)
library(tidyverse) 
library(modelr) ## packages for data manipulation and computing rmse easily.
#library(matrix)
library(caret)
library(ISLR)
library(leaps)
library(tictoc)

setwd("/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3")
data<- read_csv("./Features_Variant_1_first_500.csv", col_names = FALSE)

names(data) =  c("X1", "X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20",
                "X21", "X22","X23", "X24","X25","X26","X27","X28","X29","X30","X31","X32","X33",
                 "X34", "X35","X36", "X37","X38","X39","X40","X41","X42","X43","X44","X45","X46",
                 "X47", "X48","X49", "X50","X51","X52", "X53" ,"comment")
#data =  read_csv("/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/data_altered.csv", col_names = TRUE)
#data = na.omit(data)
data_altered = data

dim(data_altered)
colnames(data_altered)
n  = dim(data_altered)[1]
p  = dim(data_altered)[2]-1
nsim = 1
###################### part 3 of the project ######################

X_all = as.matrix(data_altered[,1:p])
y_all = data_altered$comment
d = ncol(X_all)

n = nrow(data_altered)

#n_train = 10*(ceiling(0.8*n)%/%10)  ## make sure it is a multiple of 10
n_train       = floor(0.80*n)
n_test        = n - n_train
print(paste('Number of trainning data is:',n_train))
print(paste('Number of test data is:',n_test))

#train_ind    = sample(seq_len(nrow(data_altered)),size = n_train) #Randomly identifies therows equal to sample size
#X.train      = data_altered[train_ind,]   #initially
#X.test       = data_altered[-train_ind,]

train_ind  = sample(1:n_train)
X.train    = data_altered[train_ind,]
X.test     = data_altered[-train_ind,]


## create a matrix to store the shufflings for 100 simulation of Cross Validation
## this is the step to create reproducible output
set.seed(0)
i.mix = matrix(NA, nrow=nsim, ncol = n)
for (i in 1:nsim){
  i.mix[i, ] = sample(1:n)
}
str(i.mix)


## setting the lambdas for tuning
lambda.las = c(seq(1e-1,2,length=100),seq(2.0001,10,length=100))
lambda.rid = lambda.las*10
lambda.elast = lambda.las*2

#plot(lambda.las)

## 200 lambda values
nlam = length(lambda.las)
nlam

#----------------------------------------------------------------------------------
#                        50 simulation of 10 fold Cross Validation
#----------------------------------------------------------------------------------
  ## set the number of simulation we want

num_columns = ncol(X.train)
nsim_betas_ridge         = data.frame(matrix(data=NA, nrow=nsim, ncol=p)) ## nsim X num_columns matrix
nsim_betas_lasso         = data.frame(matrix(data=NA, nrow=nsim, ncol=p))
nsim_betas_elast         = data.frame(matrix(data=NA, nrow=nsim, ncol=p))
nsim_betas_forest        = data.frame(matrix(data=NA, nrow=nsim, ncol=p))

colnames(nsim_betas_ridge)   = colnames(X.train[1:p])
colnames(nsim_betas_lasso)   = colnames(X.train[1:p])
colnames(nsim_betas_elast)   = colnames(X.train[1:p])
colnames(nsim_betas_forest)  = colnames(X.train[1:p])

#########Alternatively
beta_hat_ridge_     = matrix(data=0, nrow=p, ncol=nsim) ## nsim X num_columns matrix
beta_hat_lasso_     = matrix(data=0, nrow=p, ncol=nsim)
beta_hat_elast_    = matrix(data=0, nrow=p, ncol=nsim)
beta_hat_forest_   = matrix(data=0, nrow=p, ncol=nsim)

beta_hat_ridge_all_     = matrix(data=0, nrow=p, ncol=nsim) ## nsim X num_columns matrix
beta_hat_lasso_all_     = matrix(data=0, nrow=p, ncol=nsim)
beta_hat_elast_all_    = matrix(data=0, nrow=p, ncol=nsim)
beta_hat_forest_all_   = matrix(data=0, nrow=p, ncol=nsim)

#################


## TO store nsim results of (best)lambda used, train R^2, test R^2, model type, time elapsed
values_ridge = data.frame(matrix(data=NA, nrow=nsim, ncol=6))
values_lasso = data.frame(matrix(data=NA, nrow=nsim, ncol=6))
values_elast = data.frame(matrix(data=NA, nrow=nsim, ncol=6))
values_forest= data.frame(matrix(data=NA, nrow=nsim, ncol=6))

colnames(values_ridge)  = c('lambda', 'Train_R_Squared','Test_R_Squared', 'Model', 'Time_Elapsed_cv', 'Fitting_Time')
colnames(values_lasso)  = c('lambda', 'Train_R_Squared','Test_R_Squared', 'Model', 'Time_Elapsed_cv', 'Fitting_Time')
colnames(values_elast)  = c('lambda', 'Train_R_Squared','Test_R_Squared', 'Model', 'Time_Elapsed_cv', 'Fitting_Time')
colnames(values_forest) = c('lambda','Train_R_Squared' ,'Test_R_Squared', 'Model', 'Time_Elapsed_cv', 'Fitting_Time')

values_ridge$Model      = 'Ridge'
values_lasso$Model      = 'Lasso'
values_elast$Model      = 'Elastic-Net'
values_forest$Model     = 'Random Forest'

############ For All Data ####################################
all_values_ridge = data.frame(matrix(data=NA, nrow=nsim, ncol=4))
all_values_lasso = data.frame(matrix(data=NA, nrow=nsim, ncol=4))
all_values_elast = data.frame(matrix(data=NA, nrow=nsim, ncol=4))
all_values_forest= data.frame(matrix(data=NA, nrow=nsim, ncol=4))

colnames(all_values_ridge)  = c('all_R_Squared', 'Model', 'Time_Elapsed_cv', 'Fitting_Time')
colnames(all_values_lasso)  = c('all_R_Squared', 'Model', 'Time_Elapsed_cv', 'Fitting_Time')
colnames(all_values_elast)  = c('all_R_Squared', 'Model', 'Time_Elapsed_cv', 'Fitting_Time')
colnames(all_values_forest) = c('all_R_Squared', 'Model', 'Time_Elapsed_cv', 'Fitting_Time')

all_values_ridge$Model      = 'Ridge'
all_values_lasso$Model      = 'Lasso'
all_values_elast$Model      = 'Elastic-Net'
all_values_forest$Model     = 'Random Forest'


######## Residuals for one of 100 samples ###########3

residuals_train           = data.frame(matrix(data=NA, nrow=n_train, ncol=4))
colnames(residuals_train) = c("ridge_r_train","lasso_r_train","elnet_r_train","rf_r_train")

residuals_test            = data.frame(matrix(data=NA, nrow=n_test, ncol=4))
colnames(residuals_test) = c("ridge_r_test","lasso_r_test","elnet_r_test","rf_r_test")

# Let's try a better way to store residuals for one of the 100 samples
residuals_df = data.frame(matrix(data=NA, nrow=4*n , ncol=3))
colnames(residuals_df) = c("Residuals","Subset","Model")

#i =1
                                                                                                          #**************** FOR LOOP STARTS ********************
for (i in 1:nsim){
  cat("****************************************************** CROSS VALIDATION SIMULATION",i,"************************************\n")

  X.test = X_all[i.mix[i,],][1:n_test, ]       #i.mix[i,] is the shuffling of 1:n for the ith simulation
  y.test = y_all[i.mix[i,]][1:n_test]          # the first n_test observation of the shuffled data will be test set
  X.train = X_all[i.mix[i,],][-(1:n_test), ]
  y.train = y_all[i.mix[i,]][-(1:n_test)]
  
  #_________________________________________________________________________________
  #_______________________K-fold cross validation for Ridge Test Train _____________
  #_________________________________________________________________________________
  time.start           = Sys.time()   ## time ridge for tuning
  ridge_cv             = cv.glmnet(x = X.train, y=y.train, alpha = 0, intercept = TRUE, nfolds = 10)
  values_ridge[i,1]    = ridge_cv$lambda.min ## storing the best lambda
  values_ridge[i,5]    = Sys.time() - time.start  ## ## end time recorded for cv
  ## time for single fitting
  time.start  = Sys.time()
  ridge_best  =     glmnet(x = X.train, y=y.train, lambda =  ridge_cv$lambda.min, alpha = 0, intercept = TRUE
                                            )
  beta0.hat.ridge      =     ridge_best$a0
  beta.hat.ridge       =     as.vector(ridge_best$beta)
#  beta0.hat.ridge_    =     ridge_best$a0[ridge_best$lambda==ridge_cv$lambda.min]
  beta_hat_ridge_[,i]  =     as.vector(ridge_best$beta[ ,ridge_best$lambda==ridge_cv$lambda.min])  #Alternatively
                                  
  y.train.hat.ridge    =     predict(ridge_best, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat.ridge     =     predict(ridge_best, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  
  values_ridge[i,2]  =     1-mean((y.train - y.train.hat.ridge)^2)/mean((y.train - mean(y.train))^2)
  values_ridge[i,3]  =     1-mean((y.test - y.test.hat.ridge)^2)/mean((y.test - mean(y.test))^2)
  values_ridge[i,6]  =     Sys.time() - time.start ## end time recorded
  
  #______________________________ All Data K-fold for Ridge  _______________________
  tic("Ridge CV time- all data" )
  all.time.start.rg.cv           = Sys.time()   ## time ridge for tuning
  all_ridge_cv             = cv.glmnet(x = X_all, y=y_all, alpha = 0, intercept = TRUE, nfolds = 10)
  all_values_ridge[i,3]    = Sys.time() - all.time.start.rg.cv  ## ## end time recorded for cv
  toc()
  ## time for single fitting
  tic("Ridge single fitting- all data")
  all.time.start.rg.sg  = Sys.time()
  all_ridge_best  =     glmnet(x = X_all, y=y_all, lambda =  all_ridge_cv$lambda.min, alpha = 0, intercept = TRUE)
  beta0.hat.ridge.all      =     all_ridge_best$a0
  beta.hat.ridge.all       =     as.vector(all_ridge_best$beta)
  #  beta0.hat.ridge_    =     ridge_best$a0[ridge_best$lambda==ridge_cv$lambda.min]
  beta_hat_ridge_all_[,i]  =     as.vector(all_ridge_best$beta[ ,all_ridge_best$lambda==all_ridge_cv$lambda.min])  #Alternatively
  y.hat.ridge.all    =     predict(all_ridge_best, newx = X_all, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  all_values_ridge[i,1]  =     1-mean((y_all - y.hat.ridge.all)^2)/mean((y_all - mean(y_all))^2)
  all_values_ridge[i,4]  =     Sys.time() - all.time.start.rg.sg ## end time recorded
  toc()
  #______________________________ End of All Data K-fold for Ridge  _______________________
  
  #_________________________________ end of K-fold for Ridge  _______________________
  
  
  
  #_________________________________________________________________________________
  #_______________________K-fold cross validation for Lasso   ______________________
  #_________________________________________________________________________________
  
  time.start    = Sys.time()   ## time lasso for tuning
  lasso_cv      =     cv.glmnet(x = X.train, y=y.train,
                              alpha = 1,
                              intercept = TRUE,
 
                              nfolds = 10)

  values_lasso[i,1]  = lasso_cv$lambda.min     ## storing the best lambda

  values_lasso[i,5]  = Sys.time() - time.start  ## ## end time recorded for cv
  
  
  ## time for single fitting
  time.start  = Sys.time()
  lasso_best  =     glmnet(x = X.train, y=y.train,
                           lambda =  lasso_cv$lambda.min,
                           alpha = 1,
                           intercept = TRUE
                                            )
  beta0.hat.lasso       =    lasso_best$a0
  beta.hat.lasso        =    as.vector(lasso_best$beta)
#  beta0.hat.lasso_     =    lasso_best$a0[lasso_best$lambda==lasso_cv$lambda.min]
  beta_hat_lasso_[,i]   =    as.vector(lasso_best$beta[ ,lasso_best$lambda==lasso_cv$lambda.min])

# prob.test             =    exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
  y.train.hat.lasso     =    predict(lasso_best, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat.lasso      =    predict(lasso_best, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  
  values_lasso[i,2]   =    1-mean((y.train - y.train.hat.lasso)^2)/mean((y.train - mean(y.train))^2)
  values_lasso[i,3]   =    1-mean((y.test - y.test.hat.lasso)^2)/mean((y.test - mean(y.test))^2)
  values_lasso[i,6]   =    Sys.time() - time.start ## end time recorded
  
  #______________________________ All Data K-fold for Lasso  _______________________
  tic("Lasso CV time- all data")
  all.time.start.ls.cv           = Sys.time()   ## time ridge for tuning
  all_lasso_cv             = cv.glmnet(x = X_all, y=y_all, alpha = 1, intercept = TRUE, nfolds = 10)
  all_values_lasso[i,3]    = Sys.time() - all.time.start.ls.cv  ## ## end time recorded for cv
  toc()
  ## time for single fitting
  tic("Lasso single fitting- all data")
  all.time.start.ls.sg  = Sys.time()
  all_lasso_best  =     glmnet(x = X_all, y=y_all, lambda =  all_lasso_cv$lambda.min, alpha = 1, intercept = TRUE)
  beta0.hat.lasso.all      =     all_lasso_best$a0
  beta.hat.lasso.all       =     as.vector(all_lasso_best$beta)
  #  beta0.hat.ridge_    =     ridge_best$a0[ridge_best$lambda==ridge_cv$lambda.min]
  beta_hat_lasso_all_[,i]  =     as.vector(all_lasso_best$beta[ ,all_lasso_best$lambda==all_lasso_cv$lambda.min])  #Alternatively
  y.hat.lasso.all    =     predict(all_lasso_best, newx = X_all, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  all_values_lasso[i,1]  =     1-mean((y_all - y.hat.lasso.all)^2)/mean((y_all - mean(y_all))^2)
  all_values_lasso[i,4]  =     Sys.time() - all.time.start.ls.sg ## end time recorded
  toc()
  #______________________________ End of All Data K-fold for Lasso  _______________________
  
  #_________________________________ end of K-fold for lasso  _______________________
  
  
  
  #_________________________________________________________________________________
  #_________________K-fold cross validation for Elastic Net   ______________________
  #_________________________________________________________________________________
  time.start      = Sys.time()   ## time ridge for tuning
  elast_cv      =     cv.glmnet(x = X.train, y=y.train,alpha = 0.5, intercept = TRUE, nfolds = 10)

  values_elast[i,1]    =        elast_cv$lambda.min     ## storing the best lambda
# auc_elast[i,2]       =        max(elast_cv$cvm)        ## the best lambda maximize the cross validation auc measure
  values_elast[i,5]    =        Sys.time() - time.start  ## ## end time recorded for cv
  
  elast_cv_all      =     cv.glmnet(x = X.train, y=y.train,alpha = 0.5, intercept = TRUE, nfolds = 10)
  
  
  
  ## time for single fitting
  time.start  = Sys.time()
      elast_best        =     glmnet(x = X.train, y=y.train,
                           lambda =  elast_cv$lambda.min,
                           alpha = 0.5,
                           intercept = TRUE,
                                        )
  beta0.hat.elast       =     elast_best$a0
  beta.hat.elast        =     as.vector(elast_best$beta)
#  beta0.hat.elast_     =     elast_best$a0[elast_best$lambda==elast_cv$lambda.min]
  beta_hat_elast_[,i]   =     as.vector(elast_best$beta[ ,elast_best$lambda==elast_cv$lambda.min])
                                     #Alternatively
  # prob.test           =     exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
  y.train.hat.elastic   =     predict(elast_best, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat.elastic    =     predict(elast_best, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  
  values_elast[i,2]  =     1-mean((y.train - y.train.hat.elastic)^2)/mean((y.train - mean(y.train))^2)
  values_elast[i,3]  =     1-mean((y.test -  y.test.hat.elastic)^2)/mean((y.test - mean(y.test))^2)
  values_elast[i,6]  =     Sys.time() - time.start ## end time recorded
  
  #______________________________ All Data K-fold for Elastic-Net  _______________________
  tic('El-net CV time- all data')
  all.time.start.el.cv           = Sys.time()   ## time ridge for tuning
  all_elast_cv             = cv.glmnet(x = X_all, y=y_all, alpha = 0.5, intercept = TRUE, nfolds = 10)
  all_values_elast[i,3]    = Sys.time() - all.time.start.el.cv  ## ## end time recorded for cv
  toc()
  ## time for single fitting
  tic("El-net single fitting- all data")
  all.time.start.el.sg  = Sys.time()
  all_elast_best  =     glmnet(x = X_all, y=y_all, lambda =  all_elast_cv$lambda.min, alpha = 0.5, intercept = TRUE)
  beta0.hat.elast.all      =     all_elast_best$a0
  beta.hat.elast.all       =     as.vector(all_elast_best$beta)
  #  beta0.hat.ridge_    =     ridge_best$a0[ridge_best$lambda==ridge_cv$lambda.min]
  beta_hat_elast_all_[,i]  =     as.vector(all_elast_best$beta[ ,all_elast_best$lambda==all_elast_cv$lambda.min])  #Alternatively
  y.hat.elast.all    =     predict(all_elast_best, newx = X_all, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  all_values_elast[i,1]  =     1-mean((y_all - y.hat.elast.all)^2)/mean((y_all - mean(y_all))^2)
  all_values_elast[i,4]  =     Sys.time() - all.time.start.el.sg ## end time recorded
  toc()
  #______________________________ End of All Data K-fold for Elastic-Net  _______________________
  
  #_____________________________ end of K-fold for Elastic Net  _______________________
  
  ##_________________________________________________________________________________
  ##____________________________ random forest ______________________________________
  ##_________________________________________________________________________________
  
  time.start            =   Sys.time() #time random forest
  rf_model              =   randomForest( X.train, y.train, mtry = floor(sqrt(p)), importance = TRUE)
  y.train.hat.RForest   =   as.vector(predict( rf_model, X.train))
  y.test.hat.RForest    =   as.vector(predict( rf_model, X.test)) 
  
  values_forest[i,2]    =   1 - mean((y.train - y.train.hat.RForest)^2)  /(mean((y.train - mean(y.train))^2))
  values_forest[i,3]    =   1 - mean((y.test  - y.test.hat.RForest )^2)/(mean((y.test - mean(y.test))^2))
  values_forest[i,5]    =   Sys.time() - time.start  ## time recorded
  values_forest[i,6]    =   values_forest[i,5]          ## since there is no tuning for random forest
  
  #______________________________ All Data K-fold for Random Forest   _______________________
  tic("RF single fitting- all data")
  time.start.rf.all             =   Sys.time() #time random forest
  rf_model.all           =   randomForest( X_all, y_all, mtry = floor(sqrt(p)), importance = TRUE)
  y.hat.RForest.all      =   as.vector(predict( rf_model.all, X_all))
  all_values_forest[i,1] =   1 - mean((y_all - y.hat.RForest.all)^2)  /(mean((y_all - mean(y_all))^2))
  all_values_forest[i,4]     =   Sys.time() - time.start.rf.all  ## time recorded
  toc()
  #______________________________ End of All Data K-fold for Random Forest   _______________________
  
  ##_________________________________________________________________________________
  ##____________________________ end of random forest _______________________________
  ##_________________________________________________________________________________
  
  ### storing betas values for linear model  and importance for random forest
  nsim_betas_ridge[i, ]         =  ridge_best$beta
  nsim_betas_lasso[i, ]         =  lasso_best$beta
  nsim_betas_elast[i, ]         =  elast_best$beta
  nsim_betas_forest[i, ]        =  rf_model$importance[,1]
  
  # or equivalently 
  beta_hat_forest_[,i ]         =  as.vector(rf_model$importance[,1])

  ##_________________________________________________________________________________                      ####### Single Fitting ################
  ##____________________________ single fitting  _______________________________
  ##_________________________________________________________________________________

  if ( i == nsim){                                                           
    residuals_train[,"ridge_r_train"]   = y.train - y.train.hat.ridge
    residuals_train[,"lasso_r_train"]   = y.train - y.train.hat.lasso
    residuals_train[,"elnet_r_train"]   = y.train - y.train.hat.elastic
    residuals_train[,"rf_r_train"]      = y.train - y.train.hat.RForest
  
    residuals_test[,"ridge_r_test"]    = y.test  - y.test.hat.ridge
    residuals_test[,"lasso_r_test"]    = y.test  - y.test.hat.lasso
    residuals_test[,"elnet_r_test"]    = y.test  - y.test.hat.elastic
    residuals_test[,"rf_r_test"]       = y.test  - y.test.hat.RForest
    
    residuals_df$Residuals[1:n_train]                                   = residuals_train[,"ridge_r_train"] ; residuals_df$Subset =rep("Train",n_train); residuals_df$Model = rep("Ridge",n_train)
    residuals_df$Residuals[(n_train+1):(2*n_train)]                     = residuals_train[,"lasso_r_train"] ; residuals_df$Subset =rep("Train",n_train); residuals_df$Model = rep("Lasso",n_train)
    residuals_df$Residuals[(2*n_train+1):(3*n_train)]                   = residuals_train[,"elnet_r_train"] ; residuals_df$Subset =rep("Train",n_train); residuals_df$Model = rep("Elastic Net",n_train)
    residuals_df$Residuals[(3*n_train+1):(4*n_train)]                   = residuals_train[,"rf_r_train"]    ; residuals_df$Subset =rep("Train",n_train); residuals_df$Model = rep("Random Forest",n_train)
    residuals_df$Residuals[(4*n_train+1):(4*n_train+n_test)]            = residuals_test[,"ridge_r_test"]   ; residuals_df$Subset =rep("Test",n_test)  ; residuals_df$Model = rep("Ridge",n_test)
    residuals_df$Residuals[(4*n_train+n_test+1):(4*n_train+2*n_test)]   = residuals_test[,"lasso_r_test"]   ; residuals_df$Subset =rep("Test",n_test)  ; residuals_df$Model = rep("Lasso",n_test)
    residuals_df$Residuals[(4*n_train+2*n_test+1):(4*n_train+3*n_test)] = residuals_test[,"elnet_r_test"]   ; residuals_df$Subset =rep("Test",n_test)  ; residuals_df$Model = rep("Elastic Net",n_test)
    residuals_df$Residuals[(4*n_train+3*n_test+1):(4*n)]                = residuals_test[,"rf_r_test"]      ; residuals_df$Subset =rep("Test",n_test)  ; residuals_df$Model = rep("Random Forest",n_test)
    
    
    residuals_train_plot =  residuals_train %>% select(ridge_r_train, lasso_r_train, elnet_r_train, rf_r_train) %>%
    pivot_longer(., cols = c(ridge_r_train, lasso_r_train, elnet_r_train, rf_r_train ), names_to = "Models", values_to = "Residuals")%>%
    ggplot(aes(x = Models, y = Residuals)) +
    geom_boxplot()+
    labs(title = "Train")+
    theme(plot.title = element_text(hjust = 0.6) , axis.text.x = element_text(angle = 45, hjust=1))+
    labs(x = "Models", y = "Residuals")
    residuals_train_plot
    
    residuals_test_plot =  residuals_test %>% select(ridge_r_test, lasso_r_test, elnet_r_test, rf_r_test) %>%
    pivot_longer(., cols = c(ridge_r_test, lasso_r_test, elnet_r_test, rf_r_test ), names_to = "Models", values_to = "Residuals")%>%
    ggplot(aes(x = Models, y = Residuals)) +
    geom_boxplot()+
    labs(title = "Test")+
    theme(plot.title = element_text(hjust = 0.6) , axis.text.x = element_text(angle = 45, hjust=1))+
    labs(x = "Models", y = "Residuals")
    residuals_test_plot
  
    grid.arrange(residuals_test_plot, residuals_train_plot, nrow = 1, ncol= 2, top = "Residuals Boxplot")
    
    rg.time.start.one.cv  = Sys.time()
    rg_cv_one          =     cv.glmnet(x = X.train, y=y.train, alpha = 0, intercept = TRUE, nfolds = 10)
    rg.time.one.cv     =     round(Sys.time() - rg.time.start.one.cv, digits = 2)
    plot(rg_cv_one)
    title(main=paste0("Ridge CV Time: ",rg.time.one.cv , " seconds"), outer=TRUE, line=-1)
    
    ls.time.start.one.cv  = Sys.time()
    ls_cv_one          =     cv.glmnet(x = X.train, y=y.train, alpha = 1, intercept = TRUE, nfolds = 10)
    ls.time.one.cv     =    round(Sys.time() - ls.time.start.one.cv, digits=2)
    plot(ls_cv_one)
    title(main=paste0("Lasso CV Time: ", ls.time.one.cv, " seconds"), outer=TRUE, line=-1)
    
    tic()
    el.time.start.one.cv  = Sys.time()
    elast_cv_one          =     cv.glmnet(x = X.train, y=y.train, alpha = 0.5, intercept = TRUE, nfolds = 10)
    el.time.one.cv        =    round(Sys.time() - el.time.start.one.cv, digits= 2)
    plot(elast_cv_one)
    title(main=paste0("Elastic-Net CV Time: ", el.time.one.cv, " seconds"), outer=TRUE, line=-1)
    toc()
  }
  ##_________________________________________________________________________________                      ####### End of Single Fitting ################
  ##____________________________ End of single fitting  _______________________________
  ##_________________________________________________________________________________
  ##_________________________________________________________________________________                      ####### End of Single Fitting ################
  ##____________________________ End of For Loop  _______________________________
  ##_________________________________________________________________________________
}
                                                                                                    #**************** END OF FOR LOOP ********************
## to build a more robust standard deviation for apply function
sd_na_rm = function(x){
  if (   sum(!is.na(x)) < 2 ) return(0)
  sd(x, na.rm = T) ## else return 
}

# this is the standard order for beta order for bar plots
base_order    = colMeans(nsim_betas_elast, na.rm = T)
beta_label    = reorder( colnames(X.train), -abs(base_order))  ## sort the beta values in descending order

################################## Alternatively: #########################################
ridge.sd_     =    apply(beta_hat_ridge_, 1, "sd")
lasso.sd_     =    apply(beta_hat_lasso_, 1, "sd")
elastic.sd_   =    apply(beta_hat_elast_ , 1, "sd") 
rforest.sd_   =    apply(beta_hat_forest_, 1, "sd")

ridge.sd_     =    as.data.frame(ridge.sd_ ,col.names= "error")
lasso.sd_     =    as.data.frame(lasso.sd_ ,col.names= "error")
elastic.sd_   =    as.data.frame(elastic.sd_ ,col.names= "error")
rforest.sd_   =    as.data.frame(rforest.sd_ ,col.names= "error")

mean.beta_hat_ridge_ = as.data.frame(rowMeans(beta_hat_ridge_))
mean.beta_hat_lasso_ = as.data.frame(rowMeans(beta_hat_lasso_))
mean.beta_hat_elast_ = as.data.frame(rowMeans(beta_hat_elast_))
mean.beta_hat_forest_ = as.data.frame(rowMeans(beta_hat_forest_))


#ORDER BETAS
betaS.rg      =  cbind(beta_hat_ridge_  , mean.beta_hat_ridge_, 2*ridge.sd_ , c(1:p)) 
betaS.ls      =  cbind(beta_hat_lasso_  , mean.beta_hat_lasso_, 2*lasso.sd_ , c(1:p))
betaS.en      =  cbind(beta_hat_elast_  , mean.beta_hat_elast_, 2*rforest.sd_ , c(1:p))
betaS.rf      =  cbind(beta_hat_forest_ , mean.beta_hat_forest_,2*rforest.sd_ , c(1:p))

colnames(betaS.rg) = c(1:nsim,"avg_betaS", "error", "feature")
colnames(betaS.ls) = c(1:nsim,"avg_betaS", "error", "feature")
colnames(betaS.en) = c(1:nsim,"avg_betaS", "error", "feature")
colnames(betaS.rf) = c(1:nsim,"avg_betaS", "error", "feature")

betaS.rg$feature     =  factor(betaS.rg$feature, levels = betaS.en$feature[order(betaS.en$avg_betaS, decreasing = TRUE)])
betaS.ls$feature     =  factor(betaS.ls$feature, levels = betaS.en$feature[order(betaS.en$avg_betaS, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.en$feature[order(betaS.en$avg_betaS, decreasing = TRUE)])
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.en$avg_betaS, decreasing = TRUE)])

#y_elast =   colMeans(nsim_betas_elast, na.rm = T) 
#err_elast = 2*apply(nsim_betas_elast, 2, sd_na_rm)
y_elast_col_cond2 = ifelse(betaS.en$avg_betaS > 0 ,"positive","negative")
y_ridge_col_cond2 = ifelse(betaS.rg$avg_betaS > 0 ,"positive","negative")
y_lasso_col_cond2 = ifelse(betaS.ls$avg_betaS > 0 ,"positive","negative")
y_rf_col_cond2    = ifelse(betaS.rf$avg_betaS > 0 ,"positive","negative")


rgPlot =  ggplot(betaS.rg, aes(x=feature, y=avg_betaS, fill = y_ridge_col_cond2 )) +
  geom_bar(stat = "identity", colour="black")    +
  geom_errorbar(aes(ymin=avg_betaS-error, ymax=avg_betaS+error,), width=.2)+
  xlab('Feature')+
  ylab('Ridge Betas')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  scale_y_continuous(limits = c(-0.75,0.75)) +
  scale_fill_manual(values = c("#FFDDDD","#CCEEFF"), guide = FALSE)
rgPlot

lsPlot =  ggplot(betaS.ls, aes(x=feature, y=avg_betaS ,fill = y_lasso_col_cond2)) +
  geom_bar(stat = "identity", colour="black")    +
  geom_errorbar(aes(ymin=avg_betaS-error, ymax=avg_betaS+error), width=.2)+
  xlab('Feature')+
  ylab('Lasso Betas')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  scale_y_continuous(limits = c(-0.75,0.75)) +
  scale_fill_manual(values = c("#FFDDDD","#CCEEFF"), guide = FALSE)
lsPlot

enPlot =  ggplot(betaS.en, aes(x=feature, y=avg_betaS, fill = y_elast_col_cond2)) +
  geom_bar(stat = "identity", colour="black")    +
  geom_errorbar(aes(ymin=avg_betaS-error, ymax=avg_betaS+error,), width=.2)+
  xlab('Feature')+
  ylab('El-Net Betas')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  scale_y_continuous(limits = c(-0.75,0.75)) +
  scale_fill_manual(values = c("#FFDDDD","#CCEEFF"), guide = FALSE)
enPlot

rfPlot =  ggplot(betaS.rf, aes(x=feature, y=avg_betaS, fill = y_rf_col_cond2)) +
  geom_bar(stat = "identity", colour="black")    +
  geom_errorbar(aes(ymin=avg_betaS-error, ymax=avg_betaS+error,), width=.2)+
  xlab('Feature')+
  ylab('R-Forest Betas')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  scale_fill_manual(values = c("#FFDDDD","#CCEEFF"), guide = FALSE)
rfPlot



#ggarrange(plotlist = c(enPlot,rgPlot,lsPlot,rfPlot),ncol = NULL,nrow = NULL,labels = NULL,label.x = 0,label.y = 1,hjust = -0.5,vjust = 1.5,font.label = list(size = 14, color = "black", face = "bold", family = NULL),align = c("none", "h", "v", "hv"),widths = 1,heights = 1,legend = NULL,common.legend = TRUE,legend.grob = NULL)
grid.arrange(enPlot,rgPlot, lsPlot, rfPlot,  nrow = 4, ncol= 1)

#########################  Doing the Same for All Data ###########################################
ridge.sd_all     =    apply(beta_hat_ridge_all_, 1, "sd")
lasso.sd_all     =    apply(beta_hat_lasso_all_, 1, "sd")
elastic.sd_all   =    apply(beta_hat_elast_all_ , 1, "sd") 
rforest.sd_all   =    apply(beta_hat_forest_all_, 1, "sd")

ridge.sd_all     =    as.data.frame(ridge.sd_all ,col.names= "error")
lasso.sd_all     =    as.data.frame(lasso.sd_all ,col.names= "error")
elastic.sd_all   =    as.data.frame(elastic.sd_all ,col.names= "error")
rforest.sd_all   =    as.data.frame(rforest.sd_all ,col.names= "error")

mean.beta_hat_ridge_all = as.data.frame(rowMeans(beta_hat_ridge_all_))
mean.beta_hat_lasso_all = as.data.frame(rowMeans(beta_hat_lasso_all_))
mean.beta_hat_elast_all = as.data.frame(rowMeans(beta_hat_elast_all_))
mean.beta_hat_forest_all = as.data.frame(rowMeans(beta_hat_forest_all_))


#ORDER BETAS
betaS.rg.all      =  cbind(beta_hat_ridge_all_  , mean.beta_hat_ridge_all, 2*ridge.sd_all , c(1:p)) 
betaS.ls.all      =  cbind(beta_hat_lasso_all_  , mean.beta_hat_lasso_all, 2*lasso.sd_all , c(1:p))
betaS.en.all      =  cbind(beta_hat_elast_all_  , mean.beta_hat_elast_all, 2*rforest.sd_all , c(1:p))
betaS.rf.all      =  cbind(beta_hat_forest_all_ , mean.beta_hat_forest_all,2*rforest.sd_all , c(1:p))

colnames(betaS.rg.all) = c(1:nsim,"avg_betaS", "error", "feature")
colnames(betaS.ls.all) = c(1:nsim,"avg_betaS", "error", "feature")
colnames(betaS.en.all) = c(1:nsim,"avg_betaS", "error", "feature")
colnames(betaS.rf.all) = c(1:nsim,"avg_betaS", "error", "feature")

betaS.rg.all$feature     =  factor(betaS.rg.all$feature, levels = betaS.en.all$feature[order(betaS.en.all$avg_betaS, decreasing = TRUE)])
betaS.ls.all$feature     =  factor(betaS.ls.all$feature, levels = betaS.en.all$feature[order(betaS.en.all$avg_betaS, decreasing = TRUE)])
betaS.en.all$feature     =  factor(betaS.en.all$feature, levels = betaS.en.all$feature[order(betaS.en.all$avg_betaS, decreasing = TRUE)])
betaS.rf.all$feature     =  factor(betaS.rf.all$feature, levels = betaS.rf.all$feature[order(betaS.en.all$avg_betaS, decreasing = TRUE)])

#y_elast =   colMeans(nsim_betas_elast, na.rm = T) 
#err_elast = 2*apply(nsim_betas_elast, 2, sd_na_rm)
y_elast_cond2_all  = ifelse(betaS.en.all$avg_betaS > 0 ,"positive","negative")
y_ridge_cond2_all  = ifelse(betaS.rg.all$avg_betaS > 0 ,"positive","negative")
y_lasso_cond2_all = ifelse(betaS.ls.all$avg_betaS > 0 ,"positive","negative")
y_rf_cond2_all = ifelse(betaS.rf.all$avg_betaS > 0 ,"positive","negative")


rgPlot.all =  ggplot(betaS.rg.all, aes(x=feature, y=avg_betaS, fill = y_ridge_cond2_all )) +
  geom_bar(stat = "identity", colour="black")    +
  geom_errorbar(aes(ymin=avg_betaS-error, ymax=avg_betaS+error,), width=.2)+
  xlab('Feature')+
  ylab('Ridge Betas')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  scale_y_continuous(limits = c(-0.75,0.75)) +
  scale_fill_manual(values = c("#FFDDDD","#CCEEFF"), guide = FALSE)
rgPlot.all

lsPlot.all =  ggplot(betaS.ls.all, aes(x=feature, y=avg_betaS ,fill = y_lasso_cond2_all)) +
  geom_bar(stat = "identity", colour="black")    +
  geom_errorbar(aes(ymin=avg_betaS-error, ymax=avg_betaS+error), width=.2)+
  xlab('Feature')+
  ylab('Lasso Betas')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  scale_y_continuous(limits = c(-0.75,0.75)) +
  scale_fill_manual(values = c("#FFDDDD","#CCEEFF"), guide = FALSE)
lsPlot.all

enPlot.all =  ggplot(betaS.en, aes(x=feature, y=avg_betaS, fill = y_elast_cond2_all)) +
  geom_bar(stat = "identity", colour="black")    +
  geom_errorbar(aes(ymin=avg_betaS-error, ymax=avg_betaS+error,), width=.2)+
  xlab('Feature')+
  ylab('El-Net Betas')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  scale_y_continuous(limits = c(-0.75,0.75)) +
  scale_fill_manual(values = c("#FFDDDD","#CCEEFF"), guide = FALSE)
enPlot.all

rfPlot.all =  ggplot(betaS.rf.all, aes(x=feature, y=avg_betaS, fill = y_rf_cond2_all)) +
  geom_bar(stat = "identity", colour="black")    +
  geom_errorbar(aes(ymin=avg_betaS-error, ymax=avg_betaS+error,), width=.2)+
  xlab('Feature')+
  ylab('R-Forest Betas')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  scale_fill_manual(values = c("#FFDDDD","#CCEEFF"), guide = FALSE)
rfPlot.all

#ggarrange(plotlist = c(enPlot,rgPlot,lsPlot,rfPlot),ncol = NULL,nrow = NULL,labels = NULL,label.x = 0,label.y = 1,hjust = -0.5,vjust = 1.5,font.label = list(size = 14, color = "black", face = "bold", family = NULL),align = c("none", "h", "v", "hv"),widths = 1,heights = 1,legend = NULL,common.legend = TRUE,legend.grob = NULL)
grid.arrange(enPlot.all,rgPlot.all, lsPlot.all, rfPlot.all,  nrow = 4, ncol= 1 , top = "Beta Values for All Data")

########################################### End of Betas All data #################################################

########################################## Beta Coefficients Plot #################################
y_elast =   colMeans(nsim_betas_elast, na.rm = T) 
err_elast = 2*apply(nsim_betas_elast, 2, sd_na_rm)
y_elast_col_cond = ifelse(y_elast > 0 ,"positive","negative")

elast_Plot =  ggplot() + 
  aes(x= beta_label, y= y_elast , fill = y_elast_col_cond)+
  geom_bar(stat = "identity", colour="black")    +
  geom_errorbar(aes(ymin=y_elast-err_elast, ymax=y_elast+err_elast), width=.2) +
  ggtitle('Elastic Net Beta Values')+
  xlab('Feature')+
  ylab('Betas')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  scale_fill_manual(values = c("#FFDDDD","#CCEEFF"), guide = FALSE)
elast_Plot


y_ridge     =   colMeans(nsim_betas_ridge, na.rm = T)
err_ridge   = 2* apply(nsim_betas_ridge, 2, sd_na_rm)  ## 2*standard error
y_ridge_col_cond = ifelse(y_ridge > 0 ,"positive","negative")

ridge_Plot  =  ggplot() + aes(x= beta_label, y= y_ridge , fill= y_ridge_col_cond)+
  geom_bar(stat = "identity",  colour="black")    +
  geom_errorbar(aes(ymin=y_ridge-err_ridge, ymax=y_ridge+err_ridge), width=0.2) +
  ggtitle('Ridge Beta Values')+
  xlab('Feature')+
  ylab('Betas')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  scale_fill_manual(values = c("#FFDDDD","#CCEEFF" ), guide = FALSE)
 ridge_Plot


y_lasso     =   colMeans(nsim_betas_lasso, na.rm = T) 
err_lasso   = 2*apply(nsim_betas_lasso, 2, sd_na_rm)
y_lasso_col_cond = ifelse(y_lasso > 0 ,"positive","negative")

#clrs <- setNames(c("green", "red"), c("green", "red"))
lasso_Plot  =  ggplot() + aes(x= beta_label, y= y_lasso , fill = y_lasso_col_cond)+
  geom_bar(stat = "identity", colour="black")    +
  geom_errorbar(aes(ymin=y_lasso-err_lasso, ymax=y_lasso+err_lasso), width=.2) +
  ggtitle('Lasso Beta Values')+
  xlab('Feature')+
  ylab('Betas')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  scale_fill_manual(values = c("#FFDDDD","#CCEEFF"), guide = FALSE)
  
#  scale_colour_manual(values = clrs) +
#  scale_fill_manual(values = clrs) 
 lasso_Plot


y_rf     =   colMeans(nsim_betas_forest, na.rm = T) 
err_rf   = 2* apply(nsim_betas_forest, 2, sd_na_rm)
y_rf_col_cond = ifelse(y_rf > 0 ,"positive","negative")

rf_Plot  =  ggplot() + aes(x= beta_label, y= y_rf , fill= y_rf_col_cond)+
  geom_bar(stat = "identity", colour="black")    +
  geom_errorbar(aes(ymin=y_rf-err_rf, ymax=y_rf+err_rf), width=.2) +
  ggtitle('Random Forest Variable Importance')+
  xlab('Feature')+
  ylab('Betas')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  scale_fill_manual(values = c("#FFDDDD","#CCEEFF"), guide = FALSE)
 rf_Plot

## save these plots
ggarrange(elast_Plot, lasso_Plot, ridge_Plot,rf_Plot, nrow =4, ncol = 1)
########################################################################################################
################ Cross-Validation Plots ##############
plot(ridge_cv)
title(main=paste0("Ridge"), outer=TRUE, line=-1)
plot(lasso_cv)
title(main=paste0("Lasso"), outer=TRUE, line=-1)
plot(elast_cv)
title(main=paste0("Elastic-Net"), outer=TRUE, line=-1)

values_ridge
values_lasso
values_elast
values_forest

all_values_ridge
all_values_lasso
all_values_elast
all_values_forest

result0 = as.data.frame(rbind(values_ridge, values_lasso, values_elast))  ## without random forest
result = as.data.frame(rbind(result0, values_forest))  ## includes random forest

result0_all = as.data.frame(rbind(all_values_ridge, all_values_lasso, all_values_elast))  ## without random forest
result_all = as.data.frame(rbind(result0_all, all_values_forest))  ## includes random forest

library(ggplot2)
library(dplyr)
## average time spent
result %>%
  group_by(Model) %>%
  mutate(Average_Time_per_Simulation = mean(Time_Elapsed_cv, Average_Fitting_Time = mean(Fitting_Time))) %>%
  select(Model, Average_Time_per_Simulation) %>%
  unique()


##time spent on tuning lambda
ggplot(result0)+
  aes(y=Time_Elapsed_cv, x=Model, col=Model)+
  geom_boxplot()+
  ggtitle('Boxplot of Time Spent Tuning')+
  labs(x = "Models", y = "Cross-Validation Time")
  theme(plot.title = element_text(hjust = 0.5))

ggplot(result)+
  aes(x=Time_Elapsed_cv, col=Model, fill = Model)+
  geom_density()+
  ggtitle('Density of Time Spent Tuning')+
  theme(plot.title = element_text(hjust = 0.5))

ggplot(result)+
  aes(x=Time_Elapsed_cv, fill=Model)+
  geom_histogram()+
  ggtitle('Histogram of Time Spent Tuning')+
  theme(plot.title = element_text(hjust = 0.5))+
  facet_grid(Model~.)


## average lambda value pick by CV
library(dplyr)
result %>%
  group_by(Model) %>%
  mutate(mean_lambda = mean(lambda)) %>%
  select(Model, mean_lambda) %>% unique()

ggplot(result0)+
  aes(x=lambda, fill=Model)+
  geom_density()+
  ggtitle('Distribution of Lambda Picked by CV')+
  theme(plot.title = element_text(hjust = 0.5))+
  facet_grid(Model~.)

## doing a boxplot for in one scale
colnames(result)
train_result = result[,-3]
train_result['Train_Test'] = 'Train'
colnames(train_result)[2] = 'R_squared'

test_result = result[,-2]
test_result['Train_Test'] = 'Test'
colnames(test_result)[2] = 'R_squared'

r.squared= rbind(train_result,test_result)
colnames(r.squared)

ggplot(r.squared)+
  aes(x=Model, y = R_squared, col=Model)+
  geom_boxplot()+
  facet_grid(.~Train_Test)+
  ggtitle('Comparison Between Test and Train R-Squared')+
  labs(x = "Models", y = "R Squared")+
  theme(plot.title = element_text(hjust = 0.6) , axis.text.x = element_text(angle = 45, hjust=1))

ggplot(r.squared)+
  aes(col=Model, x = R_squared)+
  geom_density()+
  facet_grid(.~Train_Test)+
  ggtitle('Approximated PDF for Test and Train R-Squared')+
  theme(plot.title = element_text(hjust = 0.5))

grid.arrange(residuals_test_plot, residuals_train_plot, nrow = 1, ncol= 2, top = "Residuals Boxplots")

###### Presentation table for average R-Squared ##########
table_r_squared = data.frame(nrow =4, ncol=2)
colnames(table_r_squared) = c("Train R Squared(Average)" ,"Test R Squared(Average)")
#row.names(table_r_squared) = c("Ridge", "Lasso", "Elastic Net", "Random Forest")
table_r_squared[1,1] =  mean(result[result$Model == "Ridge", 2])
table_r_squared[2,1] =  mean(result[result$Model == "Lasso", 2])
table_r_squared[3,1] =  mean(result[result$Model == "Elastic-Net", 2])
table_r_squared[4,1] =  mean(result[result$Model == "Random Forest", 2])
table_r_squared[1,2] =  mean(result[result$Model == "Ridge", 3])
table_r_squared[2,2] =  mean(result[result$Model == "Lasso", 3])
table_r_squared[3,2] =  mean(result[result$Model == "Elastic-Net", 3])
table_r_squared[4,2] =  mean(result[result$Model == "Random Forest", 3])
table_r_squared

table_r_squared_all = data.frame(nrow =4, ncol=1)
colnames(table_r_squared_all) = c("R Squared All Data")
table_r_squared_all[1,1] =  mean(result_all[result$Model == "Ridge", 1])
table_r_squared_all[2,1] =  mean(result_all[result$Model == "Lasso", 1])
table_r_squared_all[3,1] =  mean(result_all[result$Model == "Elastic-Net", 1])
table_r_squared_all[4,1] =  mean(result_all[result$Model == "Random Forest", 1])
table_r_squared_all

#####################################################################
#######################Test 90% Confidence Interval and time ########
table_time_ci = data.frame(data = NA,nrow =4, ncol=4)
#colnames(table_time_ci) = c("Test R Squared Lower Bound" ,"Test R Squared Upper Bound", "CV Average Time", "Single Fitting Average Time")
#rownames(table_time_ci) = c("Ridge", "Lasso", "Elastic Net", "Random Forest")

#qt(0.975,df=length(w1$vals)-1)*sd(w1$vals)/sqrt(length(w1$vals))
table_time_ci[1,1] = mean(result[result$Model == "Ridge", 3]) - qt(0.95,df=length(result[result$Model == "Ridge", 2])-1)*sd(result[result$Model == "Ridge", 2])/sqrt(length(result[result$Model == "Ridge", 2]))
table_time_ci[2,1] = mean(result[result$Model == "Lasso", 3]) - qt(0.95,df=length(result[result$Model == "Lasso", 2])-1)*sd(result[result$Model == "Lasso", 2])/sqrt(length(result[result$Model == "Lasso", 2]))
table_time_ci[3,1] = mean(result[result$Model == "Elastic-Net", 3]) - qt(0.95,df=length(result[result$Model == "Elastic-Net", 2])-1)*sd(result[result$Model == "Elastic-Net", 2])/sqrt(length(result[result$Model == "Elastic-Net", 2]))
table_time_ci[4,1] = mean(result[result$Model == "Random Forest", 3]) - qt(0.95,df=length(result[result$Model == "Random Forest", 2])-1)*sd(result[result$Model == "Random Forest", 2])/sqrt(length(result[result$Model == "Random Forest", 2]))

table_time_ci[1,2] = mean(result[result$Model == "Ridge", 3]) + qt(0.95,df=length(result[result$Model == "Ridge", 2])-1)*sd(result[result$Model == "Ridge", 2])/sqrt(length(result[result$Model == "Ridge", 2]))
table_time_ci[2,2] = mean(result[result$Model == "Lasso", 3]) + qt(0.95,df=length(result[result$Model == "Lasso", 2])-1)*sd(result[result$Model == "Lasso", 2])/sqrt(length(result[result$Model == "Lasso", 2]))
table_time_ci[3,2] = mean(result[result$Model == "Elastic-Net", 3]) + qt(0.95,df=length(result[result$Model == "Elastic-Net", 2])-1)*sd(result[result$Model == "Elastic-Net", 2])/sqrt(length(result[result$Model == "Elastic-Net", 2]))
table_time_ci[4,2] = mean(result[result$Model == "Random Forest", 3]) + qt(0.95,df=length(result[result$Model == "Random Forest", 2])-1)*sd(result[result$Model == "Random Forest", 2])/sqrt(length(result[result$Model == "Random Forest", 2]))

table_time_ci[1,3] = mean(result[result$Model == "Ridge", 5])
table_time_ci[2,3] = mean(result[result$Model == "Lasso", 5])
table_time_ci[3,3] = mean(result[result$Model == "Elastic-Net", 5])
table_time_ci[4,3] = NA

table_time_ci[1,4] = mean(result[result$Model == "Ridge", 6])
table_time_ci[2,4] = mean(result[result$Model == "Lasso", 6])
table_time_ci[3,4] = mean(result[result$Model == "Elastic-Net", 6])
table_time_ci[4,4] = mean(result[result$Model == "Random Forest", 6])

table_time_ci

#######################Test 90% Confidence Interval and time of All Data ###########################################
table_time_ci_all = data.frame(data = NA,nrow =4, ncol=4)
colnames(table_time_ci) = c("Test R Squared Lower Bound" ,"Test R Squared Upper Bound", "CV Average Time", "Single Fitting Average Time")
#rownames(table_time_ci) = c("Ridge", "Lasso", "Elastic Net", "Random Forest")

#qt(0.975,df=length(w1$vals)-1)*sd(w1$vals)/sqrt(length(w1$vals))
table_time_ci_all[1,1] = table_time_ci[1,1]
table_time_ci_all[2,1] = table_time_ci[2,1]
table_time_ci_all[3,1] = table_time_ci[3,1]
table_time_ci_all[4,1] = table_time_ci[4,1]

table_time_ci_all[1,2] = table_time_ci[1,2]
table_time_ci_all[2,2] = table_time_ci[2,2]
table_time_ci_all[3,2] = table_time_ci[3,2]
table_time_ci_all[4,2] = table_time_ci[4,2]

table_time_ci_all[1,3] = mean(result_all[result_all$Model == "Ridge", 3])
table_time_ci_all[2,3] = mean(result_all[result$Modelall == "Lasso", 3])
table_time_ci_all[3,3] = mean(result_all[result_all$Model == "Elastic-Net", 3])
table_time_ci_all[4,3] = NA

table_time_ci_all[1,4] = mean(result_all[result_all$Model == "Ridge", 4])
table_time_ci_all[2,4] = mean(result_all[result_all$Model == "Lasso", 4])
table_time_ci_all[3,4] = mean(result_all[result_all$Model == "Elastic-Net", 4])
table_time_ci_all[4,4] = mean(result_all[result_all$Model == "Random Forest", 4])

table_time_ci_all



## lets save the result data for analysis
write.csv(result, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/result.csv")
write.csv(result0, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/result0.csv")
write.csv(table_r_squared, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/table_r_squared.csv")
write.csv(table_time_ci, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/table_time_ci.csv")

write.csv(result_all, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/result_all.csv")
write.csv(result0_all, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/result0_all.csv")
write.csv(table_r_squared_all, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/table_r_squared_all.csv")
write.csv(table_time_ci_all, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/table_time_ci_all.csv")

write.csv(nsim_betas_ridge, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/nsim_betas_ridge.csv")
write.csv(nsim_betas_elast, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/nsim_betas_lasso.csv")
write.csv(nsim_betas_lasso, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/nsim_betas_elast.csv")
write.csv(nsim_betas_forest, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/nsim_betas_forest.csv")

write.csv(beta_hat_ridge_, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/beta_hat_ridge_.csv")
write.csv(beta_hat_lasso_, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/beta_hat_ridge_.csv")
write.csv(beta_hat_elast_, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/beta_hat_ridge_.csv")
write.csv(beta_hat_forest_, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/beta_hat_ridge_.csv")

write.csv(beta_hat_ridge_all_, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/beta_hat_ridge_all_.csv")
write.csv(beta_hat_lasso_all_, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/beta_hat_ridge_all_.csv")
write.csv(beta_hat_elast_all_, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/beta_hat_ridge_all_.csv")
write.csv(beta_hat_forest_all_, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/beta_hat_ridge_all_.csv")

write.csv(ridge.sd_, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/ridge.sd_.csv") 
write.csv(lasso.sd_, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/lasso.sd_.csv")
write.csv(elastic.sd_, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/elastic.sd_.csv")
write.csv(rforest.sd_, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/rforest.sd_.csv")

write.csv(ridge.sd_all, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/ridge.sd_all.csv") 
write.csv(lasso.sd_all, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/lasso.sd_all.csv")
write.csv(elastic.sd_all, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/elastic.sd_all.csv")
write.csv(rforest.sd_all, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/rforest.sd_all.csv")

write.csv(betaS.rg, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/betaS.rg.csv") 
write.csv(betaS.ls, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/betaS.ls.csv")
write.csv(betaS.en, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/betaS.en.csv")
write.csv(betaS.rf, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/betaS.rf.csv")

write.csv(betaS.rg.all, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/betaS.rg.all.csv") 
write.csv(betaS.ls.all, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/betaS.ls.all.csv")
write.csv(betaS.en.all, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/betaS.en.all.csv")
write.csv(betaS.rf.all, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/betaS.rf.all.csv")


write.csv(residuals_train, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/residuals_train.csv")
write.csv(residuals_test, file = "/Users/behzadpouyanfar/Desktop/STA.9890 Statistical Learning/PROJECT 9890/3/3 CSV Output/residuals_test.csv")

#################

