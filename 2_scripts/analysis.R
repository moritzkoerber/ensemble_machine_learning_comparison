# -------------- Packages --------------
library("dplyr")
library("ggplot2")
library("caret")
library("rlist")
library("xgboost")
library("mlr")
library("visdat")
library("ISLR")
library("xgboost")
library("gbm")

# -------------- Read Data --------------
df <- read.csv("1_data/pml-training.csv", na.strings = c("NA", "NaN", "", "#DIV/0!"), row.names = 1)

set.seed(31)

# -------------- clean data set --------------
# remove obvious non-predictive variables
df %>% select(-c(raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp)) -> df

#  extract id for blocking
id <- df$user_name
df %>% select(-user_name) -> df

# remove zero variance/near zero variance predictors
nzv <- nearZeroVar(df)
df <- df[-nzv]

df <- removeConstantFeatures(df, perc = .02)

# crop predictors with mostly NAs
df %>%
  select(everything()) %>%
  summarise_all(funs(sum(is.na(.)) / length(.))) -> p

# check these variables
vis_miss(df[which(p > 0.975)],
  sort_miss = TRUE, warn_large_data = F
)

# remove them if sensible
df[which(p > 0.975)] <- NULL

# visualize data
vis_dat(df, warn_large_data = F)

# are mixed data types in one variable?
vis_guess(df)

# remove highly correlated predictors NOT FINISHED --> cor[70] is -.99
nums <- select_if(df, is.numeric)
descrCor <- cor(nums)

highCorr <- sum(na.omit(abs(descrCor[upper.tri(descrCor)])) >= .98)

na.omit(descrCor[upper.tri(descrCor)])[which(na.omit(abs(descrCor[upper.tri(descrCor)])) >= .98, arr.ind = TRUE)]

which(na.omit(abs(descrCor)) >= .98 & na.omit(abs(descrCor)) < 1, arr.ind = TRUE)

# 2 variables with high correlation --> consider in model evaluation
rm <- findCorrelation(na.omit(descrCor), cutoff = .98, verbose = T, exact = T, names = T)

# find linear combinations
findLinearCombos(nums)

# save cleaned data
saveRDS(df, "cleaned_data.rds")

# read cleaned data if necessary
# df <- readRDS("1_data/cleaned_data.rds")

# -------------- plot features --------------
nums <- unlist(lapply(df, is.numeric))
featurePlot(x = df[nums], y = df$classe, plot = "strip")

# ---------- Train/Test-set ----------
# setup train and test set
train <- sample(nrow(df), 0.8 * nrow(df), replace = F)
train <- createDataPartition(df$classe, p = 0.8, list = F)
training <- df[train, ]
test <- df[-train, ]

### convert facotrs to dummy?

# normalize numeric predictors
preProcValues <- preProcess(training, method = c("center", "scale")) ### avoid overfit? always only on training?
training <- predict(preProcValues, training)
test <- predict(preProcValues, test)

## faster
# pp_no_nzv <- preProcess(schedulingData[, -8],
#                         method = c("center", "scale", "YeoJohnson", "nzv"))

# ---------- Caret ----------
# ------- GBM -------
fitControl <- trainControl( ## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated two times
  repeats = 2
)

gbmGrid <- expand.grid(
  interaction.depth = c(1, 5, 9),
  n.trees = (1:30) * 50,
  shrinkage = seq(0.1, 1, 0.1),
  n.minobsinnode = 20
)

gbmFit <- train(classe ~ yaw_arm,
  data = training[, -classe],
  method = "gbm",
  # trControl = fitControl,
  ## This last option is actually one
  ## for gbm() that passes through
  verbose = FALSE
  # tuneGrid = gbmGrid
)

gbmFit

# For a gradient boosting machine (GBM) model, there are three main tuning parameters:
#   - number of iterations, i.e. trees, (called n.trees in the gbm function)
#   - complexity of the tree, called interaction.depth
#   - learning rate: how quickly the algorithm adapts, called shrinkage
#   - the minimum number of training set samples in a node to commence splitting (n.minobsinnode)

trellis.par.set(caretTheme())
plot(gbmFit)

confusionMatrix(data = training$class, reference = test_set$obs)

# ------- xgboost -------
# xgbTree method = 'xgbTree'
# use xgboost to predict values

# performance
# confusionMatrix(data = test_set$pred, reference = test_set$obs)

# Set parameters(default)
params <- list(booster = "gbtree", objective = "multi:softprob", num_class = 4, eval_metric = "error")

# boosting
boosting <- gbm(classe ~ ., data = training, distribution = "multinomial", n.trees = 500, interaction.depth = 1, cv.folds = 5, shrinkage = 0.005)
boosting

a <- predict(boosting, training, type = "response")
# cross validation to tune hyperparameters


# ----------- mlr -----------
# -------------- preprocess data --------------
task <- makeClassifTask(id = "nike", data = df, target = "classe")

# nested resampling
rdesc.inner <- makeResampleDesc("CV", iters = 10)

measures <- list(mmce)

# random forest
lrn.rndforest <- makePreprocWrapperCaret("classif.randomForest", ppc.center = T, ppc.scale = T)

# feature selection
# wie?

ps.rndforest <- makeParamSet(
  makeIntegerParam("ntree", lower = 1, upper = 10),
  makeIntegerParam("mtry", lower = 5, upper = 10)
  # makeLogicalParam("ppc.center"),
  # makeLogicalParam("ppc.scale")
)

tune.ctrl.rndforest <- makeTuneControlRandom(maxit = 10)

tuned.lrn.rndforest <- makeTuneWrapper(lrn.rndforest,
  par.set = ps.rndforest,
  resampling = rdesc.inner,
  control = tune.ctrl.rndforest
)

# xgboost
lrn.xgboost <- makePreprocWrapperCaret("classif.xgboost", ppc.center = T, ppc.scale = T)

ps.xgboost <- makeParamSet(
  makeNumericParam("eta", lower = 0, upper = 0.5),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 0.9),
  makeNumericParam("gamma", lower = 0, upper = 2),
  makeIntegerParam("max_depth", lower = 4, upper = 10),
  makeIntegerParam("nrounds", lower = 100, upper = 2000)
)

tune.ctrl.xgboost <- makeTuneControlRandom(maxit = 50)

tuned.lrn.xgboost <- makeTuneWrapper(lrn.xgboost,
  par.set = ps.xgboost,
  resampling = rdesc.inner,
  control = tune.ctrl.xgboost
)
# create dummy variables for xgboost

# ranger (rf without tuning)
lrn.ranger <- makePreprocWrapperCaret("classif.ranger", ppc.center = T, ppc.scale = T)

# gbm
lrn.gbm <- makePreprocWrapperCaret("classif.gbm", ppc.center = T, ppc.scale = T)

ps.gbm <- makeParamSet(
  makeIntegerParam("n.trees", lower = 100, upper = 200),
  makeIntegerParam("interaction.depth", lower = 5, upper = 10),
  makeNumericParam("shrinkage", lower = 0, upper = 0.2)
)

tune.ctrl.gbm <- makeTuneControlRandom(maxit = 30)

tuned.lrn.gbm <- makeTuneWrapper(lrn.gbm,
  par.set = ps.gbm,
  resampling = rdesc.inner,
  control = tune.ctrl.gbm
)
# Define outer resampling, each learner is evaluated with that one:
rdesc.outer <- makeResampleDesc(method = "CV", iters = 3)
resample.instance.outer <- makeResampleInstance(desc = rdesc.outer, task = task)

# benchmark
bm <- benchmark(
  learners = list(
    # tuned.lrn.rndforest,
    # lrn.ranger,
    # tuned.lrn.xgboost,
    tuned.lrn.gbm
  ),
  tasks = task,
  resamplings = resample.instance.outer,
  measures = measures
)

bm

plotBMRBoxplots(bm)

# confusion matrix
for (i in (1:2)) {
  print(calculateConfusionMatrix(bm$results$nike[[i]][7]$pred))
}


## Train final model:
## ------------------------------------------------
model <- mlr::train(learner = tuned_learner_gbm, task = task)

save(list = model, file = "my_final_model.rds")

load("my_final_model.rds")


# ----------- Predict new data -----------
test <- read.csv("pml-testing.csv", na.strings = c("NA", "NaN", "", "#DIV/0!"), row.names = 1)

df %>%
  select(-classe) %>%
  colnames() -> vars

test <- test[vars]

pred <- predict(model, newdata = test)
