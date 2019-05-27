# -------------- Packages --------------
library("dplyr")
library("caret")
library("mlr")
library("visdat")
library("data.table")

# -------------- Read data --------------
df <- fread("1_data/pml-training.csv", na.strings = c("NA", "NaN", "", "#DIV/0!"), drop = 1)

set.seed(31)

# -------------- Clean data set --------------
# remove obvious non-predictive variables
df %>% select(-c(raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, user_name)) -> df

#  extract id for blocking
# id <- df$user_name
# df %>% select(-user_name) -> df

# remove zero variance/near zero variance predictors
nzv <- nearZeroVar(df)
df <- df[, -..nzv]

df <- removeConstantFeatures(as.data.frame(df), perc = .02)

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

# remove highly correlated predictors NOT FINISHED --> cor[70] is -.99
nums <- select_if(df, is.numeric)
descrCor <- cor(nums)

highCorr <- sum(na.omit(abs(descrCor[upper.tri(descrCor)])) >= .98)

na.omit(descrCor[upper.tri(descrCor)])[which(na.omit(abs(descrCor[upper.tri(descrCor)])) >= .98, arr.ind = TRUE)]

which(na.omit(abs(descrCor)) >= .98 & na.omit(abs(descrCor)) < 1, arr.ind = TRUE)

rm <- findCorrelation(na.omit(descrCor), cutoff = .98, verbose = T, exact = T, names = T)

rm # 2 variables with high correlation --> consider in model evaluation

# find linear combinations
findLinearCombos(nums)

# visualize data
vis_dat(df, warn_large_data = F)

# are mixed data types in one variable?
vis_guess(df)

# save cleaned data
saveRDS(df, "1_data/cleaned_data.rds")

# read cleaned data if necessary
# df <- readRDS("1_data/cleaned_data.rds")

# -------------- Plot features --------------
nums <- unlist(lapply(df, is.numeric))
featurePlot(x = df[nums], y = df$classe, plot = "strip")

# -------------- Training --------------
# ---------- Task ----------
task <- makeClassifTask(id = "fitness.tracker", data = df, target = "classe")

# ---------- Inner resampling ----------
rdesc.inner <- makeResampleDesc("CV", iters = 1)

# ---------- Measures ----------
measures <- list(mmce)

# ---------- Learners ----------
# ------- Random forest -------
# lrn.rndforest <- makePreprocWrapperCaret("classif.randomForest", ppc.center = T, ppc.scale = T)
# 
# ps.rndforest <- makeParamSet(
#   makeIntegerParam("ntree", lower = 100, upper = 1000),
#   makeIntegerParam("mtry", lower = 5, upper = 20)
#   # makeLogicalParam("ppc.center"),
#   # makeLogicalParam("ppc.scale")
# )
# 
# tune.ctrl.rndforest <- makeTuneControlRandom(maxit = 30)
# 
# tuned.lrn.rndforest <- makeTuneWrapper(lrn.rndforest,
#   par.set = ps.rndforest,
#   resampling = rdesc.inner,
#   control = tune.ctrl.rndforest
# )

# ------- Xgboost -------
lrn.xgboost <- makePreprocWrapperCaret("classif.xgboost", ppc.center = T, ppc.scale = T)

ps.xgboost <- makeParamSet(
  makeNumericParam("eta", lower = 0, upper = 0.5),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 0.9),
  makeNumericParam("gamma", lower = 0, upper = 2),
  makeIntegerParam("max_depth", lower = 4, upper = 10),
  makeIntegerParam("nrounds", lower = 500, upper = 1500)
)

tune.ctrl.xgboost <- makeTuneControlRandom(maxit = 1)

tuned.lrn.xgboost <- makeTuneWrapper(lrn.xgboost,
  par.set = ps.xgboost,
  resampling = rdesc.inner,
  control = tune.ctrl.xgboost
)

# # ------- Ranger -------
# lrn.ranger <- makePreprocWrapperCaret("classif.ranger", ppc.center = T, ppc.scale = T)
# 
# ---------- Outer resampling ----------
rdesc.outer <- makeResampleDesc(method = "CV", iters = 1)
resample.instance.outer <- makeResampleInstance(desc = rdesc.outer, task = task)
# 
# # ---------- Benchmark ----------
# bm <- benchmark(
#   learners = list(
#     tuned.lrn.rndforest,
#     lrn.ranger,
#     tuned.lrn.xgboost
#   ),
#   tasks = task,
#   resamplings = resample.instance.outer,
#   measures = measures
# )
# 
# bm
# 
# plotBMRBoxplots(bm)
# 
# # confusion matrix
# for (i in (1:3)) {
#   print(calculateConfusionMatrix(bm$results$nike[[i]][7]$pred))
# }

# -------------- Train final model --------------
# choose best learner:
model <- mlr::train(learner = tuned.lrn.xgboost, task = task)

saveRDS(model, "model.rds")

# -------------- Predict new data --------------
# ---------- Load new data ----------
testing <- read.csv("1_data/pml-testing.csv", na.strings = c("NA", "NaN", "", "#DIV/0!"), row.names = 1)


# make sure that they have the same columns (except target)
df %>%
  select(-classe) %>%
  colnames() -> vars

testing <- testing[vars]

# ---------- Prediction ----------
pred <- predict(model, newdata = testing)

pred
