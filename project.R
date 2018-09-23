# -------------- Packages --------------
library("dplyr")
library("ggplot2")
library("caret")
library("rlist")
library("xgboost")
# library("mlr")
library("visdat")
library(ISLR)
library(tree)
library(xgboost)
library(gbm)

# -------------- Read Data --------------
df <- read.csv("pml-training.csv", na.strings = c("NA", "NaN", "", "#DIV/0!"), row.names = 1)

set.seed(31)

# -------------- clean data set --------------
# remove easily identifiable useless data
df$user_name <- NULL

# crop predictors with mostly NAs
df %>%
  select(everything()) %>%
  summarise_all(funs(sum(is.na(.)) / length(.))) -> p

# check these variables
vis_miss(df[which(p > 0.9)],
  sort_miss = TRUE, warn_large_data = F
)

# remove them if sensible
df[which(p > 0.9)] <- NULL

# visualize data
vis_dat(df, warn_large_data = F)

# are mixed data types in one variable?
vis_guess(df)

# check for factors
l <- list()
for (i in 1:ncol(df)) {
  if (length(unique(na.omit(df[, i]))) <= 2) {
    l <- list.append(l, colnames(df)[i])
    print(l)
  }
}

# remove zero variance/near zero variance predictors
nzv <- nearZeroVar(df)
nzv
df <- df[-nzv]

# remove highly correlated predictors NOT FINISHED --> cor[70] is -.99
nums <- select_if(df, is.numeric)
descrCor <- cor(nums)

highCorr <- sum(na.omit(abs(descrCor[upper.tri(descrCor)])) >= .98)

na.omit(descrCor[upper.tri(descrCor)])[which(na.omit(abs(descrCor[upper.tri(descrCor)])) >= .98, arr.ind = TRUE)]

which(na.omit(abs(descrCor)) >= .98 & na.omit(abs(descrCor)) < 1, arr.ind = TRUE)

rm <- findCorrelation(na.omit(descrCor), cutoff = .98, verbose = T, exact = T, names = T)

df %>% select(-rm) -> df

# find linear combinations
findLinearCombos(nums)

# save cleaned data
write.csv(df, "pml-training_cleaned.csv", row.names = F, fileEncoding = "UTF-8")

# -------------- read cleaned data --------------
df <- read.csv("pml-training_cleaned.csv", na.strings = c("NA", "NaN", "", "#DIV/0!"), header = T, encoding = "UTF-8")

# -------------- plot features --------------
for (i in seq_along(l)) {
  ggplot(df, aes(x = df[, l[[i]]], y = classe)) +
    geom_point()
  ggsave(filename = paste("plots/myplot", l[i], ".png", sep = ""))
}

# featurePlot (caret)
fp <- df[which(sapply(df, class) %in% c("integer", "numeric"))]
featurePlot(x = fp[6:10], y = df$classe, plot = "pairs")


# -------------- Modeling --------------
df <- read.csv("pml-training_cleaned.csv", na.strings = c("NA", "NaN", "", "#DIV/0!"), header = T, encoding = "UTF-8")

# ---------- Train/Test-set ----------
# setup train and test set
train <- sample(nrow(df), 0.8 * nrow(df), replace = F)
train <- createDataPartition(df$classe, p = 0.8, list = F)
training <- df[train, ]
test <- df[-train, ]

# normalize numeric predictors
preProcValues <- preProcess(training, method = c("center", "scale"))
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
  data = training[,-classe],
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

a <- predict(boosting, training,type="response")
# cross validation to tune hyperparameters


# ----------- mlr -----------
# Task -> Learner -> train
## General example:
# task = makeClassifTask(data = iris, target = "Species")
#
# ## Generate the learner
# lrn = makeLearner("classif.lda")
#
# ## Train the learner
# mod = train(lrn, task)

# Genearate Task
makeClassifTask(data = df, target = "classe")

# cv with resample: makeResampleDesc/resample
