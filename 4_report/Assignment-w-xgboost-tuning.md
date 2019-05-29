---
title: "Course Assignment"
author: "Moritz Körber"
test: "LC_COLLATE=English_United States.1252;LC_CTYPE=English_United States.1252;LC_MONETARY=English_United States.1252;LC_NUMERIC=C;LC_TIME=English_United States.1252"
date: "May 28, 2019"
output: 
  html_document:
    keep_md: true
---



First step: Load required packages:


```r
library("dplyr")
library("caret")
library("mlr")
library("visdat")
library("data.table")
```

Load data and set seed:


```r
df <- fread("1_data/pml-training.csv", na.strings = c("NA", "NaN", "", "#DIV/0!"), drop = 1)

set.seed(31)
```

# 1. Inspect target variable
First, I am looking at the target variable "classe":


```r
glimpse(df$classe)
```

```
##  chr [1:19622] "A" "A" "A" "A" "A" "A" "A" "A" "A" "A" "A" "A" "A" "A" ...
```

Since it is a categorical variable, it is a classification problem. Hence, an algorithm like logistic regression, random forests, or a similar algorithm is suitable. Let's check if the classes are balanced:


```r
ggplot(df, aes(x = classe)) +
  geom_bar(stat = "count") +
  theme_classic()
```

![](Assignment_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

```r
count(df, classe)
```

```
## # A tibble: 5 x 2
##   classe     n
##   <chr>  <int>
## 1 A       5580
## 2 B       3797
## 3 C       3422
## 4 D       3216
## 5 E       3607
```

# 2. Clean the data set
Remove obvious non-predictive variables:

```r
df %>% select(-c(raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, user_name)) -> df
```

Remove zero variance/near zero variance predictors:

```r
nzv <- nearZeroVar(df)
df <- df[ , -..nzv]

df <- removeConstantFeatures(as.data.frame(df), perc = .02)
```

```
## Removing 12 columns: kurtosis_picth_belt,skewness_roll_belt.1,kurtosis_roll_arm,kurtosis_picth_arm,skewness_roll_arm,skewness_pitch_arm,kurtosis_roll_forearm,kurtosis_picth_forearm,skewness_roll_forearm,skewness_pitch_forearm,max_yaw_forearm,min_yaw_forearm
```

Remove predictors with mostly NAs:

```r
df %>%
  select(everything()) %>%
  summarise_all(funs(sum(is.na(.)) / length(.))) -> p
```

```
## Warning: funs() is soft deprecated as of dplyr 0.8.0
## please use list() instead
## 
##   # Before:
##   funs(name = f(.))
## 
##   # After: 
##   list(name = ~ f(.))
## This warning is displayed once per session.
```

```r
# check these variables
vis_miss(df[which(p > 0.975)],
  sort_miss = TRUE, warn_large_data = F
)
```

![](Assignment_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

```r
# remove them if sensible
df[which(p > 0.975)] <- NULL
```

Remove highly correlated predictors:

```r
nums <- select_if(df, is.numeric)
descrCor <- cor(nums)

highCorr <- sum(na.omit(abs(descrCor[upper.tri(descrCor)])) >= .98)

na.omit(descrCor[upper.tri(descrCor)])[which(na.omit(abs(descrCor[upper.tri(descrCor)])) >= .98, arr.ind = TRUE)]
```

```
## [1]  0.9809241 -0.9920085
```

```r
which(na.omit(abs(descrCor)) >= .98 & na.omit(abs(descrCor)) < 1, arr.ind = TRUE)
```

```
##                  row col
## total_accel_belt   5   2
## accel_belt_z      11   2
## roll_belt          2   5
## roll_belt          2  11
```

```r
findCorrelation(na.omit(descrCor), cutoff = .98, verbose = T, exact = T, names = T)
```

```
## Compare row 11  and column  2 with corr  0.992 
##   Means:  0.266 vs 0.165 so flagging column 11 
## Compare row 2  and column  5 with corr  0.981 
##   Means:  0.247 vs 0.161 so flagging column 2 
## All correlations <= 0.98
```

```
## [1] "accel_belt_z" "roll_belt"
```

There are two variables with high correlation. I will leave them in the dataset for this time.

Find linear combinations:

```r
findLinearCombos(nums)
```

```
## $linearCombos
## list()
## 
## $remove
## NULL
```

Visualize the data:

```r
vis_dat(df, warn_large_data = F)
```

![](Assignment_files/figure-html/unnamed-chunk-10-1.png)<!-- -->

Are there mixed data types in one variable?

```r
vis_guess(df)
```

![](Assignment_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

Plot the features:

```r
nums <- unlist(lapply(df, is.numeric))
featurePlot(x = df[nums], y = df$classe, plot = "strip")
```

```
## NULL
```

Save the cleaned data:

```r
saveRDS(df, "1_data/cleaned_data.rds")
```

# 3. Training
The training is run on an Amazon AWS EC2 t2.2xlarge instance. More details on the environment:

```
## R version 3.6.0 (2019-04-26)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## Running under: Windows Server x64 (build 17763)
## 
## Matrix products: default
## 
## locale:
## [1] LC_COLLATE=English_United States.1252 
## [2] LC_CTYPE=English_United States.1252   
## [3] LC_MONETARY=English_United States.1252
## [4] LC_NUMERIC=C                          
## [5] LC_TIME=English_United States.1252    
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
## [1] visdat_0.5.3      mlr_2.14.0        ParamHelpers_1.12 caret_6.0-84     
## [5] ggplot2_3.1.1     lattice_0.20-38   dplyr_0.8.1      
## 
## loaded via a namespace (and not attached):
##  [1] gbm_2.1.5           tidyselect_0.2.5    purrr_0.3.2        
##  [4] reshape2_1.4.3      splines_3.6.0       colorspace_1.4-1   
##  [7] generics_0.0.2      stats4_3.6.0        survival_2.44-1.1  
## [10] XML_3.98-1.19       prodlim_2018.04.18  rlang_0.3.4        
## [13] ModelMetrics_1.2.2  pillar_1.4.0        glue_1.3.1         
## [16] withr_2.1.2         xgboost_0.82.1      foreach_1.4.4      
## [19] plyr_1.8.4          lava_1.6.5          stringr_1.4.0      
## [22] timeDate_3043.102   munsell_0.5.0       gtable_0.3.0       
## [25] recipes_0.1.5       codetools_0.2-16    parallelMap_1.4    
## [28] parallel_3.6.0      class_7.3-15        Rcpp_1.0.1         
## [31] scales_1.0.0        backports_1.1.4     checkmate_1.9.3    
## [34] ipred_0.9-9         gridExtra_2.3       fastmatch_1.1-0    
## [37] ranger_0.11.2       stringi_1.4.3       BBmisc_1.11        
## [40] grid_3.6.0          tools_3.6.0         magrittr_1.5       
## [43] lazyeval_0.2.2      tibble_2.1.1        randomForest_4.6-14
## [46] crayon_1.3.4        pkgconfig_2.0.2     MASS_7.3-51.4      
## [49] Matrix_1.2-17       data.table_1.12.2   lubridate_1.7.4    
## [52] gower_0.2.1         assertthat_0.2.1    iterators_1.0.10   
## [55] R6_2.4.0            rpart_4.1-15        nnet_7.3-12        
## [58] nlme_3.1-139        compiler_3.6.0
```

## Task

```r
task <- makeClassifTask(id = "fitness.tracker", data = df, target = "classe")
```

## Resampling
I chose to use a nested cross validation strategy with a 5-fold inner cross validation and a 3-fold outer cross validation. The evaluation of the tuned learners is performed by 5-fold cross validation. 

```r
rdesc.inner <- makeResampleDesc("CV", iters = 5)
```

The best parameter combination is then evaluated against the remaining fold of the 3-fold outer cross validation.

```r
rdesc.outer <- makeResampleDesc(method = "CV", iters = 3)
resample.instance.outer <- makeResampleInstance(desc = rdesc.outer, task = task)
```

## Measures
The mean misclassification error is one of the most important metrics in classification problems.

```r
measures <- list(mmce)
```

## Learners
I compare the performance of three different learners. Each learner's hyperparameters are tuned in this evaluation process. For preprocessing, I center and scale the features.

### Random forest

```r
lrn.rndforest <- makePreprocWrapperCaret("classif.randomForest", ppc.center = T, ppc.scale = T)

ps.rndforest <- makeParamSet(
  makeIntegerParam("ntree", lower = 100, upper = 1000),
  makeIntegerParam("mtry", lower = 5, upper = 20)
)

tune.ctrl.rndforest <- makeTuneControlRandom(maxit = 30)

tuned.lrn.rndforest <- makeTuneWrapper(lrn.rndforest,
  par.set = ps.rndforest,
  resampling = rdesc.inner,
  control = tune.ctrl.rndforest
)
```

### XGBoost

```r
lrn.xgboost <- makePreprocWrapperCaret("classif.xgboost", ppc.center = T, ppc.scale = T)
```

```
## Warning in makeParam(id = id, type = "numeric", learner.param = TRUE, lower = lower, : NA used as a default value for learner parameter missing.
## ParamHelpers uses NA as a special value for dependent parameters.
```

```r
ps.xgboost <- makeParamSet(
  makeNumericParam("eta", lower = 0, upper = 0.5),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 0.9),
  makeNumericParam("gamma", lower = 0, upper = 2),
  makeIntegerParam("max_depth", lower = 4, upper = 10),
  makeIntegerParam("nrounds", lower = 500, upper = 1500)
)

tune.ctrl.xgboost <- makeTuneControlRandom(maxit = 30)

tuned.lrn.xgboost <- makeTuneWrapper(lrn.xgboost,
  par.set = ps.xgboost,
  resampling = rdesc.inner,
  control = tune.ctrl.xgboost
)
```

### Ranger

```r
lrn.ranger <- makePreprocWrapperCaret("classif.ranger", ppc.center = T, ppc.scale = T)
```

## Benchmark
Compare the leaners' performance:

```r
bm <- benchmark(
  learners = list(
    tuned.lrn.rndforest,
    tuned.lrn.xgboost,
    lrn.ranger
  ),
  tasks = task,
  resamplings = resample.instance.outer,
  measures = measures
)
```


```
##   task.id                         learner.id mmce.test.mean
## 1    nike classif.randomForest.preproc.tuned   0.0019366278
## 2    nike      classif.xgboost.preproc.tuned   0.0007644481
## 3    nike             classif.ranger.preproc   0.0026501542
```


```r
plotBMRBoxplots(bm)
```

![](Assignment_files/figure-html/unnamed-chunk-25-1.png)<!-- -->

XGBoost seems to be the best learner for this problem. Thus, I choose it to train my final model on the whole data set.

# 4. Train final model

```r
model <- mlr::train(learner = tuned.lrn.xgboost, task = task)
```

```
## [Tune] Started tuning learner classif.xgboost.preproc for parameter set:
```

```
##                     Type len Def         Constr Req Tunable Trafo
## eta              numeric   -   -       0 to 0.5   -    TRUE     -
## colsample_bytree numeric   -   -     0.5 to 0.9   -    TRUE     -
## gamma            numeric   -   -         0 to 2   -    TRUE     -
## max_depth        integer   -   -        4 to 10   -    TRUE     -
## nrounds          integer   -   - 500 to 1.5e+03   -    TRUE     -
```

```
## With control class: TuneControlRandom
```

```
## Imputation value: 1
```

```
## [Tune-x] 1: eta=0.372; colsample_bytree=0.86; gamma=0.0324; max_depth=6; nrounds=704
```

```
## [Tune-y] 1: mmce.test.mean=0.0003567; time: 10.5 min
```

```
## [Tune-x] 2: eta=0.409; colsample_bytree=0.619; gamma=1.57; max_depth=10; nrounds=802
```

```
## [Tune-y] 2: mmce.test.mean=0.0008664; time: 19.1 min
```

```
## [Tune-x] 3: eta=0.421; colsample_bytree=0.75; gamma=0.512; max_depth=6; nrounds=735
```

```
## [Tune-y] 3: mmce.test.mean=0.0004587; time: 13.0 min
```

```
## [Tune-x] 4: eta=0.256; colsample_bytree=0.848; gamma=1.64; max_depth=7; nrounds=584
```

```
## [Tune-y] 4: mmce.test.mean=0.0008154; time: 13.5 min
```

```
## [Tune-x] 5: eta=0.0331; colsample_bytree=0.749; gamma=0.125; max_depth=5; nrounds=1178
```

```
## [Tune-y] 5: mmce.test.mean=0.0003058; time: 13.5 min
```

```
## [Tune-x] 6: eta=0.488; colsample_bytree=0.751; gamma=1.08; max_depth=6; nrounds=508
```

```
## [Tune-y] 6: mmce.test.mean=0.0008663; time: 6.7 min
```

```
## [Tune-x] 7: eta=0.0114; colsample_bytree=0.676; gamma=0.168; max_depth=5; nrounds=516
```

```
## [Tune-y] 7: mmce.test.mean=0.0077973; time: 5.5 min
```

```
## [Tune-x] 8: eta=0.205; colsample_bytree=0.824; gamma=0.564; max_depth=9; nrounds=1161
```

```
## [Tune-y] 8: mmce.test.mean=0.0004587; time: 22.2 min
```

```
## [Tune-x] 9: eta=0.434; colsample_bytree=0.657; gamma=1.09; max_depth=5; nrounds=1343
```

```
## [Tune-y] 9: mmce.test.mean=0.0005096; time: 13.8 min
```

```
## [Tune-x] 10: eta=0.0253; colsample_bytree=0.849; gamma=0.582; max_depth=4; nrounds=1479
```

```
## [Tune-y] 10: mmce.test.mean=0.0005096; time: 15.2 min
```

```
## [Tune-x] 11: eta=0.365; colsample_bytree=0.618; gamma=0.545; max_depth=7; nrounds=658
```

```
## [Tune-y] 11: mmce.test.mean=0.0003567; time: 12.5 min
```

```
## [Tune-x] 12: eta=0.335; colsample_bytree=0.624; gamma=1.19; max_depth=7; nrounds=800
```

```
## [Tune-y] 12: mmce.test.mean=0.0004077; time: 14.1 min
```

```
## [Tune-x] 13: eta=0.0337; colsample_bytree=0.88; gamma=0.496; max_depth=7; nrounds=819
```

```
## [Tune-y] 13: mmce.test.mean=0.0005096; time: 16.2 min
```

```
## [Tune-x] 14: eta=0.125; colsample_bytree=0.565; gamma=0.435; max_depth=10; nrounds=1424
```

```
## [Tune-y] 14: mmce.test.mean=0.0004077; time: 23.0 min
```

```
## [Tune-x] 15: eta=0.24; colsample_bytree=0.81; gamma=1.58; max_depth=9; nrounds=1088
```

```
## [Tune-y] 15: mmce.test.mean=0.0005606; time: 25.9 min
```

```
## [Tune-x] 16: eta=0.493; colsample_bytree=0.885; gamma=1.66; max_depth=9; nrounds=1196
```

```
## [Tune-y] 16: mmce.test.mean=0.0008154; time: 29.5 min
```

```
## [Tune-x] 17: eta=0.301; colsample_bytree=0.692; gamma=1.7; max_depth=9; nrounds=1192
```

```
## [Tune-y] 17: mmce.test.mean=0.0005606; time: 23.9 min
```

```
## [Tune-x] 18: eta=0.483; colsample_bytree=0.671; gamma=0.355; max_depth=8; nrounds=709
```

```
## [Tune-y] 18: mmce.test.mean=0.0004077; time: 14.3 min
```

```
## [Tune-x] 19: eta=0.0819; colsample_bytree=0.833; gamma=0.409; max_depth=5; nrounds=1495
```

```
## [Tune-y] 19: mmce.test.mean=0.0005606; time: 42.2 min
```

```
## [Tune-x] 20: eta=0.345; colsample_bytree=0.695; gamma=0.381; max_depth=4; nrounds=1170
```

```
## [Tune-y] 20: mmce.test.mean=0.0004077; time: 19.8 min
```

```
## [Tune-x] 21: eta=0.184; colsample_bytree=0.595; gamma=0.865; max_depth=8; nrounds=1003
```

```
## [Tune-y] 21: mmce.test.mean=0.0004077; time: 28.2 min
```

```
## [Tune-x] 22: eta=0.331; colsample_bytree=0.799; gamma=1.85; max_depth=10; nrounds=764
```

```
## [Tune-y] 22: mmce.test.mean=0.0007135; time: 42.1 min
```

```
## [Tune-x] 23: eta=0.0671; colsample_bytree=0.691; gamma=1.29; max_depth=6; nrounds=1205
```

```
## [Tune-y] 23: mmce.test.mean=0.0003567; time: 37.0 min
```

```
## [Tune-x] 24: eta=0.397; colsample_bytree=0.811; gamma=0.573; max_depth=6; nrounds=938
```

```
## [Tune-y] 24: mmce.test.mean=0.0006115; time: 23.2 min
```

```
## [Tune-x] 25: eta=0.344; colsample_bytree=0.604; gamma=0.755; max_depth=6; nrounds=704
```

```
## [Tune-y] 25: mmce.test.mean=0.0004587; time: 15.5 min
```

```
## [Tune-x] 26: eta=0.214; colsample_bytree=0.811; gamma=0.813; max_depth=4; nrounds=864
```

```
## [Tune-y] 26: mmce.test.mean=0.0005096; time: 15.6 min
```

```
## [Tune-x] 27: eta=0.399; colsample_bytree=0.616; gamma=0.281; max_depth=6; nrounds=1419
```

```
## [Tune-y] 27: mmce.test.mean=0.0003058; time: 40.7 min
```

```
## [Tune-x] 28: eta=0.109; colsample_bytree=0.872; gamma=1.65; max_depth=6; nrounds=1392
```

```
## [Tune-y] 28: mmce.test.mean=0.0008663; time: 33.0 min
```

```
## [Tune-x] 29: eta=0.0933; colsample_bytree=0.574; gamma=1.11; max_depth=6; nrounds=723
```

```
## [Tune-y] 29: mmce.test.mean=0.0003567; time: 14.0 min
```

```
## [Tune-x] 30: eta=0.029; colsample_bytree=0.843; gamma=1.08; max_depth=6; nrounds=1264
```

```
## [Tune-y] 30: mmce.test.mean=0.0007135; time: 29.3 min
```

```
## [Tune] Result: eta=0.399; colsample_bytree=0.616; gamma=0.281; max_depth=6; nrounds=1419 : mmce.test.mean=0.0003058
```

```r
saveRDS(model, "model.rds")
```

# 5. Predict the test data
## Load and prepare test data

```r
testing <- read.csv("1_data/pml-testing.csv", na.strings = c("NA", "NaN", "", "#DIV/0!"), row.names = 1)

# make sure that they have the same columns (except the target)
df %>%
  select(-classe) %>%
  colnames() -> vars

testing <- testing[vars]
```

## Prediction
Lastly, I predict the test data.

```r
pred <- predict(model, newdata = testing)

pred
```

```
## Prediction: 20 observations
## predict.type: response
## threshold: 
## time: 0.07
##   response
## 1        B
## 2        A
## 3        B
## 4        A
## 5        A
## 6        E
## ... (#rows: 20, #cols: 1)
```
