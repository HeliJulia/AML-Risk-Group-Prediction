

# NETWORKS WITHOUT RFE
# Heli Leskelä
# University of Oulu
# July 2022


# Packages ---------------------------------------------------------------------
library(keras)
library(tensorflow)
library(dplyr)
library(rsample)
library(caret)
library(smotefamily)
library(magrittr)


# Loading the training and test sets -------------------------------------------
train <- read.csv("training_set.csv")
test <- read.csv("test_set.csv")

train %<>%
  mutate(y = as.factor (y))
test %<>%
  mutate(y = as.factor (y))


# Separating predictors and target ---------------------------------------------
table(train$y) %>% prop.table()
table(test$y) %>% prop.table()

length(which(train$y == "Normal"))
length(which(train$y == "Favorable"))
length(which(train$y == "Poor"))

length(which(test$y == "Normal"))
length(which(test$y == "Favorable"))
length(which(test$y == "Poor"))


# Number of poor and favorable samples needed to have equal distributions ------
n_poor = length(which(train$y == "Normal")) - length(which(train$y == "Poor"))
n_fav = length(which(train$y == "Normal")) - length(which(train$y == "Favorable"))


# Creating 2 sets from training data -------------------------------------------
train %>% filter(as.integer(y)>1) %>% droplevels() -> norm_poor
train %>% filter(as.integer(y)<3) %>% droplevels() -> norm_fav

table(norm_poor$y) %>% prop.table()
table(norm_fav$y) %>% prop.table()


# Using SMOTE to balance training data  ----------------------------------------
poor_smote <- SMOTE(X = norm_poor[,-1], target = norm_poor$y,
                    K = 5, dup_size = 2)$syn_data 
fav_smote <- SMOTE(X = norm_fav[,-1], target = norm_fav$y,
                   K = 5, dup_size = 2)$syn_data 

names(poor_smote)[names(poor_smote) == 'class'] <- 'y'
names(fav_smote)[names(fav_smote) == 'class'] <- 'y'

poor_smote$y %<>% as.factor()
fav_smote$y %<>% as.factor()

i_poor <- sample.int(length(poor_smote$y), n_poor)
i_fav <- sample.int(length(fav_smote$y), n_fav)

poor_smote_sample <- poor_smote[i_poor,]
fav_smote_sample <- fav_smote[i_fav,]

level_list <- list("Favorable"="Favorable", "Normal"="Normal", "Poor"="Poor")
levels(poor_smote_sample$y) <- level_list
levels(fav_smote_sample$y) <- level_list
train_smote <- rbind(train, poor_smote_sample, fav_smote_sample)

table(train_smote$y) %>% prop.table()


# Using ADASYN to balance training data ----------------------------------------
poor_adas <- ADAS(X = norm_poor[,-1], target = norm_poor$y, K = 5)$syn_data
poor_adas2 <- ADAS(X = norm_poor[,-1], target = norm_poor$y, K = 5)$syn_data
fav_adas <- ADAS(X = norm_fav[,-1], target = norm_fav$y, K = 5)$syn_data
fav_adas2 <- ADAS(X = norm_fav[,-1], target = norm_fav$y, K = 5)$syn_data

fav_adas <- rbind(fav_adas, fav_adas2)
poor_adas <- rbind(poor_adas, poor_adas2)
fav_adas %<>% distinct()
poor_adas %<>% distinct()

names(poor_adas)[names(poor_adas) == 'class'] <- 'y'
names(fav_adas)[names(fav_adas) == 'class'] <- 'y'

poor_adas$y %<>% as.factor()
fav_adas$y %<>% as.factor()

i_poor <- sample.int(length(poor_adas$y), n_poor)
i_fav <- sample.int(length(fav_adas$y), n_fav)

poor_adas_sample <- poor_adas[i_poor,]
fav_adas_sample <- fav_adas[i_fav,]


levels(poor_adas_sample$y) <- level_list
levels(fav_adas_sample$y) <- level_list
train_adas <- rbind(train, poor_adas_sample, fav_adas_sample)

table(train_adas$y) %>% prop.table()


# Preparing the response variable ----------------------------------------------
y_train <- as.numeric(train$y)
y_train %>% 
  matrix(nrow = dim(train)[1], ncol = 1) %>% 
  to_categorical() -> y_train
y_train <- y_train[, -1]

y_train_smote <- as.numeric(train_smote$y)
y_train_smote %>% 
  matrix(nrow = dim(train_smote)[1], ncol = 1) %>% 
  to_categorical() -> y_train_smote
y_train_smote <- y_train_smote[, -1]

y_train_adas <- as.numeric(train_adas$y)
y_train_adas %>% 
  matrix(nrow = dim(train_adas)[1], ncol = 1) %>% 
  to_categorical() -> y_train_adas
y_train_adas <- y_train_adas[, -1]

y_test <- as.numeric(test$y)
y_test %>% 
  matrix(nrow = dim(test)[1], ncol = 1) %>% 
  to_categorical() -> y_test
y_test <- y_test[, -1]


# Preparing the predictors -----------------------------------------------------
x_train <- train[,-1] 
x_train <- as.matrix.data.frame(x_train)
dimnames(x_train) = NULL   

x_train_smote <- train_smote[,-1] 
x_train_smote <- as.matrix.data.frame(x_train_smote)
dimnames(x_train_smote) = NULL   

x_train_adas <- train_adas[,-1] 
x_train_adas <- as.matrix.data.frame(x_train_adas)
dimnames(x_train_adas) = NULL   

x_test <- test[,-1] 
x_test <- as.matrix.data.frame(x_test)
dimnames(x_test) = NULL   


# Parameters -------------------------------------------------------------------
seed <- 2022

hyperparameters <- list(
  layers = c(3:13),
  units = c(256, 512, 1024, 2048),
  drop = c(0, 0.1, 0.2, 0.3, 0.4, 0.5),
  lr = c(0.0001, 0.001, 0.01),
  batch_size = c(32, 64, 128, 171)
)
combinations <- expand.grid(hyperparameters)


# Choosing 100 random combinations ---------------------------------------------
id <- sample(1:dim(combinations)[1], size = 100, replace = F)
config_set <- combinations[id, ]

config_set <- read.csv("config_set.txt", sep="\t")

# Neural network modeling ------------------------------------------------------
build.network <- function(layers, units, drop, lr, x_train){
  model <- keras_model_sequential()
  
  model %>% 
    layer_dense(units = units, activation = "relu",
                input_shape = c(ncol(x_train))) %>% 
    layer_batch_normalization() %>%
    layer_dropout(rate = drop)
  for (i in 1:layers) {
    model %>%
      layer_dense(units = units, activation = "relu") %>% 
      layer_batch_normalization() %>%
      layer_dropout(rate = drop)
  }
  model %>% layer_dense(units = 3, activation = "softmax")

  model %>% compile(loss = 'categorical_crossentropy',
                    optimizer = optimizer_adam(learning_rate = lr),
                    metrics = c('accuracy'))
  return(model)
}


# Creating a custom keras callback ---------------------------------------------
AccHistory <- R6::R6Class("AccHistory",
                           inherit = KerasCallback,
                           public = list(
                             accs = NULL,
                             on_epoch_end = function(batch, logs = list()) {
                               self$accs <- c(self$accs, logs[["accuracy"]]) }))


# Training ---------------------------------------------------------------------
train.network <- function(model, x_train, y_train,
                          epochs, batch_size, patience=20){
  early_stop <- callback_early_stopping(monitor = "accuracy",
                                        patience = patience)
  history <- AccHistory$new()
  
  model %>% fit(
    x_train,
    y_train,
    batch_size = batch_size,
    epoch = epochs,
    verbose = 2,
    callbacks = list(history, early_stop)
  )

  pred <- predict(model, x_test)
  pred_fac <- as.factor(apply(pred, 1, which.max))
  levels(pred_fac) <- list("1"="1", "2"="2", "3"="3"); pred_fac
  true_fac <- as.factor(as.numeric(test$y)); true_fac
  
  cat("\nTrue: \t", true_fac)
  cat("\nPred: \t", pred_fac, "\n\n")
  cat("\nBest training accuracy: ", max(history$accs), "\n\n")
  
  return(pred_fac)
}


# Tuning the hyperparameters ---------------------------------------------------
epochs = 200

tune.network <- function(config_set, epochs, x_train, y_train){
  configuration <- data.frame(matrix(ncol = 5, nrow = dim(config_set)[1]))
  x <- c("layers", "units", "drop", "lr", "batch_size"#, "accuracy"
         #,"best_training_accuracy", "F1_1", "F1_2", "F1_3"
         )
  colnames(configuration) <- x
  
  predictions <- list(true_values = as.factor(as.numeric(test$y)))
  conf_mat <- list()
  
  for(i in 1:dim(config_set)[1]){
    cat("Tuning round: ", i, "/", dim(config_set)[1], "\n")
    model <- build.network(config_set$layers[i], config_set$units[i],
                           config_set$drop[i], config_set$lr[i], x_train)
    temp <- train.network(model, x_train, y_train,
                          epochs, config_set$batch_size[i])
    
    conf_mat[[i]] <- confusionMatrix(temp, as.factor(as.numeric(test$y)),
                                mode = "everything", positive="1")
    
    configuration[i,] <- c(config_set$layers[i],
                    config_set$units[i],
                    config_set$drop[i],
                    config_set$lr[i],
                    config_set$batch_size[i])
    
    predictions[[i+1]] <- temp
  }
  
  return(list(configuration = configuration,
              predictions = predictions,
              confusion_matrix = conf_mat))
}


# Tuning the networks ----------------------------------------------------------
models <- tune.network(config_set, epochs, x_train_smote, y_train_smote)


# Saving the results -----------------------------------------------------------
accuracies <- 0
f1_fav <- 0
f1_norm <- 0
f1_poor <- 0
predictions <- 0

for(i in 1:length(models$confusion_matrix)){
  accuracies[i] <- as.numeric(models$confusion_matrix[[i]]$overall[1]) 
  f1_fav[i] <- models$confusion_matrix[[i]]$byClass[1, 7] 
  f1_norm[i] <- models$confusion_matrix[[i]]$byClass[2, 7] 
  f1_poor[i] <- models$confusion_matrix[[i]]$byClass[3, 7] 
  predictions[i] <- paste(models$predictions[[i+1]], sep="", collapse =" " )
}

results <- models$configuration
results %>%
  mutate(method = rep("SMOTE", length(models$confusion_matrix))) %>%
  mutate(accuracy = round(accuracies,3)) %>%
  mutate(f1_fav = round(f1_fav, 2)) %>%
  mutate(f1_norm = round(f1_norm, 2)) %>%
  mutate(f1_poor = round(f1_poor, 2)) %>%
  mutate(prediction = predictions) %>%
  mutate(true_values = rep(paste(models$predictions$true_values,
                                 sep="", collapse =" " ),
         length(models$confusion_matrix))) -> results

results %>% arrange(desc(accuracy))
#write.table(results, file = "tuning_models.txt", sep = "\t", row.names = FALSE)
write.table(results, file = "tuning_models.txt", sep = "\t", append = TRUE,
            row.names = FALSE, col.names = FALSE)


# Loading the saved models, and saving the best ones ---------------------------
models_df <- read.csv("tuning_models.txt", sep="\t")

models_df %>%
  arrange(desc(accuracy)) %>%
  select(layers, units, drop, lr, batch_size, method,
         accuracy, f1_fav, f1_norm, f1_poor) -> best_a
write.table(best_a[1:10,], file = "best_a.txt", sep = " & ", row.names = FALSE)

models_df %>%
  arrange(desc(f1_fav)) %>%
  select(layers, units, drop, lr, batch_size, method,
         accuracy, f1_fav, f1_norm, f1_poor) -> best_fav
write.table(best_fav[1:10,], file = "best_fav.txt", sep = " & ", row.names = FALSE)

models_df %>%
  arrange(desc(f1_norm)) %>%
  select(layers, units, drop, lr, batch_size, method,
         accuracy, f1_fav, f1_norm, f1_poor) -> best_norm
write.table(best_norm[1:10,], file = "best_norm.txt", sep = " & ", row.names = FALSE)

models_df %>%
  arrange(desc(f1_poor)) %>%
  select(layers, units, drop, lr, batch_size, method,
         accuracy, f1_fav, f1_norm, f1_poor) -> best_poor
write.table(best_poor[1:10,], file = "best_poor.txt", sep = " & ", row.names = FALSE)


models_df %>% group_by(method) %>% arrange(desc(accuracy)) %>% slice_head(n = 1) -> best_acc
models_df %>% group_by(method) %>% arrange(desc(f1_fav)) %>% slice_head(n = 1) -> best_fav
models_df %>% group_by(method) %>% arrange(desc(f1_norm)) %>% slice_head(n = 1) -> best_norm
models_df %>% group_by(method) %>% arrange(desc(f1_poor)) %>% slice_head(n = 1) -> best_poor

best_acc %>% arrange(desc(accuracy)) %>% as.data.frame()
best_fav %>% arrange(desc(f1_fav))  %>% as.data.frame()
best_norm %>% arrange(desc(f1_norm))  %>% as.data.frame()
best_poor %>% arrange(desc(f1_poor))  %>% as.data.frame()

