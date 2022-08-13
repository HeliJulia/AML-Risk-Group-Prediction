
# FEARURE SELECTION
# Heli Leskelä
# University of Oulu
# June 2022 

# Packages ---------------------------------------------------------------------
library(keras)
library(tensorflow)
library(dplyr)
library(rsample)
library(caret)
library(smotefamily)
library(magrittr)
library(ggplot2)
library(survminer)


# Loading the training and test sets -------------------------------------------
train <- read.csv("training_set.csv")
test <- read.csv("test_set.csv")

train %<>%
  mutate(y = as.factor (y))
test %<>%
  mutate(y = as.factor (y))


df <- rbind(train, test)
df %<>% mutate(y = as.factor (y))

X <- df[,-c(1)]
Y <- df[, 1]


# RFE First Set ----------------------------------------------------------------
subset1 <- rfe(x = X, y = Y,
               sizes = c(5, 10, 20, 40, 80, 160, 320,
                         641, 1283, 2566, 5132, 10265),
               rfeControl = rfeControl(functions = rfFuncs))


best1 <- predictors(subset1)

results1 <- data.frame(accuracy = max(subset1$results$Accuracy),
                      n_variables = length(best1),
                      variables = toString(best1))

write.table(results1, file = "rfe.txt", sep = "\t", row.names = FALSE)

data.frame(variables = subset1$results$Variables,
           accuracy = subset1$results$Accuracy) %>%
  ggplot( aes(x = variables, y = accuracy)) +
  geom_line(color="black", size=1.2)+
  geom_point(size=3) +
  theme(text = element_text(size = 16)) +
  xlab("\nNumber of Variables") + ylab("Accuracy (Bootstrap)\n") -> p1


data.frame(variables = subset1$results$Variables,
           accuracy = subset1$results$Accuracy) %>%
  ggplot(aes(x = 1:13, y = accuracy)) +
  geom_line(color="black", size=1.2)+
  geom_point(size=3) +
  theme(text = element_text(size = 14)) +
  xlab("\nNumber of Variables") + ylab("Accuracy (Bootstrap)\n") + 
  scale_x_discrete(
    limits=c(as.character(subset1$results$Variables))) -> p2


# The optimal number of features seems to be between 40 and 641-----------------
subset2 <- rfe(x = X, y = Y,
               sizes = c(41, 70, 90, 110, 150, 170, 190, 210, 410, 510, 560, 610),
               rfeControl = rfeControl(functions = rfFuncs))


best2 <- predictors(subset2)

results2 <- data.frame(accuracy = max(subset2$results$Accuracy),
                       n_variables = length(best2),
                       variables = toString(best2))

write.table(results2, file = "rfe.txt", sep = "\t", append = TRUE, row.names = FALSE, col.names = FALSE)

data.frame(variables = subset2$results$Variables,
           accuracy = subset2$results$Accuracy) %>%
  ggplot( aes(x = variables, y = accuracy)) +
  geom_line(color="black", size=1.2)+
  geom_point(size=3) +
  theme(text = element_text(size = 16)) +
  xlab("\nNumber of Variables") + ylab("Accuracy (Bootstrap)\n") -> p3


data.frame(variables = subset2$results$Variables,
           accuracy = subset2$results$Accuracy) %>%
  ggplot(aes(x = 1:13, y = accuracy)) +
  geom_line(color="black", size=1.2)+
  geom_point(size=3) +
  theme(text = element_text(size = 14)) +
  xlab("\nNumber of Variables") + ylab("Accuracy (Bootstrap)\n") + 
  scale_x_discrete(
    limits=c(as.character(subset2$results$Variables))) -> p4


ggarrange(p2, p4, ncol=1)


# Between 170 and 210-----------------------------------------------------------
subset3 <- rfe(x = X, y = Y,
               sizes = seq(175, 205, 5),
               rfeControl = rfeControl(functions = rfFuncs))


best3 <- predictors(subset3)

results3 <- data.frame(accuracy = max(subset3$results$Accuracy),
                       n_variables = length(best3),
                       variables = toString(best3))

write.table(results3, file = "rfe.txt", sep = "\t", append = TRUE, row.names = FALSE, col.names = FALSE)

data.frame(variables = subset3$results$Variables,
           accuracy = subset3$results$Accuracy) %>%
  ggplot( aes(x = variables, y = accuracy)) +
  geom_line(color="black", size=1.2)+
  geom_point(size=3) +
  theme(text = element_text(size = 16)) +
  xlab("\nNumber of Variables") + ylab("Accuracy (Bootstrap)\n") -> p5


data.frame(variables = subset3$results$Variables,
           accuracy = subset3$results$Accuracy) %>%
  ggplot(aes(x = 1:8, y = accuracy)) +
  geom_line(color="black", size=1.2)+
  geom_point(size=3) +
  theme(text = element_text(size = 14)) +
  xlab("\nNumber of Variables") + ylab("Accuracy (Bootstrap)\n") + 
  scale_x_discrete(
    limits=c(as.character(subset3$results$Variables))) -> p6


ggarrange(p2, p4, p6, ncol=1)


# Last try ---------------------------------------------------------------------
subset4 <- rfe(x = X, y = Y,
               sizes = seq(600, 1200, 50),
               rfeControl = rfeControl(functions = rfFuncs))


best4 <- predictors(subset4)

results4 <- data.frame(accuracy = max(subset4$results$Accuracy),
                       n_variables = length(best4),
                       variables = toString(best4))

write.table(results4, file = "rfe.txt", sep = "\t", append = TRUE, row.names = FALSE, col.names = FALSE)

data.frame(variables = subset4$results$Variables,
           accuracy = subset4$results$Accuracy) %>%
  ggplot( aes(x = variables, y = accuracy)) +
  geom_line(color="black", size=1.2)+
  geom_point(size=3) +
  theme(text = element_text(size = 16)) +
  xlab("\nNumber of Variables") + ylab("Accuracy (Bootstrap)\n") -> p7


data.frame(variables = subset4$results$Variables,
           accuracy = subset4$results$Accuracy) %>%
  ggplot(aes(x = 1:14, y = accuracy)) +
  geom_line(color="black", size=1.2)+
  geom_point(size=3) +
  theme(text = element_text(size = 14)) +
  xlab("\nNumber of Variables") + ylab("Accuracy (Bootstrap)\n") + 
  scale_x_discrete(
    limits=c(as.character(subset4$results$Variables))) -> p8


ggarrange(p2, p4, p6, p8, ncol=2, nrow=2)
