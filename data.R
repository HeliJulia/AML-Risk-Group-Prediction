

# DATA PREPARATION
# Heli Leskelä
# University of Oulu
# July 2022


# Packages ---------------------------------------------------------------------
library(dplyr)
library(rsample)


# Data preparation -------------------------------------------------------------
df <- read.csv("laml.csv")

df %>% 
  mutate(censoring = as.factor(censoring)) %>%
  mutate(risk_category = as.factor(risk_category)) %>%
  mutate(gender = as.factor(gender)) %>%
  mutate(subtype = as.factor(subtype)) -> df_final

df_final %>%
  mutate(age_group = case_when(
    age >= 60 ~ "Old",
    age <= 40 ~ "Young",
    age > 40 & age < 60 ~ "Adult")) %>% 
  mutate(new_risk_category = risk_category) %>%
  mutate(age_group = as.factor(age_group)) %>%
  relocate(age_group, .after = age) %>% 
  relocate(new_risk_category, .after = subtype)-> df_final  

levels(df_final$risk_category) <- c("Favorable", "Normal", "Poor")
levels(df_final$new_risk_category) <- c("Favorable", "Normal", "Poor")
levels(df_final$gender) <- c("Female", "Male")
levels(df_final$subtype) <- c("M1", "M2", "M3", "M4", "M5", "M6", "M7")

df_final$new_risk_category[which(df_final$subtype == "M1")] <- "Poor"
df_final$new_risk_category[which(df_final$subtype == "M3")] <- "Favorable"
df_final %>% filter(!(is.na(new_risk_category))) -> df_final


# Saving the data as a csv -----------------------------------------------------
write.csv(df_final, "laml_final.csv", row.names = FALSE)


# Separating predictors and target ---------------------------------------------
y <- df_final$new_risk_category
X <- df_final[,9:dim(df_final)[2]]


# Dividing into training and test sets with stratified sampling ----------------
df <- cbind(y, X)
split <- initial_split(df, prop = 0.8, strata = "y")
train <- training(split)
test <- testing(split)

write.csv(train, "training_set.csv", row.names = FALSE)
write.csv(test, "test_set.csv", row.names = FALSE)


# Statistics -------------------------------------------------------------------
dim(df_final)               
str(df_final[,1:10])      
summary(df_final[,1:10])

round(table(df_final$gender)/dim(df_final)[1], 3)         
round(table(df_final$censoring)/dim(df_final)[1], 3)      
round(table(df_final$risk_category)/dim(df_final)[1], 3)  
round(table(df_final$subtype)/dim(df_final)[1], 3)      
round(table(df_final$age_group)/dim(df_final)[1], 3)     
summary(df_final$age) 
summary(df_final$survival_time)/365

round(table(df_final$new_risk_category)/dim(df_final)[1], 3)  


