#Importing necessary packages
library(dplyr)
library(plm)
library(ggplot2)
library(glmnet)
library(car)

#Setting the working dir
setwd("/Users/fabioza/Desktop/Master thesis/data")

#Reading the data
data <- read.csv("df_encoded_linear30d.csv")

#Dropping unnecessary column
data <- data[,!(names(data) %in% c("X", "Sushiswap"))]

#Creating an id column that stores the pool number
data <- data %>% mutate(id = apply(select(., starts_with('pool_')), 1, function(x) which.max(x)))

#Variables to be tested for unit root
vars_to_test <- c("volume_usd", "imbalance", "X30D_volatility", "X30D_impermanent_loss", "total_USD_value", "return", "return_in_1month", "avg_gas")

#Converting data into panel data format
pdata <- pdata.frame(data, index = c("id", "start_date"))

#List to store results
results <- list()

#Loop over all variables
for (var in vars_to_test) {
  #Run IPS test to test variables for unit root
  test_result <- purtest(pdata[[var]], test = "ips", pmax = 2, exo = "intercept")
  results[[var]] <- test_result
}

#Calculating the correlation matrix
cor_matrix <- cor(data[, vars_to_test])

#Scaling volume and total usd value otherwise regression doesn't work
pdata$volume_usd <- pdata$volume_usd / 1000
pdata$total_USD_value <- pdata$total_USD_value / 1000

#Rounding data to two digts
pdata <- pdata %>% 
  mutate_if(is.numeric, round, digits = 2)

#Formula of regression model
formula <- return_in_1month ~ volume_usd + imbalance + avg_gas + X30D_impermanent_loss + X30D_volatility + total_USD_value + return + id + pool_category_0 + pool_category_1 + Uniswap 

#Sorting pdata by date and then by pool to allow correct training of the model (without leakage)
pdata <- pdata %>% arrange(start_date, id)

results_list <- list()

#Removing the last six observations per pool and adding to test set (six months of test data)
last_two_obs <- pdata %>% 
  group_by(id) %>% 
  slice_tail(n = 6) 

#Remaining observations in test set
remaining_obs <- anti_join(pdata, last_two_obs, by = c("id", "start_date"))

#Creating the training set folds, starts with 175 observations and is increased by 70 observations
train_sizes <- seq(175, nrow(remaining_obs), by = 70)

#Dataframe to store predicted and actual values
actual_vs_predicted <- data.frame(
  id = integer(),
  actual = numeric(),
  predicted = numeric(),
  stringsAsFactors = FALSE
)

for (i in train_sizes) {
  #Creating training and validation sets
  train_set <- remaining_obs[1:i, ]
  validation_set <- remaining_obs[(i + 1):nrow(remaining_obs), ]
  
  #Preparing training features and labels
  data_matrix_train <- model.matrix(formula, train_set)
  y_train <- train_set$return_in_1month
  x_train <- data_matrix_train[, -1]
  
  #Fitting the penalized regression with different values for alpha
  #For ridge alpha = 0, for lasso alpha = 1 and for elastic net = 0.5
  lasso_model_train <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 5)
  opt_lambda_train <- lasso_model_train$lambda.min
  lasso_fit_train <- glmnet(x_train, y_train, alpha = 0, lambda = opt_lambda_train)
  
  #Preparing training features and labels
  data_matrix_test <- model.matrix(formula, validation_set)
  x_test <- data_matrix_test[, -1]
  y_test <- validation_set$return_in_1month
  
  #Predicting return
  y_pred <- predict(lasso_fit_train, newx = x_test)
  
  #Storing actual and predicted values
  actual_vs_predicted <- rbind(actual_vs_predicted, data.frame(
    id = validation_set$id,
    actual = y_test,
    predicted = as.vector(y_pred),
    stringsAsFactors = FALSE
  ))
  
  #Computing evaluation metrics
  RMSE <- sqrt(mean((y_test - y_pred)^2))
  MSE <- mean((y_test - y_pred)^2)
  MAE <- mean(abs(y_test - y_pred))
  hits <- ifelse(sign(y_test) == sign(y_pred), 1, 0)
  hit_rate <- mean(hits)
  
  results_list[[length(results_list) + 1]] <- list(RMSE = RMSE, MSE = MSE, MAE = MAE, hit_rate = hit_rate)
}

#Computing average metrics over all folds
avg_RMSE <- mean(sapply(results_list, function(x) x$RMSE))
avg_MSE <- mean(sapply(results_list, function(x) x$MSE))
avg_MAE <- mean(sapply(results_list, function(x) x$MAE))
avg_hit_rate <- mean(sapply(results_list, function(x) x$hit_rate))

cat("Average RMSE:", avg_RMSE, "\n")
cat("Average MSE:", avg_MSE, "\n")
cat("Average MAE:", avg_MAE, "\n")
cat("Average Hit Rate:", avg_hit_rate, "\n")

ggplot(actual_vs_predicted, aes(x = id)) +
  geom_point(aes(y = actual, color = "Actual")) +
  geom_point(aes(y = predicted, color = "Predicted")) +
  labs(x = "Index", y = "Return in 1 month", color = "Legend") +
  theme_minimal()

