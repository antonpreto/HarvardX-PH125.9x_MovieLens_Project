# Course: Data Science: HarvardX PH125.9x Capstone
# Name: Anton Preto
# Project: MovieLens
# Date: April 2023

# Project description: 
# For this project, you will be creating a movie recommendation system using the MovieLens dataset.
# You will be creating your own recommendation system using all the tools we have shown you throughout the courses in this series. 
# We will use the 10M version of the MovieLens dataset to make the computation a little easier.
# You will train a machine learning algorithm using the inputs in one subset to predict movie ratings in the final holdout test set.

# MovieLens Project Instructions:
# The submission for the MovieLens project will be three files: a report in the form of an Rmd file, a report in the form of a PDF document knit from your Rmd file, and an R script that generates your predicted movie ratings and calculates RMSE. 
# The R script should contain all of the code and comments for your project. 
# Your grade for the project will be based on two factors:
#      Your report and script (75%)
#      The RMSE returned by testing your algorithm on the final_holdout_test set (25%)

# Data: using edX code
##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(ggplot2)
library(knitr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)

##########################################################
# Splitting edx dataset into training and test set
##########################################################

set.seed(1, sample.kind="Rounding") # Using R v4+
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE) # p=0.1 define what proportion of the data is represented by the index
edx_test_index

edx_test <- edx[edx_test_index, ]
edx_test

edx_train <- edx[-edx_test_index,]
edx_train

edx_temp <- edx[edx_test_index,]

# Matching userId and movieId in both train and test sets
edx_test <- edx_temp %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")
edx_test

# Adding back rows into train set
edx_removed <- anti_join(edx_temp, edx_test)
edx_train <- rbind(edx_train, removed)
edx_train

rm(edx_temp, edx_removed)

##########################################################
# Data inspection and Summary statistics: edx Dataset
##########################################################

options(digits=5)

# Dataset edx consist of informations (columns) about userID, movieID, Rating, Timestamp, Title and Genres
names(edx)
head(edx) # shows Preview for edx Dataset

summary(edx) # shows summary Statistics for edx Dataset
str(edx) # shows Internal Structure for edx Dataset

# Total rating observations in edx and final_holdout_test dataset
# Together there is 10 000 046 rating observations in edx (9 000 047) and final_holdout_test (999 999) dataset   
total_obs_edx <- length(edx$rating)
total_obs_edx

total_obs_final <- length(final_holdout_test$rating)
total_obs_final

total_obs_both <- length(edx$rating) + length(final_holdout_test$rating)
total_obs_both

# Number of unique movies and users in the edx dataset
# In edx dataset there is 10 669 unique movies and 69 878 users
edx_uniq_movie_user <- edx %>% summarize(number_movies = n_distinct(movieId), number_users = n_distinct(userId))
edx_uniq_movie_user

# Most rated Movies (Top 10): graph
# Most rated movie is Pulp Fiction followed by Forrest Gump, both aired in 1994
edx_most_rated_movies_graph <- edx %>%
  group_by(title) %>%
  summarize(count = n()) %>%
  arrange(-count) %>%
  top_n(10, count) %>%
  ggplot(aes(count, reorder(title, count))) +
  geom_bar(color = "black", fill = "#0B625B", stat = "identity") +
  xlab("Number of Ratings") +
  ylab(NULL)
edx_most_rated_movies_graph

# Most rated Movies (Top 10): table
edx_most_rated_movies_table <- edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  print(n = 10)
edx_most_rated_movies_table

##########################################################
# Ratings and Year split: edx Dataset
##########################################################

# To show rating per Genre and Year in edx dataset we need to subtract the Information from "genres" column (separated by "|") and Year from "title" column
edx_sep_genre_year  <- edx  %>% mutate(year = as.numeric(str_sub(title,-5,-2))) %>% separate_rows(genres, sep = "\\|")
edx_sep_genre_year

# To show average rating per Genre and Year in final_holdout_test dataset we need to subtract Information about Genre from "genres" column (separated by "|") and substract Information about Year from "title" column
final_holdout_test_sep_genre_year <- final_holdout_test  %>% mutate(year = as.numeric(str_sub(final_holdout_test$title,-5,-2))) %>% separate_rows(genres, sep = "\\|")
final_holdout_test_sep_genre_year

edx
final_holdout_test

##########################################################
# Ratings and Genres: edx Dataset
##########################################################

# Movie Rating per genre
# Some genres are rated more often
# Most rated genre is Drama
edx_rating_per_genre <- edx_sep_genre_year%>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  top_n(10, count) %>%
  arrange(desc(count))
edx_rating_per_genre

# Type of ratings: whole and decimal ratings (proportion)
# 79.5 percent of ratings are whole numbers
# Users rate movies more often with whole numbers
ratings_whole_num <- sum(edx$rating %% 1 == 0) / length(edx$rating)
ratings_whole_num

# Distribution of Ratings (Histogram consisting of Ratings and Relative Frequency)
# Users rate movies more often with whole numbers (displayed on graph)
ratings_distrib_graph_freq <- edx %>%
  ggplot() +
  aes(rating, y = after_stat(prop)) +           # y = ..prop.. bolo povodne
  geom_bar(color = "black", fill = "#0B625B") +
  labs(x = "Ratings", y = "Relative Frequency") +
  scale_x_continuous(breaks = c(0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)) + 
  ggtitle("Distribution of Ratings")
ratings_distrib_graph_freq

# Ratings given to movies: Table of Top 5
edx_most_given_ratings <- edx %>% group_by(rating) %>% 
  summarize(count = n()) %>% 
  top_n(5) %>%
  arrange(desc(count))
edx_most_given_ratings

# We want to know if the data are skewed (Number of Ratings vs Users)
ratings_distrib_graph_rat_vs_user <- edx %>% 
  group_by(userId) %>%
  summarize(count = n()) %>%
  ggplot(aes(count)) +
  geom_histogram(color = "black", fill = "#0B625B", bins = 40) +
  xlab("Ratings: Number") +
  ylab("Users: Number") +
  scale_x_log10() +
  ggtitle("Ratings vs Users")
ratings_distrib_graph_rat_vs_user

# Median of Ratings
median(edx$rating)

# Average (Mean) of Ratings
mean(edx$rating)

# Variability of ratings in Genres
edx_var_ratings_genres <- edx_sep_genre_year %>% group_by(genres) %>%
  summarize(n = n(), Average_rating = mean(rating), se = sd(rating)/sqrt(n())) %>%
  mutate(genres = reorder(genres, Average_rating)) %>%
  ggplot(aes(x = genres, y = Average_rating, ymin = Average_rating - 2*se, ymax = Average_rating + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  ggtitle("Variability of ratings in Genres")
edx_var_ratings_genres

# Variability of ratings vs year first aired
edx_var_ratings_years <- edx_sep_genre_year %>% group_by(year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Variability of ratings in Years")
edx_var_ratings_years

##########################################################
# Data Frame for RMSE comparison and results
##########################################################

rmse_comp <- data_frame()
rmse_comp

# The final_holdout_test data should NOT be used for training, developing, or selecting your algorithm and it should ONLY be used for evaluating the RMSE of your final algorithm. 
# The final_holdout_test set should only be used at the end of your project with your final model. 
# It may not be used to test the RMSE of multiple models during model development. 
# You should split the edx data into separate training and test sets and/or use cross-validation to design and test your algorithm.

# Default S3 method:
rmse(actual, predicted, ...)

##########################################################
# Model 1: using only average (mean) of values
##########################################################

# Simplest possible Recommendation System: 
# We predict the same rating for all movies regardless of user
# Method is not sufficient for building of an adequate recommendation system

# estimate that minimizes the RMSE is the least squares estimate of μ
# Overall average (mean) rating on the training dataset
mu <- mean(edx_train$rating)
mu

# We can predict all unknown ratings with mu and calculate the RMSE
RMSE_model_1 <- RMSE(edx_train$rating, mu) # validation using edx training set partition
RMSE_model_1

# Adding RMSE of "Model 1: Mean" to Data Frame for RMSE comparison and results
rmse_comp <- data_frame(Method = "Model 1: Mean", RMSE = RMSE_model_1)
rmse_comp

##########################################################
# We want better Recommendation System (higher quality, lower RMSE)
# After previously inspected data, we see there are some specifics (biases) that can affects the predictions
# Effects on data and predictions: 
# different type of movies rated differently (Movie Effect: b_i), 
# different type of users, taste and their ratings (User Effect: b_u)
# different range of ratings - Regularization - Netflix challenge (Regularized Movie and User Effect)
##########################################################

##########################################################
# Model 2: Movie Effect: b_i
##########################################################

# some movies are just generally rated higher than others

# First, we need to compute Bias based on Movie (b_i)
b_i <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

mean(b_i$b_i)

# Histogram showing variation of estimates b_i
histogram_b_i <- b_i %>% ggplot(aes(b_i)) +
  geom_histogram(color = "black", fill = "#0B625B", bins = 10) +
  xlab("Movie Effects (Bias)") +
  ylab("Count") + 
  ggtitle("Histogram: b_i")
histogram_b_i

# Computing predicted ratings for "Model 2: Movie Effect"
model_2_pred_ratings <- edx_test %>% 
  left_join(b_i, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
model_2_pred_ratings

# Computing and saving RMSE for "Model 2: Movie Effect"
RMSE_model_2 <- RMSE(model_2_pred_ratings, edx_test$rating)
RMSE_model_2

# Adding RMSE of "Model 2: Movie Effect" to Data Frame for RMSE comparison and results
rmse_comp <- bind_rows(rmse_comp,
                       data_frame(Method = "Model 2: Movie Effect",  
                                  RMSE = RMSE_model_2))
rmse_comp

# There is improvement in RMSE


##########################################################
# Model 3: User Effect: b_u (with Movie Effects) 
##########################################################

# There is substantial variability across users (different taste, different ratings)

# First, we need to compute and add Bias based on User (b_u)
b_u <- edx_train %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Histogram showing variation of estimates b_i
histogram_b_u <- b_u %>% ggplot(aes(b_u)) +
  geom_histogram(color = "black", fill = "#0B625B", bins = 30) +
  xlab("User Effects (Bias)") +
  ylab("Count") + 
  ggtitle("Histogram: b_u")
histogram_b_u

# Computing predicted ratings for "Model 3: Movie + User Effect"
model_3_pred_ratings <- edx_test %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_3_pred_ratings

# Computing and saving RMSE for "Model 3: Movie + User Effect"
RMSE_model_3 <- RMSE(model_3_pred_ratings, edx_test$rating)
RMSE_model_3

# Adding RMSE of "Model 3: Movie + User Effect" to Data Frame for RMSE comparison and results
rmse_comp <- bind_rows(rmse_comp,
                       data_frame(Method = "Model 3: Movie + User Effect",  
                                  RMSE = RMSE_model_3))
rmse_comp

# There is another improvement in RMSE


##########################################################
# Model 4: Regularization - Netflix challenge - Movie and User Effect
##########################################################

# There can be noisy estimates that we should not trust
# When making predictions, we need one number, one prediction, not an interval
# Regularization permits us to penalize large estimates that are formed using small sample sizes
# The general idea behind regularization is to constrain the total variability of the effect sizes

# Lambdas (tuning parameter) defined to compute RMSES (from 0 to 10, addition of 0.25)
lambdas <- seq(0, 10, 0.25)

# using function to find RMSEs for Model 4 (Regularized Movie and User Effects) based on defined Lambdas
rmses <- sapply(lambdas, function(lambda){
  
  b_i <- edx_train %>%                                     # Movie Effect (Regularized)
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  
  b_u <- edx_train %>%                                     # Movie and User Effect (Regularized)
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  
  model_4_pred_ratings <- edx_test %>%                     # Predicted Ratings based on Model 4 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)                                             # We want to use predictions based on Model 4
  
  return(RMSE(model_4_pred_ratings, edx_test$rating))      # We want RMSE
})

# Selecting the value that minimizes the RMSE (graph, data)
plot(lambdas, rmses, col = "#0B625B")

lambda_min <- lambdas[which.min(rmses)]
lambda_min

# Defining minimal RMSE for Model 4 and adding to comparison table
RMSE_model_4 <- min(rmses)

rmse_comp <- bind_rows(rmse_comp,
                       data_frame(Method = "Model 4: Regularization: Movie + User Effect",  
                                  RMSE = RMSE_model_4))
rmse_comp

# There is another slight improvement in RMSE


##########################################################
# Data comparison: RMSE of Models
##########################################################

# We want to use most suitable model (lowest RMSE) to test it using "final_holdout_test" dataset
# We are looking for the model with lowest RMSE
# Based on previously defined models, we see that lowest RMSE has "Model 4: Regularization: Movie + User Effect"
# The RMSE for Model 4 is 0.864
# To lower the RMSE we could continue with new model based on Matrix factorization


#########################################################################################################
# Applying "Model 4: Regularization: Movie + User Effect" on "final_holdout_test" dataset -> Final Model
#########################################################################################################

# For Final Model we choose the MIN lambda computed in "Model 4: Regularization: Movie + User Effect"
lambda_min

# Final model based on "Model 4: Regularization: Movie + User Effect"
# For training of our model we use edx dataset and to validate we use final_holdout_test" dataset
# First, we compute average of rating from edx dataset
mu_FINAL <- mean(edx$rating)
mu_FINAL

b_i_FINAL <- edx %>%                                     # Movie Effect (Regularized)
  group_by(movieId) %>%
  summarize(b_i_FINAL = sum(rating - mu_FINAL)/(n()+lambda_min))

b_u_FINAL <- edx %>%                                     # Movie and User Effect (Regularized)
  left_join(b_i_FINAL, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u_FINAL = sum(rating - b_i_FINAL - mu_FINAL)/(n()+lambda_min))

model_FINAL_pred_ratings <- final_holdout_test %>%                 # Predicted Ratings based on FINAL Model 
  left_join(b_i_FINAL, by = "movieId") %>%
  left_join(b_u_FINAL, by = "userId") %>%
  mutate(pred = mu_FINAL + b_i_FINAL + b_u_FINAL) %>%
  pull(pred)

model_FINAL_pred_ratings

# Computing and saving RMSE for Final Model: using "final_holdout_test" dataset
RMSE_model_FINAL <- RMSE(model_FINAL_pred_ratings, final_holdout_test$rating)
RMSE_model_FINAL

# Final RMSE is 0.86482
# Target RMSE by Course: 0.86490
# Final RMSE is below Target RMSE 
