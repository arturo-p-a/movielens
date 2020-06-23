#-------------------------------------------------------------------------------
# Dataset preparation
#-------------------------------------------------------------------------------

# Required libraries

if(!require(tidyverse)) 
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")

if(!require(caret)) 
  install.packages("caret", repos = "http://cran.us.r-project.org")

if(!require(data.table)) 
  install.packages("data.table", repos = "http://cran.us.r-project.org")

# Download files if they are not present in the working directory.

if(!file.exists("ml-10M100K/ratings.dat") | !file.exists("ml-10M100K/movies.dat")) {
  dl <- tempfile()
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)  
  unzip(dl, "ml-10M100K/ratings.dat")
  unzip(dl, "ml-10M100K/movies.dat")
  rm(dl)
}

# Read data from the downloaded files

ratings <- fread(text = gsub("::", "\t", readLines("ml-10M100K/ratings.dat")), 
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines("ml-10M100K/movies.dat"), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1) # if using R higher than 3.5 , use `set.seed(1, sample.kind="Rounding")`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(ratings, movies, test_index, temp, movielens, removed)

#-------------------------------------------------------------------------------
# Analysis
#-------------------------------------------------------------------------------

# We are going to split the _edx_ set into a train_set and a test_set
# so we can "play" without using the validation set 

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test_set are also in train_set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test_set back into train_set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)
rm(test_index, temp, removed)

# We will also define the same function _RMSE_ used in the Machine Learning course:

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#-------------------------------------------------------------------------------
# Model #1
#-------------------------------------------------------------------------------

# We are going to use the _Regularized Movie + User Effect Model_ built in the Machine Learning course as our first model.
# The code is almost the same used in the course but enclosed in a function to make the optimization easier.

model1 <- function(train_set, test_set, lambda) {
  
  # Overall average
  mu <- mean(train_set$rating)

  # Regularized movie effect
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))

  # Regularized user effect
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  
  # Prediction
  predictions <-
    test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)

  return (predictions)
}

# We are going the penalty, searching in the vicinity of the optimal value found in the course 3.75.

lambdas <- seq(2.75, 4.75, 0.25)

# Compute the RMSE for each lambda
rmses <- sapply(lambdas, function(lambda) {
  predictions <- model1(train_set, test_set, lambda)
  return(RMSE(predictions, test_set$rating))
})

# Plot the result
qplot(lambdas, rmses)

# Our optimal penalty is lambda = 4.5.

# Display tibbles with 7 significant digits
options(pillar.sigfig = 7)

# Store the results of model #1
rmse_results <- tibble(
  method = "Model #1", 
  RMSE = min(rmses), 
  lambda = lambdas[which.min(rmses)]
)

# And display it
rmse_results

#-------------------------------------------------------------------------------
# Model #2
#-------------------------------------------------------------------------------

# We are going to improve our initial model adding more biases.

# Let's analyze the timestamp effect plotting the average ratings over time
# We can see fluctuactions that we can capture including the timestamp as a predictor
# For this quick plot we are going to group the timestamps in periods of 30 days.
train_set %>% 
  # 30 days * 24 hours/day * 3600 seconds/hour
  mutate(rating_period = floor(timestamp / (30*24*3600))) %>%   
  group_by(rating_period) %>%
  summarize(mean = mean(rating), count = n()) %>%
  filter(count > 20) %>%
  ggplot(aes(rating_period, mean)) +
  geom_smooth(method='loess', span=0.3) +
  geom_point()

# Let's analyze the genre effect plotting a histogram of the average rating for each genre.
# We can see a great dispersion, so we can use the genre to make a better guess of the ratings
train_set %>% 
  group_by(genres) %>%
  summarize(mean = mean(rating), count = n()) %>%
  filter(count > 20) %>%
  ggplot(aes(mean)) +
  geom_histogram(bins = 30)

# Let's analyze the release year effect.
# This information is encoded in the title so we have to extrat it.
# We are going to build a data frame with each movieId and release year
release_years <- train_set %>% 
  distinct(movieId, title) %>% 
  extract(title, c("title_tmp", "release_year"), 
          regex = "^(.*)\\(([0-9]{4})\\)$", remove = F) %>% 
  mutate(release_year = strtoi(release_year)) %>%
  select(movieId, release_year)

head(release_years)

# We can join the train_set and the data frame we have just created
# to plot the average rating for each year.
# We can see a pattern where old movies are higher rated than modern.
train_set %>% 
  left_join(release_years, by='movieId') %>%
  group_by(release_year) %>%
  summarize(mean = mean(rating), count = n()) %>%
  filter(count > 100) %>%
  ggplot(aes(release_year, mean)) +
  geom_smooth(method='loess', span=0.3) +
  geom_point()

# The three variables analyzed seem to provide relevant information, 
# so we are going to include them in this second model.

# We need the lubridate package to interpret the timestamp
if(!require(lubridate)) 
  install.packages("lubridate", repos = "http://cran.us.r-project.org")

# We will enclose the model into a function as we did before:
model2 <- function(train_set, test_set, lambda) {
  
  # Overall average
  mu <- mean(train_set$rating)

  # Regularized movie effect
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))

  # Regularized user effect
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

  # NEW TERM  
  # Regularized rating month effect
  b_t <- train_set %>%
    # The rating_period is YEAR-MONTH, ex: '2020-06'
    mutate(rating_period = format(as_datetime(timestamp), '%Y-%m')) %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(rating_period) %>%
    summarize(b_t = sum(rating - b_i - b_u - mu)/(n()+lambda))
  
  # NEW TERM
  # Regularized genre effect
  b_g <- train_set %>%
    mutate(rating_period = format(as_datetime(timestamp), '%Y-%m')) %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_t, by="rating_period") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - b_t - mu)/(n()+lambda))

  # MOVIEID-RELEASE_YEAR DATAFRAME
  # Build the data frame (movieId, release_year) before computing the release_year effect.
  # We are building it into this function because we are going to perform crossvalidation
  # and the movies in the trainset will not be the same
  release_years <- train_set %>% 
    distinct(movieId, title) %>% 
    extract(title, c("title_tmp", "release_year"), 
            regex = "^(.*)\\(([0-9]{4})\\)$", remove = F) %>% 
    mutate(release_year = strtoi(release_year)) %>%
    select(movieId, release_year)
  
  # NEW TERM
  # Regularized release year effect
  b_y <- train_set %>%
    mutate(rating_period = format(as_datetime(timestamp), '%Y-%m')) %>%
    left_join(release_years, by="movieId") %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_t, by="rating_period") %>%
    left_join(b_g, by="genres") %>%
    group_by(release_year) %>%
    summarize(b_y = sum(rating - b_i - b_u - b_t - b_g - mu)/(n()+lambda))
  
  
  # Prediction
  predictions <-
    test_set %>%
    mutate(rating_period = format(as_datetime(timestamp), '%Y-%m')) %>%
    left_join(release_years, by = "movieId") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_t, by = "rating_period") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_y, by = "release_year") %>%
    mutate(pred = mu + b_i + b_u + b_t + b_g + b_y) %>%
    pull(pred)

  return (predictions)
}

# Let's find the optimal penalty for this model:
mbdas <- seq(3.5, 5.5, 0.25)

# compute the rmses
rmses <- sapply(lambdas, function(lambda) {
  predictions <- model2(train_set, test_set, lambda)
  return(RMSE(predictions, test_set$rating))
})

# and plot the result
qplot(lambdas, rmses)

# In this case the optimal lambda is 4.75, and the RMSE = 0.8648421.
# Let's add the RMSE computed for this model to _rmse_results_.
rmse_results <- bind_rows(rmse_results, tibble(
  method = "Model #2", 
  RMSE = min(rmses), 
  lambda = lambdas[which.min(rmses)]
))

rmse_results

#-------------------------------------------------------------------------------
# Model #3
#-------------------------------------------------------------------------------

# There are biases related to the deviation of the ratings around the mean. 
# Let's see it with an example, plotting the ratings given by user 12760
train_set %>%
  filter(userId == 12760) %>%
  ggplot(aes(rating)) +
  geom_histogram(bins=11, breaks = seq(0.25,5.25,0.5))

# This user has rated 518 films and never gave a rating below 2 or above 4.
# If we plot the histogram of ranges used by de users (difference between the maximum rating and the minimum), 
# we can see that few users uses the entire range of ratings available.
train_set %>%
  group_by(userId) %>%
  summarize(range = max(rating) - min(rating), count=n()) %>%
  filter(count>=20) %>%
  ggplot(aes(range)) +
  geom_histogram(bins=11, breaks = seq(0.25,5.25,0.5))

# We will limit our predictions to the range o ratings used by the user in the trainig set

# We enclose the model into a function as before.
# It's essentially the same as in model #2 with minor modifications, commented in capitals.
model3 <- function(train_set, test_set, lambda) {
  
  # Overall average
  mu <- mean(train_set$rating)

  # Regularized movie effect
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))

  # Regularized user effect
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  
  # Regularized rating month effect
  b_t <- train_set %>%
    mutate(rating_period = format(as_datetime(timestamp), '%Y-%m')) %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(rating_period) %>%
    summarize(b_t = sum(rating - b_i - b_u - mu)/(n()+lambda))
  
  # Regularized genre effect
  b_g <- train_set %>%
    mutate(rating_period = format(as_datetime(timestamp), '%Y-%m')) %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_t, by="rating_period") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - b_t - mu)/(n()+lambda))

  # Regularized release year effect
  release_years <- train_set %>% 
    distinct(movieId, title) %>% 
    extract(title, c("title_tmp", "release_year"), regex = "^(.*)\\(([0-9]{4})\\)$", remove = F) %>% 
    mutate(release_year = strtoi(release_year)) %>%
    select(movieId, release_year)
  
  b_y <- train_set %>%
    mutate(rating_period = format(as_datetime(timestamp), '%Y-%m')) %>%
    left_join(release_years, by="movieId") %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_t, by="rating_period") %>%
    left_join(b_g, by="genres") %>%
    group_by(release_year) %>%
    summarize(b_y = sum(rating - b_i - b_u - b_t - b_g - mu)/(n()+lambda))
  
  # THIS IS THE FIRST CHANGE INCLUDED IN THIS MODEL
  # Minimum and maximum ratings per user
  user_ranges <- train_set %>%
    group_by(userId) %>%
    summarise(min = min(rating), max=max(rating))
  
  # Prediction
  predictions <-
    test_set %>%
    mutate(rating_period = format(as_datetime(timestamp), '%Y-%m')) %>%
    left_join(release_years, by = "movieId") %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_t, by = "rating_period") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_y, by = "release_year") %>%
    # WE GET THE MAX AND MIN FOR EACH USER
    left_join(user_ranges, by = "userId") %>%
    mutate(
      # COMPUTE THE PREDICTION AS IN MODEL #2
      old_pred = mu + b_i + b_u + b_t + b_g + b_y, 
      # AND THEN LIMIT THE RESULT BASED ON THE MIN AND MAX FOR THE USER
      pred = ifelse(old_pred > max, max, ifelse(old_pred < min, min, old_pred ))
    ) %>%
    pull(pred)

  return (predictions)
}

#Let's find the optimal penalty for this model:
lambdas <- seq(3.75, 5.75, 0.25)

# compute the rmses
rmses <- sapply(lambdas, function(lambda) {
  predictions <- model3(train_set, test_set, lambda)
  return(RMSE(predictions, test_set$rating))
})

# and plot the result
qplot(lambdas, rmses)

# The optimal penalty for this model is lambda = 4.5, and the RMSE = 0.8646335.
rmse_results <- bind_rows(rmse_results, tibble(
  method = "Model #3", 
  RMSE = min(rmses), 
  lambda = lambdas[which.min(rmses)]
))

rmse_results

#-------------------------------------------------------------------------------
# Parameter optimization
#-------------------------------------------------------------------------------

# We are going to optimize the penalty parameter using 10-fold crossvalidation.

##### THIS BLOCK OF CODE IS VERY TIME CONSUMING #####

# lambdas near 4.5
lambdas <- seq(4, 5, 0.25)

# list of 10 test indexes
# We have to suffle the indexes before spliting for the reasons commented in movielens.pdf
indexes <- split(sample(seq(1, nrow(edx))), seq(1,10))

# compute the rmse for each test index
rmses_folds <- sapply(indexes, function(index) {
  
  # Split edx into train and test sets using the current index
  test_index <- index
  train_set <- edx[-test_index,]
  temp <- edx[test_index,]
  test_set <- temp %>% 
    semi_join(train_set, by = "movieId") %>%
    semi_join(train_set, by = "userId")
  removed <- anti_join(temp, test_set)
  train_set <- rbind(train_set, removed)
  rm(test_index, temp, removed)
  
  # Compute the RMSE for each penalty
  rmses_partition <- sapply(lambdas, function(lambda) {
    predictions <- model3(train_set, test_set, lambda)
    return(RMSE(predictions, test_set$rating))  
  })
  
  # Return the RMSEs
  return(rmses_partition)
  
})

# plot the mean rmse for each lambda
rmses <- rowSums(rmses_folds)/10
qplot(lambdas, rmses)


# We will finally use a penalty value of lambda = 4.75
rmse_results <- bind_rows(rmse_results, tibble(
  method = "Model #3 after optimization", 
  RMSE = min(rmses), 
  lambda = lambdas[which.min(rmses)]
))

rmse_results


# Let's now apply our final model to the validation set:
lambda <- 4.75
predictions <- model3(edx, validation, lambda)
RMSE(predictions, validation$rating)

# Using the proposed model, we have achieved an RMSE of 0.863978, clearly below the initial target.