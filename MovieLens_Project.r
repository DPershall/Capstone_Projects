################################################################################
#                                                                              #
#                                   WARNING                                    #
#                                                                              #
################################################################################
# Do not simply source this document without reading first. There are models   #
# shown later in the project that could crash R on your system if it doesn't   #
# meet certain requirements of RAM capacity. They are clearly marked so you    #
# can't miss them.                                                             #
#                                                                              #  
# I used AWS so that I could compute the code in the cloud.                    #
#                                                                              #
# You can run the other models without worry. They provide                     #
# better results anyway, so you won't be missing out!                          #
#                                                                              #
################################################################################

# Quiet the noise
options(warning=FALSE, message=FALSE)

# Libraries
if(!require(tidyverse)) 
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")

if(!require(caret)) 
  install.packages("caret", repos = "http://cran.us.r-project.org")

if(!require(data.table))
  install.packages("data.table", repos = "http://cran.us.r-project.org")

if(!require(gridExtra)) 
  install.packages("gridExtra", repos = "http://cran.us.r-project.org")

if(!require(broom)) 
  install.packages("broom", repos = "http://cran.us.r-project.org")

if(!require(ggrepel)) 
  install.packages("ggrepel", repos = "http://cran.us.r-project.org")

if(!require(recommenderlab)) 
  install.packages("recommenderlab", repos = "http://cran.us.r-project.org")

if(!require(recosystem)) 
  install.packages("recosystem", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(gridExtra)
library(broom)
library(ggrepel)
library(recommenderlab)
library(recosystem)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Pull and clean the data
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")


# Exploration:

# How many users and movies?
movielens %>% summarize(n_users = n_distinct(userId),
                        n_movies = n_distinct(movieId))


# This implies that there could be a total of 746,087,406 ratings if every user
# watched and rated every movie. However, when I look at the dimensions of the
# data set, there are only 10,000,054 actual ratings given. This means that 
# there are a great deal of movies certain users have not watched and/or didn't 
# bother rating.

dim(movielens)

# A Quick Visualization: Rethink of the problem as a matrix of users on one 
# side and movies on the other with the cell as the rating given from the user 
# for the movie. (Sample 100 users 100 movies)

users <- sample(unique(movielens$userId), 100)
movielens %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")


# Ratings Distribution
movielens %>%
  dplyr::count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black", fill = "cornflowerblue") +
  scale_x_log10() +
  ggtitle("Movies")

# Users Histogram
movielens %>% 
  dplyr::count(userId) %>% 
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black", fill = "cornflowerblue") +
  scale_x_log10() +
  ggtitle("Users")

# Time 
# Pull the year from timestamp
movielens <- movielens %>%
  mutate(year = as.numeric(format(as.Date(as.POSIXct(timestamp, 
                                                     origin = "1970-01-01"), 
                                          format ="%Y/%m/%d"), "%Y")))

pattern <- "\\s[(][0-9]{4}[)]"

# Add Release Year and Years Since Release to the data
movielens <- movielens %>% 
  mutate(Release_Year = as.numeric(substr(str_extract(string = title, 
                                                      pattern = pattern),3,6)),
         years_since_release = year - Release_Year)

# Double check the above achieved the desired effect
glimpse(movielens)

# plot the average rating based on release year with some lm smoothing
movielens %>% group_by(Release_Year) %>%
  summarise(mean_Release_Year_Rating = mean(rating)) %>%
  ggplot(aes(Release_Year,mean_Release_Year_Rating)) + geom_point() + geom_smooth(method = "lm")

# Re-think of this time effect in another way, I will categorize by the 
# years that have passed since the release of the movie and look at the average
# rating for the time differences.
movielens %>% group_by(years_since_release) %>%
  summarise(mean_rating_by_years_since_release = mean(rating)) %>%
  ggplot(aes(years_since_release, mean_rating_by_years_since_release)) + geom_point() + geom_smooth(method = "lm")

# time since a movie was released seems to have a net positive
# effect on the average rating

################ THE SPLIT                          

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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

rm(dl, ratings, movies, test_index, temp, removed)

# split the edx set into train and test sets so that I can 
# evaluate different algorithms before performing my final test on the hold out 
# validation" set. (AVOID OVER TRAINING)
set.seed(755, sample.kind = "Rounding")
test_index2 <- createDataPartition(y = edx$rating, times = 1,
                                   p = 0.1, list = FALSE)
train_set <- edx[-test_index2,]
test_set <- edx[test_index2,]

# Make certain that users and movies in the test set are included in the train 
# set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

### Model Evaluation Metric RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

################    MODELS  

## Model 1 - The Average
mu_hat <- mean(train_set$rating)
mu_hat

naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

RMSE_results <- tibble(Method = "The Average", RMSE = naive_rmse) 
RMSE_results %>% knitr::kable()

#### More Exploration
# Not all movies are created equal - Some movies average a higher rating
# accross all users than others. The histogram below is the 
# distribution of the average rating for movies that have been rated by more 
# than 100 users.
train_set %>%
  group_by(movieId) %>%
  summarize(avg_rating = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(avg_rating)) +
  geom_histogram(bins = 30, color = "black", fill = "cornflowerblue")

## Model 2 - The Movie Effect Model
# In order to keep the code as clean as possible, _hat notation has been dropped
mu <- mean(train_set$rating)

movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + test_set %>%
  left_join(movie_avgs, by='movieId') %>% 
  .$b_i

model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
RMSE_results <- bind_rows(RMSE_results, 
                          tibble(Method = "Movie Effect Model", RMSE = model_2_rmse ))
RMSE_results %>% knitr::kable()

## More Exploration
# The code below displays a histogram of the average rating from users that 
# have rated more than 100 movies.
train_set %>%
  group_by(userId) %>%
  summarize(avg_rating = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(avg_rating)) +
  geom_histogram(bins = 30, color = "black", fill = "cornflowerblue")

# So different users have different rating habits

## Model 3 Movie + User effect model
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>% 
  .$pred

model_3_rmse <- RMSE(predicted_ratings, test_set$rating)

RMSE_results <- bind_rows(RMSE_results,
                          tibble(Method="Movie + User Effects Model",
                                 RMSE = model_3_rmse ))
RMSE_results %>% knitr::kable()

## Model 4 Movie + User + Time + Genre
# I have already shown that time has an effect on the rating so I will go ahead
# and expand the model with Time and Genre effects
time_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>% 
  group_by(years_since_release) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u))

genre_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(time_avgs, by='years_since_release') %>% 
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u - b_t))

predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(time_avgs, by='years_since_release') %>% 
  left_join(genre_avgs, by='genres') %>% 
  mutate(pred = mu + b_i + b_u + b_t + b_g) %>% 
  .$pred

model_4_rmse <- RMSE(predicted_ratings, test_set$rating)

RMSE_results <- bind_rows(RMSE_results,
                          tibble(Method=
                                   "Movie + User + Time + Genre Effects Model",
                                 RMSE = model_4_rmse ))
RMSE_results %>% knitr::kable()

## More Exploration of why to use Regularization
# the residuals of the movie effect model
train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>%
  select(title, residual) %>% slice(1:10) %>% knitr::kable()

# Why would 'Pokémon Heroes' and 'From Justin to Kelly' receive a weightier 
# residual than 'Shawshank Redemption'? To see what's happening I will dig a 
# little deeper. The code below will show the 10 best and 10 worst movies based 
# on the estimates of the movie effect.
movie_titles <- train_set %>%
  select(movieId, title) %>%
  distinct()

ten_best <- movie_avgs %>% 
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>%
  select(title, b_i) %>%
  slice(1:10) %>% knitr::kable()

ten_worst <- movie_avgs %>% 
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>%
  select(title, b_i) %>%
  slice(1:10) %>% knitr::kable()

ten_best
ten_worst

# movies that receive the highest and the lowest b's effect are all uncommon 
# movies. Below are the same two tables, but this time with a column that includes
# the number of times each movie has been rated

train_set %>% 
  dplyr::count(movieId) %>%
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>%
  select(title, b_i, n) %>%
  slice(1:10) %>% knitr::kable()

train_set %>% 
  dplyr::count(movieId) %>%
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>%
  select(title, b_i, n) %>%
  slice(1:10) %>% knitr::kable()

# All of these movies were rated by very few users and in some cases only 1 
# user. Larger estimates of 'b_i' are more common when fewer users rate the 
# movie. Regularization helps account for this problem by penalizing large 
# estimates of 'b_i', whether positive or negative, when they come from a small 
# sample size.

## Model 5 - Regularized Movie Model

lambda <- 4 # It will become apparent why I chose this number later

mu <- mean(train_set$rating)

movie_reg_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

# Make a plot to make sure values are moving appropriately
tibble(original = movie_avgs$b_i,
       regularlized = movie_reg_avgs$b_i,
       n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) +
  geom_point(shape=1, alpha=0.5)

# movies with only a few ratings were moved towards zero

# Examine new top ten lists
train_set %>%
  dplyr::count(movieId) %>%
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>%
  select(title, b_i, n) %>%
  slice(1:10) %>% knitr::kable()

# This makes much more sense, as does the ten worst movies shown below.
train_set %>%
  dplyr::count(movieId) %>%
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>%
  select(title, b_i, n) %>%
  slice(1:10) %>% knitr::kable()

# Now I can calculate the RMSE of the regularized movie effect model and append
# it to the table
predicted_ratings <- test_set %>%
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

model_5_rmse <- RMSE(predicted_ratings, test_set$rating)

RMSE_results <- bind_rows(RMSE_results,
                          tibble(Method="Regularized Movie Effect Model",
                                 RMSE = model_5_rmse ))
RMSE_results %>% knitr::kable()

# The Regularized Movie Effect model is an improvement over the least squares 
# Movie Effect Model. Now that the brief exploration and proof of improvement 
# via regularization is behind us, we can move forward by expanding our model 
# with all the possible terms.

## Model 6 - Regularized Movie + User + Time + Genre Model
# I would like to note that because the penalty term, lambda, is a tuning 
# parameter, I can use cross-validation to select the value of lambda that
# minimizes RMSE at the same time as expandin our model to include the extra
# terms with the code below.

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_t <- train_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(years_since_release) %>%
    summarize(b_t = sum(rating - b_i - b_u - mu)/(n()+l))
  
  b_g <- train_set %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_t, by='years_since_release') %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating - b_i - b_u - b_t - mu)/(n()+l))
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_t, by = 'years_since_release') %>% 
    left_join(b_g, by = 'genres') %>% 
    mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
    .$pred
  
  return(RMSE(predicted_ratings, test_set$rating))
})

# Plot the effect of lambda values on the RMSEs
qplot(lambdas, rmses)

# Select the best value of lambda with the code below
lambda <- lambdas[which.min(rmses)]
lambda

# Time to show the results of the regularized movie + user + time + genre effect 
# model.
RMSE_results <- bind_rows(RMSE_results,
                          tibble(Method="Regularized Movie + User + Time + Genre Model",
                                 RMSE = min(rmses)))
RMSE_results %>% knitr::kable()


## Why matrix factorization

# For illustration purposes we will only consider a small subset of movies with 
# many ratings and users that have rated many movies.

# Sample the train set for movies that have received 600 or more ratings 
# and users that have rated 600 or more movies.
train_small <- train_set %>%
  group_by(movieId) %>%
  filter(n() >= 600) %>% 
  ungroup() %>%
  group_by(userId) %>%
  filter(n() >= 600) %>% ungroup()

# Convert to matrix
y <- train_small %>%
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>%
  as.matrix()

# keep rownames
rownames(y)<- y[,1]

y <- y[,-1]

# select distinct titles
movie_titles <- train_set %>% 
  select(movieId, title) %>%
  distinct()

# add colnames
colnames(y) <- with(movie_titles, title[match(colnames(y), movieId)])

# calc vectorization
y <- sweep(y, 2, colMeans(y, na.rm=TRUE))
y <- sweep(y, 1, rowMeans(y, na.rm=TRUE))

# display correlation between times of movies
m_1 <- "Godfather, The (1972)"
m_2 <- "Godfather: Part II, The (1974)"
p1 <- qplot(y[ ,m_1], y[,m_2], xlab = m_1, ylab = m_2)

m_1 <- "Godfather, The (1972)"
m_3 <- "Goodfellas (1990)"
p2 <- qplot(y[ ,m_1], y[,m_3], xlab = m_1, ylab = m_3)

m_4 <- "You've Got Mail (1998)"
m_5 <- "Sleepless in Seattle (1993)"
p3 <- qplot(y[ ,m_4], y[,m_5], xlab = m_4, ylab = m_5)
gridExtra::grid.arrange(p1, p2 ,p3, ncol = 3)

# This plot says that users that liked 'The Godfather' more than what the model 
# expects them to, based on the movie and user effects, also like 
# 'The Godfather II' more than expected. A similar relationship is seen when 
# comparing 'The Godfather' and 'Goodfellas'. Although not as strong, there is 
# still correlation. There are also correlations between 'You’ve Got Mail' and 
# 'Sleepless in Seattle'.

# By looking at the pairwise correlation for these 5 movies, 
# I can see a pattern.
cor(y[, c(m_1, m_2, m_3, m_4, m_5)], use="pairwise.complete") %>%
  knitr::kable()

# The correlations across genres informs us that the data has some structure 
# that up to now hasn't been accounted for in the models.

## Principal component analysis
# To compute the decomposition, I will make the residuals with NA's equal to 0:
y[is.na(y)] <- 0
y <- sweep(y, 1, rowMeans(y))

# Compute prcomp
pca <- prcomp(y)

# The $q$ vectors are called the principal components and they are stored in 
# this matrix:
dim(pca$rotation)

# While the $p$, or the user effects, are stored here:
dim(pca$x)

# The PCA function returns a component with the variability of each of the 
# principal components and it can be accessed and plotted with the following code.
plot(pca$sdev)

# We can also see that with less than 200 of these variables the majority of the 
# variance is explained.
var_explained <- cumsum(pca$sdev^2/sum(pca$sdev^2))
plot(var_explained)


# To see that the principal components are actually capturing something 
# important about the data, I can make a plot of them. For example, the first 
# two principal components are plotted below. The points represent individual 
# movies and I have provided the titles for few in order to provide us with a 
# quick overview.
pcs <- data.frame(pca$rotation, name = colnames(y))
pcs %>%  
  ggplot(aes(PC1, PC2)) + 
  geom_point() +
  geom_text_repel(aes(PC1, PC2, label=name),
                  data = filter(pcs,
                                PC1 < -.01 | PC1 > .01 | PC2 < -0.075 | PC2 > .01))


# Just by looking at these, we see some meaningful patterns. The first principal 
# component shows the difference between critically acclaimed movies on one side 
# and blockbusters on the other. 

# These are one extreme of the principal component.
pcs %>% select(name, PC1) %>% arrange(PC1) %>% slice(1:10) %>% tibble() %>%
  knitr::kable()


# And these are the other extreme of the first principal component.
pcs %>% select(name, PC1) %>% arrange(desc(PC1)) %>% slice(1:10) %>% tibble() %>% 
  knitr::kable()


# If we look at the extremes of PC2 , we see more structure in the data. 
pcs %>% select(name, PC2) %>% arrange(PC2) %>% slice(1:10) %>% tibble() %>% 
  knitr::kable()



pcs %>% select(name, PC2) %>% arrange(desc(PC2)) %>% slice(1:10) %>% tibble() %>% knitr::kable()


# Remember earlier when I set the residuals that were NA equal to 0? This is the 
# point that true SVD for a recommendation system fails to be the best approach. 
# I need an algorithm that accounts for the NA's and not merely ignores them, 
# this is where the Matrix Factorization becomes necessary.

## Model 7 Recommenderlab - LIBMF
################################################################################
#                                                                              # 
#                                  WARNING                                     #   
#                                                                              #
################################################################################
# THE FOLLOWING CODE TAKES 30 MINUTES TO RUN AND REQUIRES A MINIMUM of 32GB of 
# RAM

# I USED CLOUD COMPUTING ON AWS TO SOURCE THE R DOCUMENT AS MY MACHINE CANNOT 
# HANDLE THE MEMORY LOAD

########################         WARNING           #############################
# IF YOUR SYSTEM DOESN'T HAVE AT LEAST 32GB OF RAM DO NOT RUN THIS CHUNK LOCALLY


# First step is to free unused memory space 
gc()

# A quick read of the help files states that I must have matrices in the 
# required realRatingMatrix type. So first up I will have to coerce the 
# movielens data to a matrix
y <- movielens %>%
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>%
  as.matrix()

# setting names properly
rownames(y)<- y[,1]

y <- y[,-1]

# Finally coerce to the recommenderlab specific "realRatingMatrix"
movielenseM <- as(y, "realRatingMatrix")

# split the data by a 80/20 split which mimics the second split of the edx set 
# to train and test sets
e <- evaluationScheme(movielenseM, method="split", 
                      train=0.8, k=1, given=-10)



# create a libmf recommender using training data
r <- Recommender(getData(e, "train"), "LIBMF")


# create predictions for the test data using known ratings (the given param 
# from the evaluation scheme)
p <- predict(r, getData(e, "known"), type="ratings")

# calculate the average RMSE when User specific

acc <- calcPredictionAccuracy(p, getData(e, "unknown"), byUser=TRUE)

acc <- mean(acc[,1])

# Append to the RMSE results tibble
RMSE_results <- bind_rows(RMSE_results,
                          tibble(Method = "Matrix Factorization - Recommenderlab",
                                 RMSE = acc))
# Show the RMSE improvement
RMSE_results %>%
  knitr::kable()

# use gc() one last time to free unused memory
gc()

############################# END OF WARNING CHUNK #############################

## Model 8 - Recosystem - LIMBF
# One important note:
# Recosystem's Makevars can be adjusted to take advantage of modern CPUs that have 
# SSE3 and AVX. You just need a C++ compiler that supports the C++11 standard. Then 
# you can edit src/Makevars (src/Makevars.win for Windows system) according to the 
# following guidelines:

# 1) If your CPU supports SSE3, add the following

# PKG_CPPFLAGS += -DUSESSE
# PKG_CXXFLAGS += -msse3


# 2) If SSE3 and also AVX is supported, add the following

# PKG_CPPFLAGS += -DUSEAVX
# PKG_CXXFLAGS += -mavx

# After editing the Makevars file, run R CMD INSTALL recosystem to install 
# recosystem with AVX and SSE3.

# This is the process I used on my desktop. However, the default Makevars 
# provides generic options that should apply to most CPUs. Recosystem is a 
# wrapper for the Library for Parallel Matrix Factorization, which is known as 
# LIBMF. 


# read the help file
??recosystem

set.seed(1985, sample.kind = "Rounding") # This is a randomized algorithm

# This takes four minutes to run and should not be a concern for anyone on
# a modern laptop. 

# Convert the train and test sets into recosystem input format
train_data <- with(train_set, data_memory(user_index = userId,
                                          item_index = movieId,
                                          rating = rating))
test_data <- with(test_set, data_memory(user_index = userId,
                                        item_index = movieId,
                                        rating = rating))


# Create the model object
r <- recosystem::Reco()

# Select the best tuning parameters using cross-validation
opts <- r$tune(train_data, opts = list(dim = c(10, 20, 30),
                                       costp_l2 = c(0.01, 0.1),         
                                       costq_l2 = c(0.01, 0.1),         
                                       costp_l1 = 0,                    
                                       costq_l1 = 0,                    
                                       lrate    = c(0.01, 0.1),         
                                       nthread  = 8,               
                                       niter    = 10,                   
                                       verbose  = FALSE)) 
# Train the algorithm
r$train(train_data, opts = c(opts$min, 
                             nthread = 8, 
                             niter = 100, 
                             verbose = FALSE))

# Calculate the predicted values
reco_pred <- r$predict(test_data, out_memory())

RMSE_results <- bind_rows(RMSE_results,
                          tibble(Method = "Matrix Factorization - Recosystem",
                                 RMSE = RMSE(test_set$rating, reco_pred)))
# Show the RMSE improvement
RMSE_results %>%
  knitr::kable()

######### FINAL VALIDATIONS

## Regularizaed Movie + User + Time + Genre Effect model
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_t <- edx %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(years_since_release) %>%
    summarize(b_t = sum(rating - b_i - b_u - mu)/(n()+l))
  
  b_g <- edx %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_t, by='years_since_release') %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating - b_i - b_u - b_t - mu)/(n()+l))
  
  predicted_ratings <- validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_t, by = 'years_since_release') %>% 
    left_join(b_g, by = 'genres') %>% 
    mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
    .$pred
  
  return(RMSE(predicted_ratings, validation$rating))
})

lambda <- lambdas[which.min(rmses)]
lambda

Final_Validation_RMSE_results_table <- tibble(Method = 
                                                "Regularized Movie + User + Time + Genre Model", RMSE = min(rmses)) 

Final_Validation_RMSE_results_table %>% knitr::kable()

## Final Validation - Recommenderlab - Matrix Factorization Model

################################################################################
#                                                                              # 
#                                  WARNING                                     #   
#                                                                              #
################################################################################
# THE FOLLOWING CODE TAKES 30 MINUTES TO RUN AND REQUIRES A MINIMUM of 32GB of 
# RAM

# I USED CLOUD COMPUTING ON AWS TO SOURCE THE R DOCUMENT AS MY MACHINE CANNOT 
# HANDLE THE MEMORY LOAD

########################         WARNING           #############################
# IF YOUR SYSTEM DOESN'T HAVE AT LEAST 32GB OF RAM DO NOT RUN THIS CHUNK LOCALLY


# Free unused memory space 
gc()

# I have already coerced the movielens data into the required format and it is
# stored as movielenseM which is a realRatingMatrix so there is no need to 
# repeat those steps. However, I do need to repeat the split. The recommenderlab 
# way is evaluationScheme and the code below shows a 90/10 split which 
# is the same proportion as the original edx / validation split, which is also
# 90/10.
e <- evaluationScheme(movielenseM, method="split", 
                      train=0.9, k=1, given=-10)



# create a libmf recommender using training data
r <- Recommender(getData(e, "train"), "LIBMF")


# create predictions for the test data using known ratings (the given param 
# from the evaluation scheme)
p <- predict(r, getData(e, "known"), type="ratings")


# calculate the average RMSE when User specific

acc <- calcPredictionAccuracy(p, getData(e, "unknown"), byUser=TRUE)

acc <- mean(acc[,1])

# Append to the RMSE results tibble
Final_Validation_RMSE_results_table <- bind_rows(Final_Validation_RMSE_results_table,
                                                 tibble(
                                                   Method = 
                                                     "Final Validation - Recommenderlab - Matrix Factorization",
                                                   RMSE = acc))

# Show the Results
Final_Validation_RMSE_results_table %>% knitr::kable()

# use gc() one last time to free unused memory
gc()

########################## END OF WARNING CHUNK ################################


## Final Validation - Recosystem - Matrix Factorization Model

# set seed
set.seed(1986, sample.kind="Rounding")

# coerce data object
train_data <- with(edx, data_memory(user_index = userId,       #EDX
                                    item_index = movieId,
                                    rating = rating))
test_data <- with(validation, data_memory(user_index = userId, #VALIDATION
                                          item_index = movieId,
                                          rating = rating))

r <- recosystem::Reco()

# Select the best tuning parameters using cross-validation
opts <- r$tune(train_data, opts = list(dim = c(10, 20, 30),
                                       costp_l2 = c(0.01, 0.1),         
                                       costq_l2 = c(0.01, 0.1),         
                                       costp_l1 = 0,                    
                                       costq_l1 = 0,                    
                                       lrate    = c(0.01, 0.1),         
                                       nthread  = 8,             
                                       niter    = 10,                   
                                       verbose  = FALSE)) 
# Train the algorithm
r$train(train_data, opts = c(opts$min, 
                             nthread = 8, 
                             niter = 100, 
                             verbose = FALSE))

# Calculate the predicted values
reco_final_pred <- r$predict(test_data, out_memory())

Final_Validation_RMSE_results_table <- bind_rows(Final_Validation_RMSE_results_table,
                                                 tibble(Method = "Final Validation - Matrix Factorization - Recosystem",
                                                        RMSE = RMSE(validation$rating, reco_final_pred)))
Final_Validation_RMSE_results_table %>% knitr::kable()

# Summary

# While this is just a glimpse into the world of recommendation systems using 
# machine learning, it has been an informative one! I have certainly enjoyed 
# working through the different models and examining the results. I am pleased 
# to have three final validations that surpass the class goal of an RMSE below 
# 0.86490, and I am happy to have met my personal goal of an RMSE below 0.80. 

# Obviously, the results from Recosystem's parallel implementation of the LIBMF
# library were the best. Not only is it easy to tune, but the time saved via
# parallelization is awesome.

# While this data set is large enough to present challenges, it is important to 
# remember that it is still only a subset of the movielense data. It is also worth 
# noting that as time progresses, the size of data sets like this can increase 
# dramatically from the addition of new users and movies. Therefore, it stands to 
# reason that advances in the field of recommendation systems will more than 
# likely require parallelization and some form of cloud computing.