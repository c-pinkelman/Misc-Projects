###################################################
## READ IN LIBRARIES AND DATA
###################################################

library(readr)
library(tidyverse)
library(ggplot2)
library(lightgbm)
library(reshape2)
library(lubridate)
library(rpart)

# read in data
account_dat <- read_csv("Downloads/account_dat.csv")
category_ref <- read_csv("Downloads/category_ref.csv")
device_ref <- read_csv("Downloads/device_ref.csv")
in_app_dat <- read_csv("Downloads/in-app_dat.csv")
app_dat <- read_csv("Downloads/app_dat.csv")
transaction_dat <- read_csv("Downloads/transaction_dat.csv")

###################################################
## DATA CLEANING AND MANIPULATION
###################################################

# main data frame
all_dat <- left_join(transaction_dat, account_dat, by = 'acct_id')
all_dat <- left_join(all_dat, app_dat, by = 'content_id')
all_dat <- left_join(all_dat, in_app_dat, by = 'content_id')
all_dat <- left_join(all_dat, device_ref, by = c('device_id.x' = 'device_id'))
all_dat <- left_join(all_dat, category_ref, by = 'category_id')

colnames(all_dat) <- c(
  'trans_create_dt'
  ,'content_id'
  ,'acct_id'
  ,'price'
  ,'trans_device_id'
  ,'acct_create_dt'
  ,'payment_type'
  ,'app_name'
  ,'category_id'
  ,'app_device_id'
  ,'parent_app_content_id'
  ,'inapp_type'
  ,'device_name'
  ,'category_name'
)  

# fix format of dates
head(all_dat$acct_create_dt)
all_dat$acct_create_dt <- as.POSIXct(all_dat$acct_create_dt, format="%m/%d/%Y", tz="UTC")
all_dat$acct_create_dt <- all_dat$acct_create_dt %m+% years(2000)
all_dat[which(is.na(all_dat$acct_create_dt)),]$acct_create_dt <- median(all_dat$acct_create_dt, na.rm = TRUE)
head(all_dat$acct_create_dt)

head(all_dat$trans_create_dt)
all_dat$trans_create_dt <- as.POSIXct(all_dat$trans_create_dt, format="%Y-%m-%d", tz="UTC")
head(all_dat$trans_create_dt)

as.numeric((difftime(max(all_dat$trans_create_dt), min(all_dat$acct_create_dt), units = 'days')))/365

# save main data frame
write.csv(all_dat, "all_dat.csv", row.names=FALSE)

###################################################
## GENERAL EDA
###################################################

# distribution of transaction spends
hist(all_dat$price, breaks = 25)

# distribution of spend per user
per_user_spend <- all_dat %>% group_by(acct_id) %>%
    summarise(n = n()
      ,n_paid = length(which(price > 0))
      ,n_free = length(which(price == 0))
      ,pct_paid =  length(which(price > 0))/n()
      ,total_spend = sum(price)
      ,med_spend = median(price)
      ,mean_spend = mean(price))

ggplot(per_user_spend, aes(x=pct_paid)) + 
  geom_histogram(binwidth = 0.025) + 
  theme_bw() +
  ggtitle("Paid App Purchases Per Account") +
  labs(y = "Number of Accounts", x = "Proportion of Paid Purchases")

hist(per_user_spend$n, breaks = 25)
summary(per_user_spend$n)
hist(per_user_spend$total_spend, breaks = 25)
summary(per_user_spend$total_spend)
hist(per_user_spend$med_spend, breaks = 25)
summary(per_user_spend$med_spend)
hist(per_user_spend$mean_spend, breaks = 25)
summary(per_user_spend$mean_spend)

length(which(per_user_spend$total_spend > 0))/nrow(per_user_spend)

###################################################
## PER USER TABLE
###################################################

freqfunc <- function(x, n){
  names(tail(sort(table(unlist(strsplit(as.character(x), ", ")))), n))
}

final_dat <- all_dat %>% group_by(acct_id) %>%
  summarise(n = n()
            ,n_paid = length(which(price > 0))
            ,n_free = length(which(price == 0))
            ,pct_paid = length(which(price > 0))/n()
            ,total_spend = sum(price)
            ,cust_tenure = as.numeric(difftime(max(trans_create_dt), min(acct_create_dt), units = 'days'))/365
            ,n_unique_apps =length(unique(app_name))
            ,n_pmof = length(which(payment_type == 'PMOF'))
            ,n_free = length(which(payment_type == 'Free only'))
            ,n_subscription = length(which(inapp_type == 'subscription'))
            ,n_consumable = length(which(inapp_type == 'consumable'))
            #,fave_method = freqfunc(app_device_id, 1)
            ,n_iphone = length(which(device_name == 'iPhone'))
            ,n_ipad = length(which(device_name == 'iPad')) 
            ,iphone_user = ifelse(length(which(device_name == 'iPhone')) > length(which(device_name == 'iPad')), 1, 0)
            ,ipad_user = ifelse(length(which(device_name == 'iPad')) > length(which(device_name == 'iPhone')), 1, 0)
            ,n_enter = length(which(category_name == 'Entertainment'))
            ,n_games = length(which(category_name == 'Games'))
            ,n_photos = length(which(category_name == 'Photos & Videos'))
            ,n_social = length(which(category_name == 'Social Networking'))
            ,n_utilities = length(which(category_name == 'Utilities'))
            ,fave_app_type = freqfunc(category_name, 1)
  )

final_dat$paid_status <- ifelse(final_dat$pct_paid > .5, 1, 0)

###################################################
## FREE VS. PAYING USERS
###################################################

## creating correlation matrix
# drop irrelevant columns
cor_dat <- final_dat[,-which(names(final_dat) %in% c('acct_id', 'n', 'paid_status', 'n_paid', 'total_spend'
                                                     , 'ipad_user', 'iphone_user', 'n_free', 'fave_app_type'))]
corr_mat <- round(cor(cor_dat),3)
# reduce the size of correlation matrix
melted_corr_mat <- melt(corr_mat)
# plotting the correlation heatmap
ggplot(data = melted_corr_mat, aes(x=Var1, y=Var2, fill=value)) + geom_tile() +
  geom_text(aes(Var2, Var1, label = value), color = "white", size = 4)

# examine features before training
lapply(final_dat,summary)

# build simple decision tree and plot
samps <- sample(1:2, nrow(final_dat), replace=TRUE, prob = c(0.7, 0.3))
train <- final_dat[samps == 1,]
test <- final_dat[samps == 2,]
tree <- rpart(paid_status ~ cust_tenure +
                #n_unique_apps +
                n_iphone +
                n_ipad +
                n_enter +
                n_games + 
                n_photos +
                n_social +
                n_utilities +
                fave_app_type
                , data = train
                , method = "class")
rpart.plot::rpart.plot(tree, 
                       main = "Decision Tree for Predicting Free vs. Paying Users", cex.sub = .8)

###################################################
## FREE USERS
###################################################

## what predicts a high number of free transactions
free_users <- final_dat[which(final_dat$paid_status == 0 & final_dat$n_free > 0), 
                        -which(names(final_dat) %in% c('n', 'paid_status',
                                                       'n_paid', 'pct_paid', 'total_spend'))]
## FIX CUT POINTS BASED ON MEAN+1STD

# cut points based on mean and standard deviation
mu <- mean(free_users$n_free)
sigma1 <- mean(free_users$n_free) - sd(free_users$n_free)
sigma2 <- mean(free_users$n_free) + sd(free_users$n_free)

ggplot(free_users, aes(x=n_free)) + 
  geom_histogram(binwidth = 2) + 
  theme_bw() +
  ggtitle("Distribution of Free App Transactions") +
  labs(y = "Number of Accounts", x = "Number of Free Transactions") +
  geom_vline(xintercept=sigma1, linetype="dashed", color = "red") +
  geom_vline(xintercept=sigma2, linetype="dashed", color = "red")

## what predicts high volume for free users?
free_users$n_category <- cut(free_users$n_free, breaks = c(0, sigma1, sigma2, max(free_users$n_free)),
    labels = c("low", "mid", "high"))
table(free_users$n_category)/nrow(free_users)

# build simple decision tree and plot
samps <- sample(1:2, nrow(free_users), replace=TRUE, prob = c(0.7, 0.3))
train <- free_users[samps == 1,]
test <- free_users[samps == 2,]
tree <- rpart(n_category ~ cust_tenure +
                #n_unique_apps +
                n_iphone +
                n_ipad +
                n_enter +
                n_games + 
                n_photos +
                n_social +
                n_utilities +
                fave_app_type 
              , data = train
              , method = "class")
rpart.plot::rpart.plot(tree, 
                       main = "Decision Tree for Predicting Number of Transactions for Free Users", cex.sub = .8)
                       # box.palette = "blue")

###################################################
## PAYING USERS
###################################################
paying_users <- final_dat[which(final_dat$paid_status == 1), 
                        -which(names(final_dat) %in% c('n', 'paid_status',
                                                       'n_free', 'pct_paid', 'n_paid'))]
# distributions of n free
ggplot(paying_users, aes(x=total_spend)) + 
  geom_histogram(binwidth = 10) + 
  theme_bw() +
  ggtitle("Distribution of App Store Spend ($)") +
  labs(y = "Number of Accounts", x = "Dollars Spent")

ggplot(paying_users[which(paying_users$total_spend > 5000),], aes(x=total_spend)) + 
  geom_histogram(binwidth = 50) + 
  theme_bw() +
  ggtitle("Distribution of App Store Spend ($) - Outlier Group") +
  labs(y = "Number of Accounts", x = "Dollars Spent") 

paying_users_outliers <- paying_users[which(paying_users$total_spend > 5000),]
paying_users_normal <- paying_users[which(paying_users$total_spend <= 5000),]

outliers_transactions <- all_dat[which(all_dat$acct_id %in% paying_users_outliers$acct_id),]
table(outliers_transactions$price)

nonoutliers_transactions <- all_dat[which(!(all_dat$acct_id %in% paying_users_outliers$acct_id) & 
                                            !(all_dat$acct_id %in% free_users$acct_id)),]

# distribution of prices
table(outliers_transactions$price)/nrow(outliers_transactions)
table(nonoutliers_transactions$price)/nrow(nonoutliers_transactions)

# three roughly even groups for spend
mu <- mean(paying_users_normal$total_spend)
sigma1 <- mean(paying_users_normal$total_spend) - sd(paying_users_normal$total_spend)
sigma2 <- mean(paying_users_normal$total_spend) + sd(paying_users_normal$total_spend)

ggplot(paying_users_normal, aes(x=total_spend)) + 
  geom_histogram(binwidth = 50) + 
  theme_bw() +
  ggtitle("Distribution of App Store Spend ($) - Typical Group") +
  labs(y = "Number of Accounts", x = "Dollars Spent")  +
  geom_vline(xintercept=sigma1, linetype="dashed", color = "red") +
  geom_vline(xintercept=sigma2, linetype="dashed", color = "red")

paying_users_normal$n_category <- cut(paying_users_normal$total_spend, breaks = c(0, sigma1, sigma2, max(paying_users_normal$total_spend)),
                             labels = c("low", "mid", "high"))
table(paying_users_normal$n_category)/nrow(paying_users_normal)

# build simple decision tree and plot
samps <- sample(1:2, nrow(paying_users_normal), replace=TRUE, prob = c(0.7, 0.3))
train <- paying_users_normal[samps == 1,]
test <- paying_users_normal[samps == 2,]
tree <- rpart(n_category ~ cust_tenure +
                n_unique_apps +
                #n_iphone +
                n_ipad +
                n_enter +
                n_games + 
                n_photos +
                n_social +
                n_utilities +
                fave_app_type 
              , data = train
              , method = "class")
rpart.plot::rpart.plot(tree, 
                       main = "Decision Tree for Predicting Total Spend for Paying Users", cex.sub = .8)

# decision tree did not work well here, how does a linear model do?
summary(glm(total_spend ~ cust_tenure +
              #n_unique_apps +
              #n_iphone +
              #n_ipad +
              n_enter +
              n_games + 
              n_photos +
              n_social +
              n_utilities
              #fave_app_type 
            , data = train))
