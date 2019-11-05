library(data.table)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(knitr)

#-------------------------------------------------------------------------------------------------------------------------------
#             Loading the datasets
#-------------------------------------------------------------------------------------------------------------------------------

orders <- fread('orders.csv')
products <- fread('products.csv')
order_products <- fread('order_products__train.csv')
order_products_prior <- fread('order_products__prior.csv')
aisles <- fread('aisles.csv')
departments <- fread('departments.csv')

#-------------------------------------------------------------------------------------------------------------------------------
#             EDA Orders Data
#-------------------------------------------------------------------------------------------------------------------------------

str(orders)
summary(orders)
kable(head(orders,50))
table(orders$order_dow)

# Trend on Day of Week
ggplot(orders, aes(x = order_dow)) + geom_bar()

# Trend on Hour of Day
ggplot(orders, aes(x = order_hour_of_day)) + geom_bar()

#Trend on Day of week and Hour of Day
orders %>% group_by(order_dow, order_hour_of_day) %>% ggplot(aes(x = order_hour_of_day)) +geom_bar() + 
  facet_wrap(~order_dow)

# DOW - 0,1 - Seems to be saturday and sunday  

#-------------------------------------------------------------------------------------------------------------------------------
#             EDA Products Data
#-------------------------------------------------------------------------------------------------------------------------------


str(products)
summary(products)
kable(head(products, 20))

#-------------------------------------------------------------------------------------------------------------------------------
#             EDA Order Products Train Data
#-------------------------------------------------------------------------------------------------------------------------------


kable(head(order_products,20))

#-------------------------------------------------------------------------------------------------------------------------------
#             EDA ORder Proudcts Prior Data
#-------------------------------------------------------------------------------------------------------------------------------














