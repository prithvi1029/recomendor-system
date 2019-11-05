library(dplyr)
library(arules)
library(tibble)

aisles = read.csv('aisles.csv')
orders = read.csv('orders.csv')
products = read.csv('products.csv')
order_prod_prior = read.csv('order_products_prior.csv')
order_prod_train = read.csv('order_products_train.csv')

order = as_tibble(orders)
order_prior = as_tibble(order_prod_prior)

# Preliminary Viewing
View(orders[1:20,])
View(order_products[1:20,])
View(order_prod_prior[1:20,])

# aisles - aisle id , aisle
# departments - dept id, dept
# products - product id, prod name, dept id, aisle id
# order_products (train)- Contains the specific details of each order
# order_products_prior - Contains the specific details of each order - order id, prod id, add to cart order, reoder
# orders - Contains the overall view of each order - orderid, userid, evalset, orderno., dow, hourod, day_sincelast


order_basket = order_prior %>% 
                                inner_join(products, by = 'product_id') %>% 
                                group_by(order_id) %>%
                                summarise(basket = as.vector(list(product_name)))
View(order_basket[1:20,])
gc()
order_basket$basket = as.factor(order_basket$basket)

transactions = as(order_basket, 'transactions')
