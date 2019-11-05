
# rm(list = ls())

library(ggplot2)
library(dplyr)
library(data.table)

#==========================================================================================
## Reading the data & Quick Analysis
#==========================================================================================
rm(order_prior)
datapath <- "C:/Users/shubh/Google Drive/Study/PGDBA/IIMC Shared Drive/Datasets/Kaggle/InstaCart/"

insta_aisles <- fread(paste0(datapath,"aisles.csv"))
insta_aisles

insta_dept <- fread(paste0(datapath,"departments.csv"))
insta_dept

insta_prod <- fread(paste0(datapath,"products.csv"))
insta_prod
insta_prod <- insta_prod %>% left_join(insta_aisles, by = "aisle_id") %>%
  left_join(insta_dept, by = "department_id")
str(insta_prod)
length(unique(insta_prod$product_name))

insta_prior <- fread(paste0(datapath,"order_products__prior.csv"))
str(insta_prior)
#insta_prior <- insta_ordrProd_prior %>% left_join(insta_prod, by = "product_id") %>%
#  mutate(product_name = as.factor(product_name),
#         aisle = as.factor(aisle),
#         department = as.factor(department))
#setDT(insta_prior)
#str(insta_prior)
head(insta_prior,10)
length(unique(insta_prior$product_id))
summary(insta_prior)


insta_train <- fread(paste0(datapath,"order_products__train.csv"))
str(insta_train)
#insta_train <- insta_ordrProd_train %>% left_join(insta_prod, by = "product_id") %>%
#  mutate(product_name = as.factor(product_name),
#         aisle = as.factor(aisle),
#         department = as.factor(department))
#str(insta_train)
head(insta_train,10)
length(unique(insta_train$product_id))
summary(insta_train)


insta_orders <- fread(paste0(datapath,"orders.csv"))
head(insta_orders,10)
summary(insta_orders)
str(insta_orders)
length(unique(insta_orders$user_id))

priorSet <- insta_orders %>% filter(eval_set == "prior") %>% 
  left_join(insta_prior, by = 'order_id') %>%
  left_join(insta_prod, by = 'product_id')
head(priorSet)

trainSet <- insta_orders %>% filter(eval_set == "train") %>% 
  left_join(insta_train, by = 'order_id') %>%
  left_join(insta_prod, by = 'product_id')
head(trainSet)

# fwrite(trainSet, "InstaTrain.csv", nThread = 7)
# fwrite(priorSet, "InstaPrior.csv", nThread = 7)

rm(insta_train)
rm(insta_prior)
rm(insta_orders)
gc()

####################################################################################################

library(arules)
library(arulesViz)

### Apriori Rules on training set
ar_train <- split(trainSet$product_name, trainSet$order_id)
head(ar_train)

ar_train <- as(ar_train, 'transactions')
summary(ar_train)
itemFrequencyPlot(ar_train, topN = 25)

rules_train <- apriori(ar_train, parameter = list(sup = 0.01, conf = 0.1, target = "rules"))
rules_train <- sort(rules_train, by = 'support', decreasing = TRUE)
inspect(rules_train)

rules1 <- apriori(ar_train, parameter = list(supp = 0.00001, conf = 0.6, maxlen=3),
                  control = list(verbose = FALSE)) 

summary(quality(rules1))
plot(rules1)

inspect(sort(rules1, by="lift")[1:10])

inspect(sort(rules1, by="confidence")[1:10])

## Increase the support and decrease confidence to get rules of some 
## more frequent items but with less confidence
rules2 <- apriori(ar_train, parameter = list(supp = 0.001, conf = 0.4, maxlen=3), 
                  control = list(verbose = FALSE))

summary(quality(rules2))
plot(rules2)

inspect(sort(rules2, by="lift")[1:10])

sinspect(sort(rules2, by="confidence")[1:10])

##
rules3 <- apriori(ar_train, parameter = list(supp = 0.005, conf = 0.1, maxlen=3),
                  control = list(verbose = FALSE))
summary(quality(rules3))
plot(rules3)

inspect(sort(rules3, by="lift")[1:10])
inspect(sort(rules3, by="confidence")[1:10])


#################
### Ariori Rules on Prior Set
ar_prior <- split(priorSet$product_name, priorSet$order_id)
head(ar_prior)

ar_prior <- as(ar_prior, 'transactions')
summary(ar_prior)
itemFrequencyPlot(ar_prior, topN = 25)
rules_prior <- apriori(ar_prior, parameter = list(sup = 0.01, conf = 0.1, target = "rules"))
rules_prior <- sort(rules_prior, by = 'confidence', decreasing = TRUE)
inspect(rules_prior)
##################