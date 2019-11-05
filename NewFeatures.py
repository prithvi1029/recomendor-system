# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 01:56:07 2017

@author: Korah
"""

import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

## Loading Data --------------------------------------------------------------------------
# path_data = os.getcwd() + '\\Data\\'
path_data = 'C:\\Users\\Korah\\Documents\\Data Science\\PGDBA\\CDS\\Project\\Data\\'
priors = pd.read_csv(path_data + 'order_products_prior.csv', 
                     dtype={'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})

train = pd.read_csv(path_data + 'order_products_train.csv', 
                    dtype={'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})

orders = pd.read_csv(path_data + 'orders.csv', 
                     dtype={'order_id': np.int32,
                            'user_id': np.int64,
                            'eval_set': 'category',
                            'order_number': np.int16,
                            'order_dow': np.int8,
                            'order_hour_of_day': np.int8,
                            'days_since_prior_order': np.float32})

products = pd.read_csv(path_data + 'products.csv')
aisles = pd.read_csv(path_data + "aisles.csv")
departments = pd.read_csv(path_data + "departments.csv")
sample_submission = pd.read_csv(path_data + "sample_submission.csv")

products = products.merge(aisles, 'left', 'aisle_id').merge(departments, 'left', 'department_id')
#================================================================================================
## Data Manipulation ----------------------------------------------------------------------------
#================================================================================================

### Adding Custom variables
## Adding times Zone
orders.sort_values('order_hour_of_day', inplace=True)
orders.drop_duplicates(inplace=True)
orders.reset_index(drop=True, inplace=True)

def timezone(s):
    if s < 6:
        return 'midnight'
    elif s < 12:
        return 'morning'
    elif s < 18:
        return 'noon'
    else:
        return 'night'

orders['timezone'] = orders.order_hour_of_day.map(timezone)
orders.head()

## Adding type of items
products['item_is_Organic'] = products.product_name.map(lambda x: 'organic' in x.lower())*1
products['item_is_Gluten-Free'] = products.product_name.map(lambda x: 'gluten' in x.lower() and 'free' in x.lower())*1
products['item_is_Asian'] = products.product_name.map(lambda x: 'asian' in x.lower())*1

combiSet = pd.concat([priors, train], ignore_index = 1)
combiSet.sort_values(['order_id', 'add_to_cart_order'], inplace=True)
combiSet.reset_index(drop=1, inplace=True)
combiSet = pd.merge(combiSet, products, on = 'product_id', how = 'inner')
combiSet = pd.merge(combiSet, orders, on = 'order_id', how = 'inner')
combiSet['order_number_rev'] = combiSet.groupby('user_id').order_number.transform(np.max) - combiSet.order_number

del priors, train, orders, aisles, departments, products
gc.collect()
combiSet.to_pickle('prior-train_combi.p')
# combiSet = pd.read_pickle('prior-train_combi.p')

## Expanding aisles and department to create new features
order_department = pd.crosstab(combiSet['order_id'], 
                               combiSet['department_id']).add_prefix('department_').reset_index()
order_department.shape
order_department.head()

order_aisle = pd.crosstab(combiSet['order_id'], 
                          combiSet['aisle_id']).add_prefix('aisle_').reset_index()
order_aisle.shape

## Items Bought in a row
col = ['order_id', 'user_id', 'product_id', 'order_number', 'reordered']
log = combiSet[col]
log.sort_values(['user_id', 'product_id', 'order_number'], inplace=True)

uid_bk = pid_bk = onum_bk = None
ret = []
miniters = int(log.shape[0]/50)
col = ['user_id', 'product_id', 'order_number']
for uid,pid,onum in tqdm(log[col].values, miniters = miniters):
    if uid_bk is None:
        cnt = 1
        ret.append(cnt)
    elif uid == uid_bk and pid == pid_bk:
        if onum - onum_bk == 1:
            cnt+=1
            ret.append(cnt)
        else:
            cnt = 1
            ret.append(cnt)
        pass
    elif uid == uid_bk and pid != pid_bk: # item change
        cnt = 1
        ret.append(cnt)
    elif uid != uid_bk: # user change
        cnt = 1
        ret.append(cnt)
    else:
        raise Exception('?')

    uid_bk = uid
    pid_bk = pid
    onum_bk = onum

combiSet['buy_item_inarow'] = ret
del log

## Days since prior order for a particular product
# TO check

## None
## Order_tbl from pickle
order_tbl = pd.read_pickle('input/mk/order_tbl.p')

# No Need to Run. Result is stored
col = ['order_id', 'user_id','order_number','product_name', 'eval_set']
order_tbl = pd.read_pickle('input/mk/order_tbl.p')[col]
order_tbl.sort_values(['user_id', 'order_number'], inplace=True)
order_tbl = order_tbl[order_tbl.eval_set!='test']

uid_bk = None
product_name_all = [] # 2d list
pname_unq = []        # 1d list
pname_unq_len = []     # 1d list
for uid,pnames in tqdm(order_tbl[['user_id', 'product_name']].values):
    if uid_bk is None:
        pname_unq += pnames
    elif uid == uid_bk:
        pname_unq += pnames
    elif uid != uid_bk:
        pname_unq = pnames[:]
        
    uid_bk = uid
    pname_unq = list(set(pname_unq))
    pname_unq_len.append(len(pname_unq))
    product_name_all.append(pname_unq)

order_tbl['product_name_all'] = product_name_all
order_tbl['product_unq_len'] = pname_unq_len
order_tbl['new_item_cnt'] = order_tbl.groupby('user_id').product_unq_len.diff()
order_tbl['product_len'] = order_tbl['product_name'].map(len)
order_tbl['is_None'] = (order_tbl.new_item_cnt == order_tbl.product_len)*1

col = ['order_id', 'product_unq_len', 'is_None']
order_tbl[col].to_pickle('input/mk/order_None.p')

order_None = pd.read_pickle('input/mk/order_None.p')

## Streak
# To be Done

## Replacements
replacement = pd.read_pickle('input/mk/replacement.p')

## aisle_dep_cumsum
col = ['user_id', 'order_number', 'order_id']
log = utils.read_pickles('input/mk/log', col).drop_duplicates().sort_values(col)

ai_dep = pd.read_pickle('input/mk/order_aisle-department.p')

log = pd.merge(log, ai_dep, on='order_id', how='left')

#==============================================================================
# calc
#==============================================================================
col = [c for c in log.columns if 'aisle_' in c or 'dep' in c]
di = defaultdict(int)
uid_bk = None

li1 = []
for args in tqdm(log[['user_id']+col].values):
    uid = args[0]
    
    if uid_bk is None:
        pass
    elif uid == uid_bk:
        pass
    elif uid != uid_bk:
        di = defaultdict(int)
    li2 = []
    for i,c in enumerate(col):
        di[c] += args[i+1]
        li2.append(di[c])
    li1.append(li2)
    
    uid_bk = uid
#==============================================================================
df = pd.DataFrame(li1, columns=col).add_suffix('_cumsum')
df['order_id'] = log['order_id']

df.to_pickle('input/mk/order_aisle-department_cumsum.p')
df.to_csv('input/mk/order_aisle-department_cumsum.csv')

order_ad_cumsum = pd.read_csv('input/mk/order_aisle-department_cumsum.csv')


## Repeat previous

#==============================================================================
# reordered_ratio
#==============================================================================
col = ['order_id', 'user_id', 'reordered', 'order_number']
temp = combiSet[col]
reordered_ratio = temp.groupby(['order_id']).reordered.mean().reset_index()
reordered_ratio.columns = ['order_id', 'reordered_ratio']
temp = pd.merge(temp, reordered_ratio, on='order_id', how='left')

temp['unreordered'] = 1-temp.reordered
unreordered_ratio = temp.groupby(['order_id']).unreordered.mean().reset_index()
unreordered_ratio.columns = ['order_id', 'unreordered_ratio']
temp = pd.merge(temp, unreordered_ratio, on='order_id', how='left')

temp.head()
del reordered_ratio, unreordered_ratio; gc.collect()

#==============================================================================
# total_unique_item
#==============================================================================

order_unique_item = temp.groupby('order_id').unreordered.sum().reset_index()
order_unique_item.columns = ['order_id', 'unreordered_sum']
temp = pd.merge(temp, order_unique_item, on='order_id', how='left')

temp['total_unique_item'] = temp.groupby('user_id').unreordered_sum.cumsum()
temp['total_unique_item_ratio'] = temp['total_unique_item']/temp['order_number']

del order_unique_item; gc.collect()

#==============================================================================
# ordered item
#==============================================================================

ordered_item = temp.groupby('order_id').size().reset_index()
ordered_item.columns = ['order_id', 'ordered_item']

temp = pd.merge(temp, ordered_item, on='order_id', how='left')

temp['total_ordered_item'] = temp.groupby('user_id').ordered_item.cumsum()
temp['total_ordered_item_ratio'] = temp['total_ordered_item']/temp['order_number']

del ordered_item; gc.collect()
temp.head()
order_feat = temp.copy()
order_feat = order_feat.drop(['reordered', 'order_number', 'unreordered'], axis = 1)
del temp ; gc.collect()

## Average orderspan
users = combiSet[combiSet.order_number_rev > 0].groupby('user_id')['days_since_prior_order'].mean().reset_index()
users.columns = ['user_id', 'days_order_mean']

## Type of User
temp = combiSet[combiSet.order_number_rev > 0]
user = temp.groupby(['user_id']).size().reset_index()
user.columns = ['user_id', 'total']
user['organic_cnt'] = temp.groupby(['user_id'])['item_is_Organic'].sum()
user['glutenfree_cnt'] = temp.groupby(['user_id'])['item_is_Gluten-Free'].sum()
user['Asian_cnt'] = temp.groupby(['user_id'])['item_is_Asian'].sum()
    
user['organic_ratio'] = user['organic_cnt'] / user.total
user['glutenfree_ratio'] = user['glutenfree_cnt'] / user.total
user['Asian_ratio'] = user['Asian_cnt'] / user.total

user.drop('total', axis=1, inplace=True)

users = users.merge(user, how = 'left', on = 'user_id')
del user, temp

## Order Size
temp = combiSet.groupby('order_id').size().reset_index()
temp.columns = ['order_id', 'order_size']

temp = pd.merge(temp, combiSet[['order_id', 'user_id']].drop_duplicates())

user_osz = temp.groupby(['user_id']).order_size.min().reset_index()
user_osz.columns = ['user_id', 'user_order_size-min']
user_osz['user_order_size-max'] = temp.groupby(['user_id']).order_size.max()
user_osz['user_order_size-median'] = temp.groupby(['user_id']).order_size.median()
user_osz['user_order_size-mean'] = temp.groupby(['user_id']).order_size.mean()
user_osz['user_order_size-std'] = temp.groupby(['user_id']).order_size.std()

users = users.merge(user_osz, how = 'left', on = 'user_id')
del user_osz, temp
users = users.fillna(0)
combiSet.to_pickle('Combi-data.p')
combiSet.head()
combiSet = combiSet.merge(users, how = 'left', on = 'user_id')
combiSet = combiSet.merge(order_feat, how = 'left', on = ['order_id','user_id'])
combiSet = combiSet.drop(['product_name','aisle','department'], axis = 1)
combiSet.to_pickle('prior-train_combi.p')
# order_feat.to_pickle('order_feat.p')
## Have you bought this product:
"""
pid      freq
-------------
24852    57186
13176    47063
21137    39871
21903    38095
47209    30047
47626    28741
47766    28478
26209    26199
16797    25621
24964    21090
22935    20824
27966    20193
39275    20134
45007    19652
49683    17508
4605     16176
27845    16134
40706    16054
5876     15765
4920     15150
28204    14802
42265    14766
30391    14089
31717    13949
8277     13900
8518     13770
27104    13719
17794    13642
46979    13491
45066    13289

"""

col = [ 'order_id', 'user_id', 'product_id', 'order_number', 'reordered', 'order_number_rev']
temp = combiSet[col]

user = temp.drop_duplicates('user_id')[['user_id']].reset_index(drop=True)
    
# have you bought -> hyb
tag_user = temp[temp.product_id==24852].user_id
user['hyb_Banana'] = 0
user.loc[user.user_id.isin(tag_user), 'hyb_Banana'] = 1
    
tag_user = temp[temp.product_id==13176].user_id
user['hyb_BoO-Bananas'] = 0
user.loc[user.user_id.isin(tag_user), 'hyb_BoO-Bananas'] = 1
    
tag_user = temp[temp.product_id==21137].user_id
user['hyb_Organic-Strawberries'] = 0
user.loc[user.user_id.isin(tag_user), 'hyb_Organic-Strawberries'] = 1
    
tag_user = temp[temp.product_id==21903].user_id
user['hyb_Organic-Baby-Spinach'] = 0
user.loc[user.user_id.isin(tag_user), 'hyb_Organic-Baby-Spinach'] = 1
    
tag_user = temp[temp.product_id==47209].user_id
user['hyb_Organic-Hass-Avocado'] = 0
user.loc[user.user_id.isin(tag_user), 'hyb_Organic-Hass-Avocado'] = 1

# combiSet = combiSet.merge(user, how = 'left', on = 'user_id')    
users = users.merge(user, how = 'left', on = 'user_id')
del user, temp ;  gc.collect()

## None
LOOP = 20
temp = order_tbl[['order_id', 'user_id', 'order_number']].sort_values(['user_id', 'order_number', 'order_id'])
for i in range(1, LOOP):
    temp['t-{}_order_id'.format(i)] = temp.groupby('user_id')['order_id'].shift(i)

col = [c for c in temp.columns if 'order_id' in c]
temp = temp[col]

order_None = pd.read_pickle('../input/mk/order_None.p')

df = temp.copy()

for i in tqdm(range(1, LOOP)):
    df = pd.merge(df, order_None.add_prefix('t-{}_'.format(i)), 
                on='t-{}_order_id'.format(i), how='left')
    
col = [c for c in df.columns if c.endswith('_order_id')]
df.drop(col, axis=1, inplace=True)
df.head()
df.fillna(-1, inplace=True)

order_None = df.copy()
df.to_pickle('None_order.p')

del temp, LOOP, col, df
# df.to_pickle('feature/trainT-0/f110_order.p')
# df.to_pickle('feature/test/f110_order.p')


##=============================================================================================================
## Item Features ----------------------------------------------------------------------------------------------
combiSet.info()
train = combiSet[combiSet.eval_set == 'train']
train.to_csv('trainData.csv')

