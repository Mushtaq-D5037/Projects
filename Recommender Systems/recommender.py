# HYPOTHESIS:
# unit price  should be > 0
# quantity should be    > 0
# invoice date should   < today

# read csv file
df_retail = pd.read_csv('Online Retail.csv',encoding = "ISO-8859-1")

# some basic checks
df_retail.columns
df_retail.info()

# box plot
df_retail['Quantity'].plot.box(showfliers = False)
df_retail['UnitPrice'].plot.box(showfliers = False)

# dropping Quantity less than zero
df_retail = df_retail[df_retail['Quantity']>0]

# missing values
df_retail.isnull().sum()
# observation
# customerId has more number of missing values
# a look at some of the missing customerID info
df_retail[df_retail['CustomerID'].isna()].head(10)
# dropping CustomerID 
df_retail.dropna(subset=['CustomerID'],inplace = True)

# Collaborative Filtering
# Building Customer Item Matrix
customer_item_matrix = df_retail.pivot_table(index='CustomerID',columns = 'StockCode',values='Quantity',aggfunc='sum')
# encoding 0-1; 0--means not purchased, 1--> means purchased
customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x>0 else 0)

from sklearn.metrics.pairwise import cosine_similarity
#
# user based collaborative Filtering
#
user_user_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix))
user_user_sim_matrix.head()

# adding columns
user_user_sim_matrix.columns = customer_item_matrix.index
user_user_sim_matrix['CustomerID'] = customer_item_matrix.index
user_user_sim_matrix = user_user_sim_matrix.set_index('CustomerID')

# checking similarity between the customers
user_user_sim_matrix.loc[12350.0].sort_values(ascending = False)
# observation
# customer to themselves similarity is 1
# these are most similar customers to 12350

# recommending a product to customerID 17935 based on CustomerID 12350 similarity
# step 1:identifying the products that both the customers have purchased
# step 2:finding the product that 17935 has not purchased but 12350 has purchased
item_bought_by_12350 = set(customer_item_matrix.loc[12350.0].iloc[customer_item_matrix.loc[12350.0].nonzero()].index)
item_bought_by_17935 = set(customer_item_matrix.loc[17935.0].iloc[customer_item_matrix.loc[17935.0].nonzero()].index)

item_to_recommend = item_bought_by_12350 - item_bought_by_17935
# getting description of these Items
item_to_recommend = df_retail.loc[df_retail['StockCode'].isin(item_to_recommend),['StockCode','Description']].drop_duplicates().set_index('StockCode')

#
# Item-Base Collaborative Filtering
#
item_item_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix.T))
item_item_sim_matrix.head()
# adding column names
item_item_sim_matrix.columns = customer_item_matrix.T.index
item_item_sim_matrix['StockCode']= customer_item_matrix.T.index
item_item_sim_matrix = item_item_sim_matrix.set_index('StockCode')

# top 10 similar Items
top_10_similar_items = list(item_item_sim_matrix.loc['23166']
                                                 .sort_values(ascending=False)
                                                 .iloc[:10]
                                                 .index )

# recommending products using item-item collaborative filtering
df_retail.loc[ df_retail['StockCode'].isin(top_10_similar_items),['StockCode', 'Description']].drop_duplicates().set_index('StockCode').loc[top_10_similar_items]
