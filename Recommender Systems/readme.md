## Product Recommender System  
Goal of predicting and compiling a list of items that a customer likely to purchase
<br>Eg:Music streaming service, Pandora,utilizes recommender systems for music recommendations for their listeners
E-Commerce Company Amazon,utilizes recommender systems to recommend er systems to predict and show a list of products that a customer likely to purchase

Two ways to produce list of recommendations:
* **Collaborative Filtering**
* **Content-based Filtering**

**Collaborative Filtering :** It is based on previous user behaviours,such as pages that they viewed,products that they purchased or ratings that they have given to different items.
The collaborative Filtering approach uses data to find similarities between users or items,and recommend the most similar items or contents to the users.The basic assumption behind collaborative filtering method is that those who have viewed or purchased similar products or content in the past are likely to purchase similar kind of contents or products in the future.

**Content Based Filtering:** It produces a list of recommendations based on the characterstics of an item or user.
The basic assumption behind the content based filtering method is that the users likely to view or purchase items that are similar to those items that they have bought or viewed in the past

**Collaborative Filtering:**<br>
There are two approaches<br> 
* **User based approach** (calculate similarites between users)<br> 
* **Item based approach**(calculate similarities between the two items)

**User Based Approach:**<br>
1.Build a user-item matrix (a user-item matrix comprises individual users in rows and individual items in columns) <br>
2.Measure the similarities between users (To measure the similarities cosine similarity is frequently used)<br>

**Item Based Approach:**<br>
1.Build a item-user matrix ( in simple terms it is a transpose of user-item matrix)<br>
2.Measure the similarities between the items<br>
