#import the reqired libraries
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import joblib 
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings; warnings.simplefilter('ignore')

#Importing Surprise and relevant packages to do some hyper parameter tuning through Grid Search
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import SVD, SVDpp, NMF
from surprise import SlopeOne, CoClustering
# %matplotlib inline

# Import the dataset and give the column names
columns=['userId', 'productId', 'ratings','timestamp']
electronics_df=pd.read_csv('Electronics_data.csv',names=columns)

electronics_df.head()

electronics_df.shape

electronics_df.describe()

electronics_df.info()

print('Number of missing values across columns: \n',electronics_df.isnull().sum())


#Check the number of rows and columns
rows,columns=electronics_df.shape
print('Number of rows: ',rows)
print('Number of columns: ',columns)

electronics_df.hist(figsize=(20, 15))
plt.show()

electronics_df.drop('timestamp',axis=1,inplace=True)

#Number of products is less than number of users, so item-item colaborative filtering would make sense
#instead of user-user colaborative filtering
print("Electronic Data Summary")
print("="*100)
print("\nTotal # of Ratings :",electronics_df.shape[0])
print("Total # of Users   :", len(np.unique(electronics_df.userId)))
print("Total # of Products  :", len(np.unique(electronics_df.productId)))
print("\n")
print("="*100)

electronics_df.shape

#Taking subset of the dataset
electronics_df1=electronics_df.iloc[:1824482,0:]
# electronics_df1=electronics_df.iloc[:424482,0:]

electronics_df1.info()

#Summary statistics of rating variable
electronics_df1['ratings'].describe().transpose()

electronics_df1['ratings'] = pd.to_numeric(electronics_df1['ratings'], errors='coerce')

# Find the minimum and maximum ratings
print('Minimum rating is: %d' % (electronics_df1['ratings'].min()))
print('Maximum rating is: %d' % (electronics_df1['ratings'].max()))


# Check the distribution of the rating
with sns.axes_style('white'):
    g = sns.factorplot("ratings", data=electronics_df1, aspect=2.0,kind='count')
    g.set_ylabels("Total number of ratings") 

sns.boxplot(y='ratings',data=electronics_df1)

#Check the top 10 users based on ratings
electronics_df1.drop_duplicates(inplace=True)
most_rated=electronics_df1.groupby('userId').size().sort_values(ascending=False)
print('Top 10 users based on ratings: \n',most_rated[:10])

counts=electronics_df1.userId.value_counts()
electronics_df1_final=electronics_df1[electronics_df1.userId.isin(counts[counts>=10].index)]
print('Number of users who have rated 10 or more items =', len(electronics_df1_final))
print('Number of unique users in the final data = ', electronics_df1_final['userId'].nunique())
print('Number of unique products in the final data = ', electronics_df1_final['userId'].nunique())

#statistical analysis of the most rated user data
most_rated.describe().astype(int).T

#Boxplot shows that we have few users who rate many items (appearing in outliers) but majority rate very few items

sns.boxplot(data=most_rated);

#Let us look at the quantile view to understand where the ratings are concentrated
quantiles = most_rated.quantile(np.arange(0,1.01,0.01), interpolation='higher')
quantiles

plt.figure(figsize=(10,10))
plt.title("Quantiles and their Values")
quantiles.plot()
# quantiles with 0.05 difference
plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='red', label="quantiles with 0.05 intervals")
# quantiles with 0.25 difference
plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='green', label = "quantiles with 0.25 intervals")
plt.ylabel('# ratings by user')
plt.xlabel('Value at the quantile')
plt.legend(loc='best')
plt.show()

#Products also have skewed ratings with majority of the products having very few ratings
no_of_ratings_per_product = electronics_df1_final.groupby(by='productId')['ratings'].count().sort_values(ascending=False)

fig = plt.figure(figsize=plt.figaspect(.5))
ax = plt.gca()
plt.plot(no_of_ratings_per_product.values)
plt.title('# RATINGS per Product')
plt.xlabel('Product')
plt.ylabel('# ratings per product')
ax.set_xticklabels([])

plt.show

ratings_mean_count = pd.DataFrame(electronics_df1_final.groupby('productId')['ratings'].mean())
ratings_mean_count['rating_counts'] = pd.DataFrame(electronics_df1_final.groupby('productId')['ratings'].count())
ratings_mean_count.head()




#The maximum number of ratings received for a product is 242

ratings_mean_count['rating_counts'].max()

#### Majority of the products have received 1 rating only and it is a right skewed distribution
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating_counts'].hist(bins=50)

#We see a left skewed distribution for the ratings
#There are clusters at each of the points 1,2,3,4,5 as that is where the means are concentrated
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['ratings'].hist(bins=20)

#From the joint plot below it seems that popular products (higher ratings) tend to be rated more frequently, majority have rated products in the higher range
#To make people more engaged (bottom of the chart) I can start by recommending them based on popularity based system and then
#slowly graduate them to collaborative system once I have sufficient number of data points to giver personlized recommendation
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='ratings', y='rating_counts', data=ratings_mean_count, alpha=0.4)

popular_products = pd.DataFrame(electronics_df1_final.groupby('productId')['ratings'].count())
most_popular = popular_products.sort_values('ratings', ascending=False)
most_popular.head(30).plot(kind = "bar")

no_of_ratings_per_user = electronics_df1_final.groupby(by='userId')['ratings'].count().sort_values(ascending=False)

#Split the data randomnly into train and test datasets into 70:30 ratio
train_data, test_data = train_test_split(electronics_df1_final, test_size = 0.3, random_state=42)
train_data.head()

print('Shape of training data: ',train_data.shape)
print('Shape of testing data: ',test_data.shape)
data=train_data

prod_ge_25=electronics_df.groupby("productId").filter(lambda x:x['ratings'].count() >= 25)
#Subsetting the data to keep users who have given at least 25 ratings
user_ge_25=electronics_df.groupby("userId").filter(lambda x:x['ratings'].count() >= 25)
user_ge_25.drop(['ratings'],inplace=True,axis=1)
user_prod_ge_25 = pd.merge(prod_ge_25,user_ge_25)
user_prod_ge_25.shape
new_df2 = user_prod_ge_25.sample(frac=0.25, replace=False, random_state=0)

# Set Rating Scale from 1 to 5
#We are running basic algorithms to check which one works best
reader = Reader(rating_scale=(1, 5))

# Load data with rating scale
#data = Dataset.load_from_df(new_df, reader)
data = Dataset.load_from_df(new_df2,reader)

# Split the data into 70% / 30%
# trainset, testset = train_test_split(data, test_size=.30)
# trainset = dataset_svd.build_full_trainset()
# testset = trainset.build_anti_testset()
raw_ratings = data.raw_ratings                         # 90% trainset, 10% testset                                                
threshold = int(.9 * len(raw_ratings))                                     
trainset_raw_ratings = raw_ratings[:threshold]                             
test_raw_ratings = raw_ratings[threshold:]             
data.raw_ratings = trainset_raw_ratings        


from surprise import accuracy                                              

# Parameter space
svd_param_grid = {'n_epochs': [20, 25], 
                  'lr_all': [0.007, 0.009, 0.01],
                  'reg_all': [0.4, 0.6]}

import joblib
# Load the SVDpp model
algo_svdpp = joblib.load('svdpp_model.pkl')

# Load the SVD model
algo_svd = joblib.load('svd_model.pkl')
# Grid search for SVDpp
svdpp_gs = GridSearchCV(SVDpp, svd_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
svdpp_gs.fit(data)
algo_svdpp = svdpp_gs.best_estimator['rmse']

# Save the best SVDpp model
joblib.dump(algo_svdpp, 'svdpp_model.pkl')

# Retrain SVDpp on the whole train set
trainset = data.build_full_trainset()
algo_svdpp.fit(trainset)

# Now test on the trainset
testset = trainset.build_testset()
predictions_train = algo_svdpp.test(testset)
print('Accuracy on the trainset:')
accuracy.rmse(predictions_train)

# Now test on the testset
testset = data.construct_testset(test_raw_ratings)
pred_svdpp = algo_svdpp.test(testset)
print('Accuracy on the testset:')
accuracy.rmse(pred_svdpp)

# Grid search for SVD
svd_gs = GridSearchCV(SVD, svd_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
svd_gs.fit(data)
algo_svd = svd_gs.best_estimator['rmse']

# Save the best SVD model
joblib.dump(algo_svd, 'svd_model.pkl')

# Retrain SVD on the whole train set
trainset = data.build_full_trainset()
algo_svd.fit(trainset)

# Now test on the trainset
testset = trainset.build_testset()
predictions_train = algo_svd.test(testset)
print('Accuracy on the trainset:')
accuracy.rmse(predictions_train)


# now test on the testset                                                  
testset = data.construct_testset(test_raw_ratings)                         
pred_svd=algo_svd.test(testset)
print('Accuracy on the testset:')                                          
accuracy.rmse(pred_svd)  




print('SVDpp - RMSE:', round(svdpp_gs.best_score['rmse'], 4), '; MAE:', round(svdpp_gs.best_score['mae'], 4))
print('SVD   - RMSE:', round(svd_gs.best_score['rmse'], 4), '; MAE:', round(svd_gs.best_score['mae'], 4))
print('RMSE =', svdpp_gs.best_params['rmse'])
print('MAE =', svdpp_gs.best_params['mae'])
print('RMSE =', svd_gs.best_params['rmse'])
print('MAE =', svd_gs.best_params['mae'])

from collections import defaultdict
def get_top_n_recommendations(reccomemndations, n=5):
    # First map the reccommendations to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in reccomemndations:
        top_n[uid].append((iid, est))

    #sort predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

top_5 = get_top_n_recommendations(pred_svd, n=5)
for uid, user_ratings in top_5.items():
    print(uid, [iid for (iid, _) in user_ratings])

