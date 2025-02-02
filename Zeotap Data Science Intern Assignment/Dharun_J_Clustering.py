#importing the required headers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#reading the data from the given csv files using pandas read
cust=pd.read_csv("Customers.csv")
trans=pd.read_csv("Transactions.csv")

#merging customer and transactions data
cust_trans = pd.merge(cust,trans,on='CustomerID',how='inner')

#grouping transactions of each customer
cust_profile = cust_trans.groupby('CustomerID').agg({'TotalValue':'sum','TransactionID':'count','Quantity':'sum'}).reset_index()

sca=MinMaxScaler()
f=cust_profile[['TotalValue', 'TransactionID', 'Quantity']]
no_f=sca.fit_transform(f)

#using DB method for clusters
db_score=[]
for k in range(2, 11):
    kmeans=KMeans(n_clusters=k, random_state=42)
    clusters=kmeans.fit_predict(no_f)
    db_index=davies_bouldin_score(no_f,clusters)
    db_score.append(db_index)

opt=db_score.index(min(db_score))+2
print(f"Optimal number of clusters:{opt}")
print(f"Davies-Bouldin Index:{min(db_score):.4f}")

#clustering with optimal
kmeans = KMeans(n_clusters=opt,random_state=42)
clusters = kmeans.fit_predict(no_f)
cust_profile['Cluster'] = clusters

#plotting
pca=PCA(n_components=2)
reduced_features=pca.fit_transform(no_f)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=reduced_features[:, 0],y=reduced_features[:, 1],hue=clusters,palette="viridis")
plt.title("Segments")
plt.show()