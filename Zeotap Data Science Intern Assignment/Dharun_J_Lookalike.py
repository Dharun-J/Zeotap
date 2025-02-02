#importing required headers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

#reading the data from the given csv files using pandas read
cust=pd.read_csv("Customers.csv")
trans=pd.read_csv("Transactions.csv")

#merging customer and transactions data
cust_trans=pd.merge(cust,trans,on='CustomerID',how='inner')

#grouping transactions of each customer
cust_profile=cust_trans.groupby('CustomerID').agg({'TotalValue':'sum','TransactionID':'count','Quantity':'sum'}).reset_index()

#merging the customer profile created with customer data
cust_profile=pd.merge(cust_profile, cust, on='CustomerID', how='inner')

sca=MinMaxScaler()
f=cust_profile[['TotalValue', 'TransactionID', 'Quantity']]
no_f=sca.fit_transform(f)

#computing the similarity matrix using the above
s_matrix=cosine_similarity(no_f)
s_df=pd.DataFrame(s_matrix, index=cust_profile['CustomerID'], columns=cust_profile['CustomerID'])

f_cust=cust_profile[cust_profile["CustomerID"].str.startswith("C")].head(20)["CustomerID"]

l_like={}
for customer in f_cust:
    similar_customers=s_df[customer].sort_values(ascending=False).iloc[1:4]
    l_like[customer]=list(similar_customers.items())

#saving the results into the Dharun_J_Lookalike.csv
l_res=[]
for customer, sim in l_like.items():
        l_res.append({'CustomerID': customer, 'Lookalikes': sim})

with open("Dharun_J_Lookalike.csv", "w") as file:
    file.write("cust_id,lookalikes\n")
    for entry in l_res:
        lookalikes_str=";".join([f"({similar[0]},{similar[1]:.4f})" for similar in entry["Lookalikes"]])
        file.write(f"{entry['CustomerID']},{lookalikes_str}\n")
print('Completed')