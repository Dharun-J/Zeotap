#importing required headers
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as pl

#reading the data from the given csv files using pandas read
cust=pd.read_csv("Customers.csv")
pro=pd.read_csv("Products.csv")
trans=pd.read_csv("Transactions.csv")

#checking structure of the datasets given
print(cust.info())
print(pro.info())
print(trans.info())

#doing statistics on the data
print(cust.describe())
print(pro.describe())
print(trans.describe())

#checking for the missing values in the data for data integrity
print(cust.isnull().sum())
print(pro.isnull().sum())
print(trans.isnull().sum())

#customers data analysis
sn.countplot(y='Region',data=cust)
pl.title("Distribution")
pl.show()

#product data analysis
categ=pro['Category'].value_counts()
categ.plot(kind="bar",title="Categories")
pl.show()

#transactions data analysis
trans['TransactionDate'] = pd.to_datetime(trans['TransactionDate'])
trans['Month'] = trans['TransactionDate'].dt.to_period('M')
trans_month = trans.groupby('Month').size()
pl.figure(figsize=(10, 6))
trans_month.plot(kind='bar')
pl.title('Transactions by Month')
pl.xlabel('Month')
pl.ylabel('Transactions')
pl.xticks(rotation=45)
pl.show()