import pandas as pd

data = pd.read_csv('Dataset/Train.csv')
data.set_index('Inv_Id',inplace=True)

data.Vendor_Code = data.Vendor_Code.astype("category",ordered=True,categories=data.Vendor_Code.unique()).cat.codes
#data.Product_Category = data.Product_Category.astype("category",ordered=True,categories=data.Product_Category.unique()).cat.codes
y = pd.get_dummies(data=data.Product_Category,columns=['Product_Category'])
product_cats = y.columns
print(product_cats)