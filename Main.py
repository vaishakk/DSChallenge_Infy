import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras import layers

data = pd.read_csv('Dataset/Train.csv')
data.set_index('Inv_Id',inplace=True)

#data.Vendor_Code = data.Vendor_Code.astype("category",ordered=True,categories=data.Vendor_Code.unique()).cat.codes
#data.Product_Category = data.Product_Category.astype("category",ordered=True,categories=data.Product_Category.unique()).cat.codes
y = pd.get_dummies(data=data.Product_Category,columns=['Product_Category'])
product_cats = y.columns
traindata_y = y.as_matrix()
traindata = data.loc[:,'Vendor_Code']+' '+data.loc[:,'Item_Description'].as_matrix()
#print(traindata.shape)

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
train_data_features = vectorizer.fit_transform(traindata)
traindata_bow = vectorizer.transform(traindata).toarray()
#print(vectorizer.vocabulary_)

input_dim = traindata_bow.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(traindata_y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
print(model.summary())

history = model.fit(traindata_bow, traindata_y, epochs=100,verbose=True,batch_size=100)
loss, accuracy = model.evaluate(traindata_bow, traindata_y, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))