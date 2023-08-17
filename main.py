#%%
import base64
import os
import pickle
import requests
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

#%%
url = 'http://localhost:5555/api/gethog'

def imgToHOG(img):
    v, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer)
    data = "image data,"+str.split(str(img_str),"'")[1]
    response = requests.get(url, json={"item_str":data})
    return response.json()

car_class = ["Audi", "Hyundai Creta", "Mahindra Scorpio", "Rolls Royce",
             "Swift", "Tata Safari", "Toyota Innova"]

#Read data train
#%%
path = "train"
hogList = []
for subfolder in os.listdir(path):
    for img_file in os.listdir(os.path.join(path, subfolder)):
        img = cv2.imread(os.path.join(path, subfolder)+"/"+img_file)
        res = imgToHOG(img)
        hog = list(res["HOG"])
        hog.append(car_class.index(subfolder))
        hogList.append(hog)

#Save Feature
write_path = "imghog.pkl"
pickle.dump(hogList, open(write_path, 'wb'))

#Read data test
#%%
path = "test"
hogList = []
for subfolder in os.listdir(path):
    for img_file in os.listdir(os.path.join(path, subfolder)):
        img = cv2.imread(os.path.join(path, subfolder)+"/"+img_file)
        res = imgToHOG(img)
        hog = list(res["HOG"])
        hog.append(car_class.index(subfolder))
        hogList.append(hog)

#Save Feature test
write_path = "imghogtest.pkl"
pickle.dump(hogList, open(write_path, 'wb'))

#load data train
#%%
imghog = pickle.load(open(r'imghog.pkl', 'rb'))

dataArr = np.array(imghog)

print(dataArr[0][-1])

# %%
x_train = dataArr[:,0:-1]
y_train = dataArr[:,-1]

print(x_train.shape)

#Train DTree
# %%
clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)

#Test
#%%
imghogtest = pickle.load(open(r'imghogtest.pkl', 'rb'))
dataArr = np.array(imghogtest)
x_test = dataArr[:,0:-1]
y_test = dataArr[:,-1]

# %%
y_pred = clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
metrics.confusion_matrix(y_test, y_pred)

#Save model
#%%
write_path = "carmodel.pkl"
pickle.dump(clf, open(write_path, 'wb'))

#Train Xgboost
# %% 
from xgboost import XGBClassifier

xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(x_train,y_train)

#Predict Xgboost
# %% 
y_pred_xgb = xgb.predict(x_test)
acc_xgb = metrics.accuracy_score(y_test,y_pred_xgb)
cofus_mect_xgb = metrics.confusion_matrix(y_test,y_pred_xgb)
print("accuracy = ",acc_xgb*100)
print('Confusion Mectrix :\n',cofus_mect_xgb)

#Save model Xgboost
# %% 
path_model = 'carmodel_xgb.pkl'
pickle.dump(xgb,open(path_model,'wb'))
# %%
