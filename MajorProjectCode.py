import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
pd.set_option('mode.chained_assignment',None)
data=pd.read_excel('Data_Train.xlsx')
data=pd.DataFrame(data,columns=data.columns.to_list())

#Cleaning the Train Data
data=data.dropna(axis=0,how='any',subset=['Mileage','Engine'])
data.iloc[:,7]=[x.strip('kmpl') for x in data.Mileage]
data.iloc[:,7]=[x.strip('km/kg') for x in data.Mileage]
data.iloc[:,8]=[x.strip('CC') for x in data.Engine]
data.iloc[:,9]=[x.strip('bhp') for x in data.Power]
data.iloc[:,9]=[x.strip('null') for x in data.Power]
data=data.replace(r'^\s*$',np.nan,regex=True)
data=data.dropna(axis=0,how='any',subset=['Power'])
n= np.zeros(76362)
n = np.reshape(n,data.shape)
n=pd.DataFrame(n,columns = data.columns.to_list())#dataframe after removal of null valued data from Price feature
for i in data.columns:
    k=0
    for j in list(data.Price.keys()):
        n[i][k] = data[i][j]
        k+=1


Q1p = n.Price.quantile(0.25)
Q3p = n.Price.quantile(0.75)
IQRp = Q3p-Q1p
w=0
for i in n.Price:
    if(i<Q1p-(1.5*IQRp) or i>Q3p+(1.5*IQRp)):
        w+=1
m = (n.shape[0]-w)*n.shape[1]
m = np.zeros(m)
m = np.reshape(m,(n.shape[0]-w,n.shape[1]))
m = pd.DataFrame(m,columns = n.columns.to_list())
for i in m.columns:
    k=0
    for j in range(len(n[i])):
        if(n["Price"][j]<=Q3p+(1.5*IQRp)):
            
            m[i][k] = n[i][j]
            k+=1            

Q1kd = m.Kilometers_Driven.quantile(0.25)
Q3kd = m.Kilometers_Driven.quantile(0.75)
IQRkd = Q3kd-Q1kd
w=0
for i in m.Kilometers_Driven:
    if(i<Q1kd-(1.5*IQRkd) or i>Q3kd+(1.5*IQRkd)):
        w+=1
p = (m.shape[0]-w)*m.shape[1]
p  =np.zeros(p)
p = np.reshape(p,(m.shape[0]-w,m.shape[1]))
p = pd.DataFrame(p,columns = m.columns.to_list())
for i in p.columns:
    k=0
    for j in range(len(m[i])):
        if(m["Kilometers_Driven"][j]>=(Q1kd-(1.5*IQRkd)) and m["Kilometers_Driven"][j]<=(Q3kd+(1.5*IQRkd))):
            p[i][k] = m[i][j]
            k+=1
            
Q1po = p.Power.quantile(0.25)
Q3po = p.Power.quantile(0.75)
IQRpo = Q3po-Q1po
w=0
for i in p.Power:
    if(i<Q1po-(1.5*IQRpo) or i>Q3po+(1.5*IQRpo)):
        w+=1
q = (p.shape[0]-w)*p.shape[1]
q  =np.zeros(q)
q = np.reshape(q,(p.shape[0]-w,p.shape[1]))
q = pd.DataFrame(q,columns = p.columns.to_list())
for i in q.columns:
    k=0
    for j in range(len(p[i])):
        if(p["Power"][j]>=(Q1po-(1.5*IQRpo)) and p["Power"][j]<=(Q3po+(1.5*IQRpo))):
            q[i][k] = p[i][j]
            k+=1

Q1m = q.Mileage.quantile(0.25)
Q3m = q.Mileage.quantile(0.75)
IQRm = Q3m-Q1m
w=0
for i in q.Mileage:
    if(i<Q1m-(1.5*IQRm) or i>Q3m+(1.5*IQRm)):
        w+=1
b = (q.shape[0]-w)*q.shape[1]
b  =np.zeros(b)
b = np.reshape(b,(q.shape[0]-w,q.shape[1]))
b = pd.DataFrame(b,columns = q.columns.to_list())
for i in b.columns:
    k=0
    for j in range(len(q[i])):
        if(q["Mileage"][j]>=(Q1m-(1.5*IQRm)) and q["Mileage"][j]<=(Q3m+(1.5*IQRm))):
            b[i][k] = q[i][j]
            k+=1


Q1e = b.Engine.quantile(0.25)
Q3e= b.Engine.quantile(0.75)
IQRe = Q3e-Q1e
w=0
for i in b.Engine:
    if(i<Q1e-(1.5*IQRe) or i>Q3e+(1.5*IQRe)):
        w+=1
f = (b.shape[0]-w)*b.shape[1]
f  =np.zeros(f)
f = np.reshape(f,(b.shape[0]-w,b.shape[1]))
f = pd.DataFrame(f,columns = b.columns.to_list())
for i in f.columns:
    k=0
    for j in range(len(b[i])):
        if(b["Engine"][j]>=(Q1e-(1.5*IQRe)) and b["Engine"][j]<=(Q3e+(1.5*IQRe))):
            f[i][k] = b[i][j]
            k+=1











col = ["Mileage","Engine","Power","Seats","Kilometers_Driven"]
from sklearn.preprocessing import MinMaxScaler as mms
scaler = mms()
for i in col:
    mi = f[i].to_list()
    mi = np.reshape(mi,(-1,1))
    col_value = scaler.fit_transform(mi)
    #print(mileage)
    for j in range(len(f[i])):
        f[i][j] = col_value[j]
X=f.iloc[:,[2,3,4,5,6,7,8,9,10]].values

y=f.iloc[:,12].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:,8:9])
X[:,8:9] = imputer.transform(X[:,8:9])
lbl_obj=LabelEncoder()



X[:,2]=lbl_obj.fit_transform(X[:,2])
X[:,3]=lbl_obj.fit_transform(X[:,3])
X[:,4]=lbl_obj.fit_transform(X[:,4])
onehotencoder = OneHotEncoder(categorical_features = [2,3,4])
X = onehotencoder.fit_transform(X).toarray()

testing = pd.read_excel('Data_Test.xlsx')

testing=testing.dropna(axis=0,how='any',subset=['Seats','Mileage','Engine'])
testing.iloc[:,7]=[x.strip('kmpl') for x in testing.Mileage]
testing.iloc[:,7]=[x.strip('km/kg') for x in testing.Mileage]
testing.iloc[:,8]=[x.strip('CC') for x in testing.Engine]
testing.iloc[:,9]=[x.strip('bhp') for x in testing.Power]
testing.iloc[:,9]=[x.strip('null') for x in testing.Power]
testing=testing.replace(r'^\s*$',np.nan,regex=True)
testing=testing.dropna(axis=0,how='any',subset=['Power'])
testdata = pd.DataFrame(testing, columns = testing.columns.to_list())

col = ["Mileage","Engine","Power","Seats","Kilometers_Driven"]
for i in col:
    for j in list(testdata[i].keys()):
        testdata[i][j] = float(testdata[i][j])

col = ["Mileage","Engine","Power","Kilometers_Driven"]

for i in col:
    if(i=="Mileage"):
        Q1 = Q1m
        Q3 = Q3m
        IQR = IQRm
        med = f["Mileage"].median()
    elif(i=="Power"):
        Q1 = Q1po
        Q3 = Q3po
        IQR = IQRpo
        med = f["Power"].median()
    elif(i=="Engine"):
        Q1 = Q1e
        Q3 = Q3e
        IQR = IQRe
        med = f["Engine"].median()
    elif(i == "Kilometers_Driven"):
        Q1 = Q1kd
        Q3 = Q3kd
        IQR = IQRkd
        med = f["Kilometers_Driven"].median()
    for j in list(testdata[i].keys()):
        if(testdata[i][j]<Q1-(1.5*IQR) or testdata[i][j]>Q3 - (1.5*IQR)):
            testdata[i][j] = med




col = ["Mileage","Engine","Power","Seats","Kilometers_Driven"]
from sklearn.preprocessing import MinMaxScaler as mms
scaler = mms()
for i in col:
    mi = testdata[i].to_list()
    mi = np.reshape(mi,(-1,1))
    col_value = scaler.fit_transform(mi)
    #print(mileage)
    for j in range(len(testdata[i])):
        testdata[i][j] = col_value[j]



Xt=testdata.iloc[:,[2,3,4,5,6,7,8,9,10]].values



from sklearn.preprocessing import LabelEncoder,OneHotEncoder,Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(Xt[:,8:9])
Xt[:,8:9] = imputer.transform(Xt[:,8:9])
lbl_obj=LabelEncoder()

Xt[:,2]=lbl_obj.fit_transform(Xt[:,2])
Xt[:,3]=lbl_obj.fit_transform(Xt[:,3])
Xt[:,4]=lbl_obj.fit_transform(Xt[:,4])
onehotencoder = OneHotEncoder(categorical_features = [2,3,4])
Xt = onehotencoder.fit_transform(Xt).toarray()




# Import the model we are using
from sklearn.ensemble import RandomForestRegressor# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)# Train the model on training data
rf.fit(X, y);

predictions = rf.predict(Xt)
print(predictions)






