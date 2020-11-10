import pandas as pd
df=pd.read_csv("vgsales.csv")
#%%
df.drop(['Rank','Name','Platform','Genre','Publisher'],axis=1,inplace=True)
#%%
x=df.iloc[:,1::].values
y=df.iloc[:,0].values
#%%
from sklearn.linear_model import LinearRegression
mlr=LinearRegression()
mlr.fit(x,y)
mlr_pred=mlr.predict(x)
#%%
from  sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print("mlr r2:",r2_score(y,mlr_pred))
print("mlr mae:",mean_absolute_error(y,mlr_pred))
print("mlr mse:",mean_squared_error(y,mlr_pred))
#%%
from sklearn.preprocessing import PolynomialFeatures
plr=PolynomialFeatures(degree=3)

x_pol=plr.fit_transform(x)
lr_pol=LinearRegression()
lr_pol.fit(x_pol,y)
xnew=plr.fit_transform(x)
y_pred_pol=lr_pol.predict(xnew)
#%%
pol_pred=lr_pol.predict(x_pol)

print("pol r2:",r2_score(y,pol_pred))
print("pol mae:",mean_absolute_error(y,pol_pred))
print("pol mse:",mean_squared_error(y,pol_pred))
#%%
from  sklearn.tree import  DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x,y)
dt_pred=dt.predict(x)
#%%
from  sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print("dt_r2:",r2_score(y,dt_pred))   
print("mae:",mean_absolute_error(y,dt_pred))
print("mse:",mean_squared_error(y,dt_pred))
#%%
from  sklearn.ensemble import  RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x,y.ravel())
rf_pred=rf.predict(x)
#%%
from  sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print("rf","r2:",r2_score(y,rf_pred),"mae:",mean_absolute_error(y,rf_pred))
print("mse:",mean_squared_error(y,rf_pred))











