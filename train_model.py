import os, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'dataset', 'bengaluru_ev_location_data.csv')
OUT = os.path.join(BASE, 'model', 'location_demand_model.pkl')
METRICS = os.path.join(BASE, 'model', 'metrics.json')

df = pd.read_csv(DATA)
features = ['area','nearby_ev_population','traffic_score','existing_stations','distance_to_nearest_station_km','station_gap_score','accessibility_score','hour','day_of_week','month','weather']
target = 'future_ev_demand'
X = df[features]
y = df[target]
cat = ['area','weather']
num = [c for c in features if c not in cat]
pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat), ('num','passthrough',num)])
model = Pipeline([('preprocessor', pre), ('regressor', RandomForestRegressor(n_estimators=220, random_state=42, n_jobs=-1, max_depth=18))])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42)
model.fit(X_train,y_train)
pred=model.predict(X_test)
joblib.dump(model, OUT)
import json
with open(METRICS,'w') as f: json.dump({'mae':round(float(mean_absolute_error(y_test,pred)),2),'r2':round(float(r2_score(y_test,pred)),3)},f,indent=2)
print('Saved:', OUT)
