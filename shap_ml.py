import xgboost
import lightgbm as lgb
from constants import ml_model_dic
# import catboost
def shap_ml_explain(data,model):
    X,y=data.iloc[:,1:],data.iloc[:,0]
    ml_model=ml_model_dic[model].fit(X=X,y=y)
    return(X,y,ml_model)
