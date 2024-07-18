import joblib

def predict(data):
    log_reg = joblib.load('model/log_reg_model.sav')
    return log_reg.predict_proba(data)