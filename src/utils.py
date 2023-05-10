import os 
import sys
import numpy as np 
import pandas as pd 
import pickle

from src.exception import  CustomException
from src.logger import logging
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report,accuracy_score,precision_score,recall_score,f1_score

def save_object(file_path,obj):
    try:
        
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e :
        logging.info('Error has ocured in the save_object')
        raise CustomException(e,sys)
    

def evaluate_model(x_train,y_train,x_test,y_test,model):
    try:
        
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        performance = classification_report(y_test,y_pred)
        auc = roc_auc_score(y_test,model.predict(x_test))
        cm = confusion_matrix(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred)

        report = {}
        report['accuracy'] = accuracy
        report['performance'] = performance
        report['auc'] = auc
        report['cm'] = cm
        report['precision']=precision
        report['recall']=recall
        report['f1']=f1
        
        logging.info('Model evaluation completed')
        logging.info(report)

        return report
    
    except Exception as e:
        logging.info('Error has occured in evaluate model')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try :
        with open (file_path,'rb') as file_obj:
            return pickle.load(file_obj)
            
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)