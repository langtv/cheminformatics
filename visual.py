#######################################
# Data Visualization and ...          #
# @author: A.Prof. Tran Van Lang, PhD #
# File: visual.py                     #
#######################################

from sklearn.metrics import *
from joblib import dump, load
import matplotlib.pyplot as plt

# Trực quan hoá dữ liệu
def visualization(estimator, model, X_test, y_test ):
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('AUC:',roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    
    dump( model, 'Bioassay'+estimator+'.joblib' )
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay( confusion_matrix=cm,display_labels=["Active","Inactive"] ).plot()
    plt.show()