##LogisticRegressionModels.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
stats.chisqprob = lambda chisdq, df: stats.chi2.sf(chisq, df)
sns.set()

# data_folder = Path("The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S36_L236/")
#
# file_to_open = data_folder / "2.01. Admittance.csv"
#
# raw_data = pd.read_csv(file_to_open)
# data = raw_data.copy()
# data['Admitted'] = data['Admitted'].map({'Yes':1,'No':0})
# y = data['Admitted']
# x1= data['SAT']
#
# x=sm.add_constant(x1)
# reg_lin = sm.OLS(y,x)
# results_lin = reg_lin.fit()
#
# plt.scatter(x1,y,color='C0')
# yhat = x1*results_lin.params[1]+results_lin.params[0]
#
#
# reg_log = sm.Logit(y, x)
# results_log = reg_log.fit()
# print(results_log.summary())
# def f(x, b0, b1):
#     return np.array(np.exp(b0+x*b1) / (1+np.exp(b0+x*b1)))
#
# f_sorted = np.sort(f(x1, results_log.params[0], results_log.params[1]))
# x_sorted = np.sort(np.array(x1))
#
#
# plt.scatter(x1,y, color="C0")
# plt.xlabel('SAT', fontsize=20)
# plt.ylabel('Admitted', fontsize=20)
# plt.plot(x_sorted, f_sorted,color='C8')
# plt.show()

##Logistic models predict the probability of something happening
##The curve is defined by a logistic regression(an S between 0 and 1)
##logistic expression = e^(b0+b1x1+b2x2...)/(1+e^(b0+b1x1+b2x2....))
##We can also do the Logit form(P(X)/(1-P(X)) = e^(b0+b1x1+b2x2))
##Read as, the probability of the event occuring divided by the probability of the event not occuring is equal to the exponential
##You can take the log of both sides and the log of the odds of the event is equal to a linear model
data_folder = Path("The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S36_L244/")

file_to_open = data_folder / "2.02. Binary predictors.csv"

raw_data = pd.read_csv(file_to_open)
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1,'No':0})
data['Gender'] = data['Gender'].map({'Female':1,'Male':0})

y=data['Admitted']
x1=data[['SAT','Gender']]
x=sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()

np.array(data['Admitted'])

print(results_log.pred_table()) ##Print the confusion matrix(shows how confused your model is)
cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0:'Actual 0', 1:'Actual 1'})
print(cm_df)

cm=np.array(cm_df)
accuracy_train = (cm[0,0]+cm[1,1])/cm.sum()
print(accuracy_train)

#training your model

test_data = pd.read_csv('/Users/tannerbraithwaite/github/python_scripts/The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S36_L249/2.03. Test dataset.csv')
test_data['Admitted'] = test_data['Admitted'].map({'Yes':1,'No':0})
test_data['Gender'] = test_data['Gender'].map({'Female':1,'Male':0})
test_actual = test_data['Admitted']
test_data=test_data.drop(['Admitted'], axis=1)
test_data=sm.add_constant(test_data)

def confusion_matrix(data,actual_values,model):
    pred_values = model.predict(data)
    bins =np.array([0,0.5,1])
    cm = np.histogram2d(actual_values,pred_values, bins=bins)[0]
    accuracy = (cm[0,0]+cm[1,1])/cm.sum()
    return cm, accuracy



cm = confusion_matrix(test_data,test_actual,results_log)
cm_df = pd.DataFrame(cm[0])
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0:'Actual 0', 1:'Actual 1'})
print(cm_df)
