import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
sns.set()

##modelling Linear Regression, we assume y=B0+B1X1+e
##Y(the dependant or predicted variable) is dependant on X(the predictor variable or independant variable)
def simple_linear_regression_model():
    data=pd.read_csv('/Users/tannerbraithwaite/github/python_scripts/The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S32_L187/1.01. Simple linear regression.csv')
    print(data.describe())
    y=data['GPA']
    x1=data['SAT']
    plt.scatter(x1, y)
    plt.xlabel('SAT', fontsize=20)
    plt.ylabel('GPA', fontsize=20)
    yhat=0.0017*x1 + 0.275
    fig = plt.plot(x1, yhat, lw=4, c='orange', label = 'regression line')
    plt.show()
    x = sm.add_constant(x1)
    results = sm.OLS(y,x).fit()
    print(results.summary())
    yhat=0.017*x1 + 0.275


##Notes:
## or TSS(Sum of Squares Total)= measures variability of between the difference of y values and the average y
##SSR or ESS - sum of squares of the regression - measures variability between your yhat and the y mean
##SSE(sum of squares error) - measures unexplained variability by regression(sum of the errors squared)
##SST = SSR+SSE can read as "The total error is the sum of the error between the regression, and the unexplained error"

def multiple_linear_regression_model():
    data=pd.read_csv('/Users/tannerbraithwaite/github/python_scripts/The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S33_L195/1.02. Multiple linear regression.csv')
    print(data.describe())
    y=data['GPA']
    x1=data[['SAT', 'Rand 1,2,3']]
    x = sm.add_constant(x1)
    results = sm.OLS(y,x).fit()
    print(results.summary())

##assumptions about linear models
##1. the function is Linear
##No endogeneity - the error is not correlated to an x value(typically implies a variable has been missed)
##normality and Homoscedasticity - The variation of the errors is 0, and normally distributed, variance changes with X, can perform transformations(such as a log transformation)
##No Auto corelation/serial corellation(cannot be relaxed) - errors do not follow predictable patterns, can use a Durbin watson test-2 is no auto corelation, values below 1 and above 3 are a cause for alarm, cannot use a linear model
##No Multicollinearity - variables do not effect one another, we can fix this by dropping a variable, combining the variables, or keep them both(must be extremely careful)

##Creating dummy variables
def dummy_variable_model():
    raw_data = pd.read_csv('/Users/tannerbraithwaite/github/python_scripts/The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S33_L204/1.03. Dummies.csv')
    data= raw_data.copy()
    data['Attendance']=data['Attendance'].map({'Yes':1,'No':0})
    y=data['GPA']
    x1=data[['SAT', 'Attendance']]
    x=sm.add_constant(x1)
    results = sm.OLS(y,x).fit()
    plt.scatter(data['SAT'], y, c=data['Attendance'],cmap='RdYlGn_r')
    yhat_no = 0.6439 + 0.0014*data['SAT']
    yhat_yes = 0.8665 + 0.0014*data['SAT']
    fig = plt.plot(data['SAT'], yhat_no, lw=2, c='blue')
    fig = plt.plot(data['SAT'], yhat_yes, lw=2, c='orange')
    plt.xlabel('SAT', fontsize =20)
    plt.ylabel("GPA", fontsize=20)

    new_data=pd.DataFrame({'const':1, 'SAT':[1700, 1670], 'Attendance':[0,1]})
    new_data.rename(index={0:'Bob', 1:'Alice'})
    predictions = results.predict(new_data)
    print(predictions)
    predictionsdf=pd.DataFrame({'Predictions':predictions})
    joined=new_data.join(predictionsdf)
    joined.rename(index={0:'Bob', 1:'Alice'})
    print(joined)


##Using SciKit(a more powerful library for dealing with data)

def sklearn_linear_regression_model():
    data=pd.read_csv('/Users/tannerbraithwaite/github/python_scripts/The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S32_L187/1.01. Simple linear regression.csv')
    data.head()
    x=data['SAT']
    y=data['GPA']
    x_matrix=x.values.reshape(-1,1)
    reg=LinearRegression()
    reg.fit(x_matrix,y)
    reg.score(x_matrix, y) ##get the rsquared value
    coefficient =reg.coef_ ##returns array of variable coefficients
    intercept=reg.intercept_ ##returns intercept
    new_predict = reg.predict(1740)
    print(new_predict)

##Feature Selection - if a variable has a p value greater than0.05 you can disregard it. feautre_selection.f_regression calculates each independant linear regression and returns the p values

def feature_selection_Model():
        data=pd.read_csv('/Users/tannerbraithwaite/github/python_scripts/The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S33_L195/1.02. Multiple linear regression.csv')
        y=data['GPA']
        x1=data['SAT']
        x = sm.add_constant(x1)
        results = sm.OLS(y,x).fit()
        print(f_regression(x,y))
        pvalues=f_regression(x,y)[1]
        pvalues.round(3)

def create_feature_table():
    data=pd.read_csv('/Users/tannerbraithwaite/github/python_scripts/The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S32_L187/1.01. Simple linear regression.csv')
    x1=data[['SAT', 'Rand 1,2,3']]
    x=sm.add_constant(x1)
    reg_summary = pd.DataFrame(data=x.columns.values, columns=['Features'])
    print(reg_summary)
    reg_summary['coefficients'] = reg.coef_
    pvalues=f_regression(x,y)[1]
    pvalues.round(3)
    reg_summary['coefficients'] = reg.coef_

def standardize_data():
    data=pd.read_csv('/Users/tannerbraithwaite/github/python_scripts/The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S33_L195/1.02. Multiple linear regression.csv')
    data.head()
    x=data[['SAT','Rand 1,2,3']]
    y=data['GPA']
    scaler=StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    reg=LinearRegression()
    reg.fit(x_scaled, y)
    reg.coef_
    reg.intercept_
    reg_summary=pd.DataFrame([['Bias'],['SAT'],['Rang 1,2,3']])
    reg_summary['Weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1]
    print(reg_summary)

def predict_with_standardized_model():
    new_data = pd.DataFrame(data=[[1700,2],[1800,1]],columns=['SAT', 'Ran 1,2,3'])
    data=pd.read_csv('/Users/tannerbraithwaite/github/python_scripts/The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S33_L195/1.02. Multiple linear regression.csv')
    data.head()
    x=data[['SAT','Rand 1,2,3']]
    y=data['GPA']
    scaler=StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    reg=LinearRegression()
    reg.fit(x_scaled, y)
    reg.coef_
    reg.intercept_
    reg_summary=pd.DataFrame([['Bias'],['SAT'],['Rang 1,2,3']])
    reg_summary['Weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1]
    new_data_scaled = scaler.transform(new_data)
    reg_simple = LinearRegression()
    x_simple_matrix = x_scaled[:,0].reshape(-1,1)
    reg_simple.fit(x_simple_matrix, y)
    print(reg_simple.predict(new_data_scaled[:,0].reshape(-1,1)))

def test_split():
    a=np.arange(1,101)
    b=np.arange(501,601)
    a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=65)
    print(a_train, a_test)

##Clean the data
raw_data=pd.read_csv('/Users/tannerbraithwaite/github/python_scripts/The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S35_L226/1.04. Real-life example.csv')
data = raw_data.drop(['Model'],axis=1) #drop the models column
print(data.isnull().sum()) #sum the missing values in the data set
data_no_mv = data.dropna(axis=0)
sns.distplot(data_no_mv['Price'])##plot show many high outliers
q=data_no_mv['Price'].quantile(0.99)
data_1=data_no_mv[data_no_mv['Price']<q]##remove the top 1 percent
q=data_1['Mileage'].quantile(0.99)
data_2=data_1[data_1['Mileage']<q]
data_3=data_2[data_2['EngineV']<6.5]
q=data_3['Year'].quantile(0.01)
data_4=data_3[data_3['Year']>q]
data_cleaned = data_4.reset_index(drop=True)
print(data_cleaned.describe())

##Check for OLS assumptions

##Check to see if the graph is linear and plot a log function if it is not
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price']=log_price
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('log price and year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('log price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('log price and Mileage')
data_cleaned = data_cleaned.drop(['Price'], axis=1)
# plt.show()

##check for endogeneity
##check for Normality and Homoscedasticity(we already implemented a log function to fix that)
##Check for Autocorrelation(observations are not dependant on each other)
##Check for Multicollinearity
variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
vif=pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
print(vif)##A value of 1 means no multi collinearity, 1-5 are acceptable values and great than 6 is unacceptable. between 5-6 is debatabl
data_no_multico=data_cleaned.drop(['Year'], axis=1)
##How ot create Dummys using pandas
data_with_dummies = pd.get_dummies(data_no_multico, drop_first=True)
print(data_with_dummies)
##Rearrange
print(data_with_dummies.columns.values)
cols =['Mileage' 'EngineV' 'log_price' 'Brand_BMW' 'Brand_Mercedes-Benz'
 'Brand_Mitsubishi' 'Brand_Renault' 'Brand_Toyota' 'Brand_Volkswagen'
 'Body_hatch' 'Body_other' 'Body_sedan' 'Body_vagon' 'Body_van'
 'Engine Type_Gas' 'Engine Type_Other' 'Engine Type_Petrol'
 'Registration_yes']
data_preprocess = data_with_dummies[cols]
data_preprocess.head()
target=data_preprocess['log_price']
inputs= data_preprocess.drop(['log_price'],axis=1)
scaler=StandardScaler()
scaler.fit(inputs)
##Scaling dummy variables is usually advised against, however for machine learning it may not have an effect
input_scaled=scaler.transform(inputs)
x_train, x_test, y_train, y_test = train_test_split(input_scaled, targets, test_size=0.2, random_state=42)

reg=LinearRegress()
reg.fit(x_train,y_train)
yhat=reg.predict(x_train)

##ultimately you wan a 45 degree line
plt.scatter(y-train, yhat)
sns.distplot(y_train-yhat)
plt.titl("Residuals PDF")
plt.show()
