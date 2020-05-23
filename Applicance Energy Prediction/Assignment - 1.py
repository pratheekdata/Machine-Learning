# Packages
import pandas as pd
import numpy as np
import seaborn as sns
import random
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import math
from sklearn.metrics import accuracy_score

# Filtering Features using Corelation


def correlation(data, threshold=0.9):
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = data.corr()
    print('Correlation with more than 0.88')
    print('Corr Value',"\t""\t", 'Fearture1', "\t", 'Feature 2')
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= 0.88) and (corr_matrix.columns[j] not in col_corr):
                print(corr_matrix.iloc[i, j],"\t", corr_matrix.columns[i],"\t", corr_matrix.columns[j])


## Liner Regression Function

# Train data function
def ln_reg_train_1(x, y,ibeta):
    num_row = x.shape[0]
    num_col = x.shape[1] + 1
    print(num_col)
    lrate = [0.007, 0.005, 0.01]
    beta_dict = dict()
    cf_dict = dict()
    mse_train = dict()
    mae_train = dict()
    rsq_train = dict()
    cf = [0] * 1000
    xt = x.values.transpose()
    print(datetime.datetime.now().time())
    for alpha in lrate:
        p_diff_cost = [0] * num_col
        beta = [ibeta] * num_col
        cf = [0] * 1000
        mse = [0] * 1000
        mae = [0] * 1000
        rsq = [0] * 1000
        check = 0
        for iter in range(1000):
            for k in range(num_col):
                if (k == 0):
                    p_diff_cost[0] = ((beta[0] + np.dot(beta[1:], xt)) - y).sum()
                else:
                    p_diff_cost[k] = np.dot((beta[0] + (np.dot(beta[1:], xt)) - y), x.iloc[:, k - 1])
            for bit in range(num_col):
                beta[bit] = round((beta[bit] - (alpha * p_diff_cost[bit]) / num_row), 4)

            pred_train = beta[0] + np.dot(beta[1:], xt)

            cf[iter], mse[iter], mae[iter], rsq[iter] = measures(y, pred_train, num_row)
            if cf[iter] <= 300 and check == 0:
                print("Learning rate: %.3f, Iteration of Convergence: %d" % (alpha, iter))
                check = 1
        beta_dict[alpha], cf_dict[alpha], mse_train[alpha], mae_train[alpha], rsq_train[alpha] = beta, cf, mse, mae, rsq
    print(datetime.datetime.now().time())
    return lrate, beta_dict, cf_dict, mse_train, mae_train, rsq_train

def ln_reg_train_2(x, y,ibeta):
    num_row = x.shape[0]
    num_col = x.shape[1] + 1
    print(num_col)
    lrate = [0.01]
    threshold = [0.005,0.01,0.1]
    beta_dict = dict()
    cf_dict = dict()
    mse_train = dict()
    mae_train = dict()
    rsq_train = dict()
    cf = [0] * 1000
    xt = x.values.transpose()
    print(datetime.datetime.now().time())
    for t in threshold:
        for alpha in lrate:
            p_diff_cost = [0] * num_col
            beta = [ibeta] * num_col
            cf = [0] * 2000
            mse = [0] * 2000
            mae = [0] * 2000
            rsq = [0] * 2000
            check = 0
            for iter in range(2000):
                for k in range(num_col):
                    if (k == 0):
                        p_diff_cost[0] = ((beta[0] + np.dot(beta[1:], xt)) - y).sum()
                    else:
                        p_diff_cost[k] = np.dot((beta[0] + (np.dot(beta[1:], xt)) - y), x.iloc[:, k - 1])
                for bit in range(num_col):
                    beta[bit] = round((beta[bit] - (alpha * p_diff_cost[bit]) / num_row), 4)

                pred_train = beta[0] + np.dot(beta[1:], xt)

                cf[iter], mse[iter], mae[iter], rsq[iter] = measures(y, pred_train, num_row)
                if cf[iter-1]- cf[iter] <= t and iter > 0:
                    print("Learning rate: %.3f, Iteration of Convergence: %d, Threshold: %.3f" % (alpha, iter,t))
                    break

            beta_dict[t], cf_dict[t], mse_train[t], mae_train[t], rsq_train[t] = beta, cf, mse, mae, rsq
        print(datetime.datetime.now().time())
    return threshold, beta_dict, cf_dict, mse_train, mae_train, rsq_train


def ln_reg_train_3(x, y, ibeta):
    xt = x.values.transpose()
    num_row = x.shape[0]
    num_col = x.shape[1] + 1
    print(num_col)
    lrate = [0.01]
    beta_dict = dict()
    cf_dict = dict()
    mse_train = dict()
    mae_train = dict()
    rsq_train = dict()
    cf = [0] * 1000
    print(datetime.datetime.now().time())
    for alpha in lrate:
        p_diff_cost = [0] * num_col
        beta = [ibeta] * num_col
        cf = [0] * 1000
        mse = [0] * 1000
        mae = [0] * 1000
        rsq = [0] * 1000
        check = 0
        for iter in range(1000):
            for k in range(num_col):
                if (k == 0):
                    p_diff_cost[0] = ((beta[0] + np.dot(beta[1:], xt)) - y).sum()
                else:
                    p_diff_cost[k] = np.dot((beta[0] + (np.dot(beta[1:], xt)) - y), x.iloc[:, k - 1])
            for bit in range(num_col):
                beta[bit] = round((beta[bit] - (alpha * p_diff_cost[bit]) / num_row), 4)

            pred_train = beta[0] + np.dot(beta[1:], xt)

            cf[iter], mse[iter], mae[iter], rsq[iter] = measures(y, pred_train, num_row)
            if cf[iter] <= 300 and check == 0:
                print("Learning rate: %.3f, Iteration of Convergence: %d" % (alpha, iter))
                check = 1
        beta_dict[alpha], cf_dict[alpha], mse_train[alpha], mae_train[alpha], rsq_train[alpha] = beta, cf, mse, mae, rsq
    print(datetime.datetime.now().time())
    return lrate, beta_dict, cf_dict, mse_train, mae_train, rsq_train

def ln_reg_test_1(x, y, beta_dict):
    xt = x.values.transpose()
    num_row = x.shape[0]
    lrate = [0.007, 0.005, 0.01]
    cf_test = {}
    mse_test = {}
    mae_test = {}
    rsq_test = {}
    for alpha in lrate:
        beta = beta_dict[alpha]
        pred_test = beta[0] + np.dot(beta[1:], xt)
        cf_test[alpha], mse_test[alpha], mae_test[alpha], rsq_test[alpha] = measures(y, pred_test, num_row)
    return cf_test, mse_test, mae_test, rsq_test


def ln_reg_test_2(x, y, beta_dict):
    xt = x.values.transpose()
    num_row = x.shape[0]
    threshold = [0.005, 0.01, 0.1]
    cf_test = {}
    mse_test = {}
    mae_test = {}
    rsq_test = {}
    for t in threshold:
        beta = beta_dict[t]
        pred_test = beta[0] + np.dot(beta[1:], xt)
        cf_test[t], mse_test[t], mae_test[t], rsq_test[t] = measures(y, pred_test, num_row)
    return cf_test, mse_test, mae_test, rsq_test

def ln_reg_test_3(x, y, beta_dict):
    xt = x.values.transpose()
    num_row = x.shape[0]
    lrate = [0.01]
    cf_test = {}
    mse_test = {}
    mae_test = {}
    rsq_test = {}
    for alpha in lrate:
        beta = beta_dict[alpha]
        pred_test = beta[0] + np.dot(beta[1:], xt)
        cf_test[alpha], mse_test[alpha], mae_test[alpha], rsq_test[alpha] = measures(y, pred_test, num_row)
    return cf_test, mse_test, mae_test, rsq_test

def measures(y, yhat, n):
    # Cost Function
    cf = round((((yhat - y) ** 2).sum()) / (2 * n), 4)
    # Mean Squared Error
    mse = round((np.mean((y - yhat) ** 2)), 4)
    # Mean Absolute Error
    mae = round((np.mean(abs(y - yhat))), 4)
    # Actual Y mean
    y_mean = y.mean()
    # Total Sum of Squares
    tss = np.sum((y - y_mean) ** 2)
    # Residual Sum of Squares
    rss = np.sum((y - yhat) ** 2)
    # R Squared
    r_square = round((1 - (rss / tss)), 4)
    return cf, mse, mae, r_square




# Importing Dataset
data = pd.read_csv("energydata_complete.csv")
datadf = pd.DataFrame(data)

print(correlation(datadf))


# #Corealtion PLot
# corr = datadf.corr()
# ax = sns.heatmap(corr,vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');


# Dropping the Highly Correlated Features
datadf =datadf.drop(columns =['date', 'T3', 'RH_3', 'T_out', 'rv1', 'rv2', 'T4', 'RH_4', 'T7', 'RH_7', 'T5'])
# print(datadf.columns)

# Dropping features by values
print('Value Counts of Lights')
print(datadf['lights'].value_counts())

datadf = datadf.drop(columns=['lights', 'Visibility'])
print(datadf.columns)
# plt.boxplot(datadf)
# plt.show()
# datadf.describe()
#
# Removing Outliers using the interquartile range
datadf_rm_out = datadf[datadf['Appliances']<=175]
#
# # Describing the Data
# print('Correlation:', correlation(datadf_rm_out))
# print(datadf_rm_out.describe())
# print(datadf_rm_out.count())

#Scaling the features
scaler = StandardScaler()
datadf_sc = scaler.fit_transform(datadf_rm_out)
datadf_sc = pd.DataFrame(datadf_sc, columns=datadf_rm_out.columns)

print('Scaled Data:', datadf_sc)
print(datadf_sc.count())

y = datadf_rm_out['Appliances']
x = datadf_sc.iloc[:, 1:]

# division of data
x_train, X_test, y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=5)


# Modeling
lrate, beta_dict, cf_dict, mse_train, mae_train, rsq_train = ln_reg_train_1(x_train, y_train,0.5)

# Plots

## Cost Function Plot

for i in lrate:
    plt.plot(cf_dict[i],label='Alpha: %.3f' %i)
plt.legend(loc="upper right")
plt.title('Cost Function Plot')
plt.show()

## Mean Square Error Plot
for i in lrate:
    plt.plot(mse_train[i], label='Alpha: %.3f' %i)
plt.legend(loc="upper right")
plt.title('Mean Square Error Plot')
plt.show()

## Mean Absolute Error Plot
for i in lrate:
    plt.plot(mae_train[i], label='Alpha: %.3f' %i)
plt.legend(loc="upper right")
plt.title('Mean Absolute Error Plot')
plt.show()

# Optimal Values for Training Data
print('Optimal Values for Training Data')
cf_final = {}
mse_final = {}
mae_final = {}
r2_final = {}
for alpha in lrate:
    cf_final[alpha] = min(cf_dict[alpha])
    mse_final[alpha] = min(mse_train[alpha])
    mae_final[alpha] = min(mae_train[alpha])
    r2_final[alpha] = max(rsq_train[alpha])
#
# # Measures for Training Data
print('Measures for Training Data')
print(pd.DataFrame([cf_final, mse_final, mae_final, r2_final], index=['CF', 'MSE', 'MAE', 'R\u00b2']).T)


# Testing on test data
print('Testing on test data')
cf_test, mse_test, mae_test, rsq_test = ln_reg_test_1(X_test, Y_test, beta_dict)

# Measures for Test Data
print('Measures for Test Data')
print(pd.DataFrame([cf_test, mse_test, mae_test, rsq_test], index=['CF', 'MSE', 'MAE', 'R\u00b2']).T)

#Modeling Training Data for Exp 2
threshold, beta_dict, cf_dict, mse_train, mae_train, rsq_train = ln_reg_train_2(x_train, y_train,0.5)
# print(threshold)
# print(cf_dict)

# Plots

## Cost Function Plot-2

for i in threshold:
    plt.plot(cf_dict[i],label='Threshold: %.3f' %i)
plt.legend(loc="upper right")
plt.title('Cost Function Plot')
plt.show()

## Mean Square Error Plot-2
for i in threshold:
    plt.plot(mse_train[i], label='Threshold: %.3f' %i)
plt.legend(loc="upper right")
plt.title('Mean Square Error Plot')
plt.show()

## Mean Absolute Error Plot-2
for i in threshold:
    plt.plot(mae_train[i], label='Threshold: %.3f' %i)
plt.legend(loc="upper right")
plt.title('Mean Absolute Error Plot')
plt.show()

#Flitering
# cf_fi = {d : e for d, e in cf_dict.items() if e!='0'}
# cf_filtered = filter(lambda a: a != 2, cf_dict)






# Optimal Values for Training Data Experiment 2
print('Optimal Values for Training Data Experiment 2')
cf_final = {}
mse_final = {}
mae_final = {}
r2_final = {}
for t in threshold:
    cf_final[t] = min([x for x in cf_dict[t] if x!=0])
    mse_final[t] = min([x for x in mse_train[t] if x!=0])
    mae_final[t] = min([x for x in mae_train[t] if x!=0])
    r2_final[t] = max(rsq_train[t])

# Measures for Training Data Experiment 2
print('Measures for Training Data Experiment 2')
print(pd.DataFrame([cf_final, mse_final, mae_final, r2_final], index=['CF', 'MSE', 'MAE', 'R\u00b2']).T)


# Testing on test data
print('Testing on test data')
cf_test, mse_test, mae_test, rsq_test = ln_reg_test_2(X_test, Y_test, beta_dict)

# Measures for Test Data
print('Measures for Test Data')
print(pd.DataFrame([cf_test, mse_test, mae_test, rsq_test], index=['CF', 'MSE', 'MAE', 'R\u00b2']).T)

#_______________________________________________________________________________________________________________
# # Experiment 3
# # _______________________________________________________________________________________________________________
datadf = datadf.drop(columns=['date'])
#Removing Outliers using the interquartile range
datadf_rm_out = datadf[datadf['Appliances']<=175]

#Scaling the features
scaler = StandardScaler()
datadf_sc = scaler.fit_transform(datadf_rm_out)
datadf_sc = pd.DataFrame(datadf_sc, columns=datadf_rm_out.columns)

random.seed(1)
random_features = random.sample(range(27), 10)
print(random_features)
x_random = datadf_sc.iloc[:,random_features]

print('The Random features picked are:')
print(x_random.columns)
y=datadf_rm_out['Appliances']
x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(x_random, y, test_size=0.3,random_state=5)

lrate, beta_dict, cf_dict, mse_train, mae_train, rsq_train = ln_reg_train_3(x_train_3, y_train_3,0.5)


# Optimal Values for Training Data
print('Optimal Values for Training Data for Experiment 3')
cf_final = {}
mse_final = {}
mae_final = {}
r2_final = {}
for alpha in lrate:
    cf_final[alpha] = min(cf_dict[alpha])
    mse_final[alpha] = min(mse_train[alpha])
    mae_final[alpha] = min(mae_train[alpha])
    r2_final[alpha] = max(rsq_train[alpha])

# Measures for Training Data
print('Measures for Training Data for Experiment 3')
print(pd.DataFrame([cf_final, mse_final, mae_final, r2_final], index=['CF', 'MSE', 'MAE', 'R\u00b2']).T)


# Testing on test data
print('Testing on test data for Experiment 3')
cf_test, mse_test, mae_test, rsq_test = ln_reg_test_3(x_test_3, y_test_3, beta_dict)

# Measures for Test Data
print('Measures for Test Data for experiment 3')
print(pd.DataFrame([cf_test, mse_test, mae_test, rsq_test], index=['CF', 'MSE', 'MAE', 'R\u00b2']).T)

#_____________________________________________________________________________________________________

#Experiment 4
print(datadf_rm_out.corr())
#Dropping Columns with low corelation to Dependent Variable
finaldata = datadf
finaldata.drop(columns=['RH_1','T3','T4','RH_4','T5','T6', 'T7', 'RH_7', 'T9','RH_9', 'T_out', 'Press_mm_hg','Windspeed', 'Visibility','Tdewpoint', 'rv1', 'rv2','date'])


print(finaldata.columns)

# # # # ****************************************************************************************************************
# # LOGISTIC REGRESSION
#
# Splitting classes based on its Median
datadf_log = datadf_rm_out.copy()
datadf_log['Appliances_class'] = [0 if x <= 60 else 1 for x in datadf_rm_out['Appliances']]
datadf_log = datadf_log.drop(columns=['Appliances'])

# Exploration of Data
print('Exploration of Data')
print(datadf_log['Appliances_class'].value_counts())
print(datadf_log.head())


# Scaling Data
scaler = StandardScaler()
datadf_log_sc = scaler.fit_transform(datadf_log)
datadf_log_sc = pd.DataFrame(datadf_log_sc, columns=datadf_log.columns)

# Datasets
x = datadf_log_sc.iloc[:, :15]
y = datadf_log['Appliances_class']

# Division of Data
X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)
print(X_train.shape)

# # Logit Modeling on Train
#
# model_logit = LogisticRegression()
# model_logit.fit(X_train, Y_train)
# y_pred_logit = model_logit.predict(X_train)
# print("Training Accuracy", model_logit.score(X_train, Y_train))
#
#
# # Logit Modeling on Test
# y_pred_test = model_logit.predict(x_test)
# print("Testing Accuracy", model_logit.score(x_test, y_test))
#
# print(model_logit.predict_proba(x_test))
# print(y_pred_test)
#
# # SGD Classifier on Train
# model_logit_sgd = SGDClassifier(loss='log')
# model_logit_sgd.fit(X_train, Y_train)
# y_pred_logit_sgd = model_logit_sgd.predict(X_train)
# print("Training Accuracy", model_logit_sgd.score(X_train, Y_train))
#
# # SGD Classifier on Test
#
# y_pred_test = model_logit_sgd.predict(x_test)
# print("Testing Accuracy", model_logit_sgd.score(x_test, y_test))

# Sigmoid function
def predict(beta,xt):
    z = beta[0]+(np.dot(beta[1:],xt))
    sigmoid = 1 / (1 + np.exp(-z))
    return(sigmoid)

# Cost Function
def cost_function(final_pred,y_train):
    a=y_train*np.log(final_pred)
    b=(1-y_train)*np.log(1-final_pred)
    cost=(a+b).sum()
    return(cost)

#GD for Logistic


##TRAINING DATA

cf_dict = {}
lrate = [0.001, 0.005, 0.01, 0.05]
m = X_train.shape[0]
xt = X_train.values.transpose()

print(datetime.datetime.now().time())
for l in lrate:
    p_diff_cost = [0]*16
    beta = [0.05]*16
    cf = [0]*10
    for iter in range(10):
        for k in range(16):
            if(k == 0):
                p_diff_cost[0] = (predict(beta, xt)-Y_train).sum()
            else:
                p_diff_cost[k] = np.dot((predict(beta, xt)-Y_train), X_train.iloc[:, k-1])

        for bit in range(16):
            beta[bit] = round((beta[bit]-(l*p_diff_cost[bit])/m), 4)

        final_pred = predict(beta, xt)
        cf[iter] = -(cost_function(final_pred, y_train))/m
        cf_dict[l] = cf

print(datetime.datetime.now().time())


# PLOTS

## Cost Function Plot
for i in lrate:
    plt.plot(cf_dict[i],label='Alpha: %.3f' %i)
plt.legend(loc="upper right")
plt.title('Cost Function Plot')
plt.show()

##
# plt.plot(cf)

#Predicting Classes
pred_class = [0 if i<=0.5 else 1 for i in final_pred]
pred_class=pd.Series(pred_class)

#Confusion Matrix
print("Confusion Matrix")
print(confusion_matrix(Y_train,pred_class))

### (tn, fp, fn, tp)
tn, fp, fn, tp = confusion_matrix(Y_train,pred_class).ravel()

confusion_matrix(Y_train,pred_class).ravel()

Sensitivity = tp/(tp+fn)
Specificity = tn/(tn+fp)

print('Sensitivity:',"\t",Sensitivity)
print('Specificity:',"\t",Specificity)

print("     Accuracy Score:     ")
print("\t",accuracy_score(Y_train,pred_class))

#ROC CURVE

fpr, tpr, thresholds = metrics.roc_curve(Y_train, final_pred,pos_label=1)

# print('False Positive Rate')
# print(fpr)
#
# print('True Positive Rates')
# print(tpr)
#
# print('Thresholds')
# print(thresholds)

roc_auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve: %0.3f" %roc_auc)

plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example for Train Data')
plt.legend(loc="lower right")
plt.show()


#Test Dataset

xt=x_test.values.transpose()
class_pred_test = predict(beta,xt)
test_cf = -(cost_function(class_pred_test,y_test))/m
print(test_cf)

pred_test = [0 if i<=0.5 else 1 for i in class_pred_test]
pred_test=pd.Series(pred_test)


print("Confusion matrix for Test")
print(confusion_matrix(y_test,pred_test))

### (tn, fp, fn, tp)
tn, fp, fn, tp = confusion_matrix(y_test,pred_test).ravel()

confusion_matrix(y_test,pred_test).ravel()


Sensitivity = tp/(tp+fn)
Specificity = tn/(tn+fp)

print('Sensitivity:',Sensitivity)
print('Specificity:',Specificity)

print("     Accuracy Score:     ")
print("\t",accuracy_score(y_test,pred_test))


fpr, tpr, thresholds = metrics.roc_curve(y_test, class_pred_test,pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve: %0.3f" %roc_auc)
plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example for Test Data')
plt.legend(loc="lower right")
plt.show()


# Experiment 3
# Splitting classes based on its Median
datadf_log = datadf_rm_out.copy()
datadf_log['Appliances_class'] = [0 if x <= 60 else 1 for x in datadf_rm_out['Appliances']]
datadf_log = datadf_log.drop(columns=['Appliances'])

#Column manipulation
cols = list(datadf_log.columns)
cols = [cols[-1]] + cols[:-1]
datadf_log = datadf_log[cols]


# Scaling Data
scaler = StandardScaler()
datadf_log_sc = scaler.fit_transform(datadf_log)
datadf_log_sc = pd.DataFrame(datadf_log_sc, columns=datadf_log.columns)

random.seed(1)
random_features = random.sample(range(27), 10)
print(random_features)
x_random = datadf_log_sc.iloc[:,random_features]

print('The Random features picked are:')
print(x_random.columns)

y=datadf_log['Appliances_class']

x_train_log3, x_test_log3, y_train_log3, y_test_log3 = train_test_split(x_random, y, test_size=0.3,random_state=5)

#Sigmoid function
def predict(beta,xt):
    z = beta[0]+(np.dot(beta[1:],xt))
    sigmoid = 1 / (1 + np.exp(-z))
    return(sigmoid)

# Cost Function
def cost_function(final_pred,y_train):
    a=y_train*np.log(final_pred)
    b=(1-y_train)*np.log(1-final_pred)
    cost=(a+b).sum()
    return(cost)

#GD for Logistic


##TRAINING DATA

cf_dict = {}
lrate = [0.01]
xt = x_train_log3.values.transpose()

print(datetime.datetime.now().time())
for l in lrate:
    xt = x_train_3.values.transpose()
    num_row = x_train_3.shape[0]
    num_col = x_train_3.shape[1] + 1
    p_diff_cost = [0]*num_col
    beta = [0.05]*num_col
    cf = [0]*10
    for iter in range(10):
        for k in range(num_col):
            if(k == 0):
                p_diff_cost[0] = (predict(beta, xt)-y_train_log3).sum()
            else:
                p_diff_cost[k] = np.dot((predict(beta, xt)-y_train_log3), x_train_log3.iloc[:, k-1])

        for bit in range(num_col):
            beta[bit] = round((beta[bit]-(l*p_diff_cost[bit])/num_row), 4)

        final_pred = predict(beta, xt)
        cf[iter] = -(cost_function(final_pred, y_train_log3))/num_row
        cf_dict[l] = cf

print(datetime.datetime.now().time())

# PLOTS

## Cost Function Plot
for i in lrate:
    plt.plot(cf_dict[i],label='Alpha: %.3f' %i)
plt.legend(loc="upper right")
plt.title('Cost Function Plot')
plt.show()

##
# plt.plot(cf)

#Predicting Classes
pred_class = [0 if i<=0.5 else 1 for i in final_pred]
pred_class=pd.Series(pred_class)

#Confusion Matrix
print("Confusion Matrix")
print(confusion_matrix(y_train_log3,pred_class))

### (tn, fp, fn, tp)
tn, fp, fn, tp = confusion_matrix(y_train_log3,pred_class).ravel()

confusion_matrix(y_train_log3,pred_class).ravel()

Sensitivity = tp/(tp+fn)
Specificity = tn/(tn+fp)

print('Sensitivity:',"\t",Sensitivity)
print('Specificity:',"\t",Specificity)

print("     Accuracy Score:     ")
print("\t",accuracy_score(y_train_log3,pred_class))

#ROC CURVE

fpr, tpr, thresholds = metrics.roc_curve(y_train_log3, final_pred,pos_label=1)

# print('False Positive Rate')
# print(fpr)
#
# print('True Positive Rates')
# print(tpr)
#
# print('Thresholds')
# print(thresholds)

roc_auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve: %0.3f" %roc_auc)

plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example for Train Data')
plt.legend(loc="lower right")
plt.show()


#Test Dataset

xt=x_test_log3.values.transpose()
class_pred_test = predict(beta,xt)
test_cf = -(cost_function(class_pred_test,y_test_log3))/num_row
print(test_cf)

pred_test = [0 if i<=0.5 else 1 for i in class_pred_test]
pred_test=pd.Series(pred_test)


print("Confusion matrix for Test")
print(confusion_matrix(y_test_log3,pred_test))

### (tn, fp, fn, tp)
tn, fp, fn, tp = confusion_matrix(y_test_log3,pred_test).ravel()

confusion_matrix(y_test_log3,pred_test).ravel()


Sensitivity = tp/(tp+fn)
Specificity = tn/(tn+fp)

print('Sensitivity:',Sensitivity)
print('Specificity:',Specificity)

print("     Test Accuracy Score:     ")
print("\t",accuracy_score(y_test_log3,pred_test))


fpr, tpr, thresholds = metrics.roc_curve(y_test_log3, class_pred_test,pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve: %0.3f" %roc_auc)
plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example for Test Data')
plt.legend(loc="lower right")
plt.show()
