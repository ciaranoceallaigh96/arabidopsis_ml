#perform neural network, svm and random forest, LASSO, RIDGE, CNN
#source ~/venv/bin/activate #in python 3.5.2
#model = pickle.load(open('FILEPATH', 'rb'))
#dependencies = {'coeff_determination':coeff_determination}
#model = tf.keras.models.load_model('FILEPATH', custom_objects=dependencies)
#



import sys

num = sys.argv[1] #script number for saving out
phenfile = str(sys.argv[2]) #txt file with phenotypes
data = str(sys.argv[3]) #needs to be same size as set_size
snps = str(sys.argv[4]) #top or shuf
phenotype = str(sys.argv[5]) #make a directory for the results
set_size = int(sys.argv[6]) #how many SNPs
organism = str(sys.argv[7]) #which directory mouse or arabadopsis (the mis-spelling is needed)
binary = str(sys.argv[8]) #True or False
binary_boolean = True if binary == 'True' else False
iterations = int(sys.argv[9])


sys.path.insert(1, 'Library/Python/3.7/python/site-packages/nested_cv')
import nested_cv
from nested_cv import NestedCV


#from sklearn.model_selection import cross_val_score
import statistics
import numpy as np
import sklearn
import seaborn as sns
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
import datetime
import time
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge
#import pickle #use dill instead below
from sklearn.ensemble import RandomForestRegressor #based in part on https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
from statistics import mean
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import make_scorer
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten #.core
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor # or Classifier
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
import random
#https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/random_forest_explained/Improving%20Random%20Forest%20Part%202.ipynb
import random
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.keras.layers import Dense, Conv1D, Flatten
import collections
import operator

if binary == 'True':
        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
        from tensorflow.keras.metrics import AUC
        from sklearn.svm import LinearSVC
        from sklearn.svm import SVC


sys.path.insert(1, 'Library/Python/3.7/python/site-packages/')
import dill as pickle
for i in range(1,len(sys.argv)):
        print(sys.argv[i])

if organism != 'mouse':
        if not os.path.exists(organism + '/' + phenotype+ '/' + snps):
                os.makedirs(organism + '/' + phenotype+ '/' + snps)

        os.chdir( organism + '/' + phenotype+ '/' + snps)
else:
        if not os.path.exists('keras_tryout/nest/' + organism + '/' + phenotype):
                os.makedirs('keras_tryout/nest/' + organism + '/' + phenotype)
        os.chdir('keras_tryout/nest/' + organism + '/'  + phenotype)


date_object = datetime.datetime.now().replace(second=0,microsecond=0)
print(date_object)

def coeff_determination(y_true, y_pred):
        SS_res = K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true)))
        return (1-SS_res/SS_tot)

def pearson_corr(y_true, y_pred):
        corr = np.corrcoef(y_true.reshape(-1), y_pred.reshape(-1))[0,1]
        return corr

def load_data(data):
        dataset = np.loadtxt(data, skiprows=1, dtype='str')
        x = dataset[: , 6:set_size+6].astype(np.int) if organism != 'Arabadopsis' else dataset[: , 6:set_size+6]/2 #Arabdopsis data is inbred to homozyotisity to be 2/0
        y = dataset[: , 5 ].astype(np.float)
        y = y.reshape(-1,1)
        #print("Performing split of raw data....")
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)
        return x, y #x_train, y_train, x_test, y_test



def baseline(x, y):
        model = LinearRegression()
        model.fit(x, y)
        return model

## Credit to Zach Moshe: http://zachmoshe.com/2017/04/03/pickling-keras-models.html
def unpack(model, training_config, weights): ##https://github.com/tensorflow/tensorflow/issues/34697 #fixes an error that the early stopping callback throws up in the nested cv #something about the parralele fitting step needing everything to be pickle-able and the callback isnt
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__


def make_param_box_plot(goal_dict, time_dict, analysis, stability_dict=None): #example goal dict = {'alpha' : {0.1 : [0.3, 0.5, 0.4], 1 : [0, 0.1, 0.2]}, 'beta' : {0.1 : [0.5, 0.5, 0.45, 1 : [0.8, 0.7, 0.7]}}
        metric = 'pearson_corr' if binary == 'False' else 'AUC' #binary should be accesible from outside of function
        goal_dict2 = goal_dict
        time_dict2 = time_dict
        for run in range(1,5):
                goal_dict = goal_dict2[i]
                time_dict = time_dict2[i]
                if 'max_depth' in goal_dict.keys():
                        if None in goal_dict['max_depth'].keys():
                                old_key = None #thros up sorting error
                                new_key = 0
                                goal_dict['max_depth'][new_key] = goal_dict['max_depth'].pop(old_key)
                                time_dict['max_depth'][new_key] = time_dict['max_depth'].pop(old_key)
                                stability_dict['max_depth'][new_key] = stability_dict['max_depth'].pop(old_key)
                for param in goal_dict:
                        for value in goal_dict[param]:
                                goal_dict[param][value] = [0 if score < 0 else score for score in goal_dict[param][value]] #convert negative r2 to zeros
                for param in goal_dict:
                        #ordered_dict_items = {k:goal_dict[param][k] for k in sorted(goal_dict[param].keys())} this doesnt work in python3.5 for some reason (does work in 3.8)
                        #ordered_time_items = {k:time_dict[param][k] for k in sorted(time_dict[param].keys())}
                        sorted_dict_items = sorted(goal_dict[param].items(), key=operator.itemgetter(0))#in order python It is not possible to sort a dictionary, only to get a representation of a dictionary that is sorted
                        sorted_time_items = sorted(time_dict[param].items(), key=operator.itemgetter(0))
                        ordered_dict_items = collections.OrderedDict(sorted_dict_items) #turn back into dictionary
                        ordered_time_items = collections.OrderedDict(sorted_time_items)
                        plt.subplots(1,2,figsize=(12,8))
                        plt.subplot(121) #sorted
                        plt.boxplot(ordered_dict_items.values(), bootstrap=None,showmeans=False, meanline=False, notch=True,labels=ordered_dict_items.keys()) #orange line is median, green dotted line is mean
                        plt.xlabel(str(param).upper(), fontsize=10, fontweight='bold')
                        plt.ylabel(metric, fontsize=10,fontweight='bold')
                        plt.title('%s Score vs %s' % (metric, param), fontsize=14, fontweight='bold')
                        if param == 'initialization':
                                plt.xticks(fontsize=6)
                        plt.subplot(122)
                        plt.boxplot(ordered_time_items.values(), bootstrap=None,showmeans=False, meanline=False, notch=False,labels=ordered_time_items.keys())
                        plt.xlabel(str(param).upper(), fontsize=10, fontweight='bold')
                        plt.ylabel('Training Time', fontsize=10,fontweight='bold')
                        plt.title('Training Time vs %s' % param, fontsize=14, fontweight='bold')
                        plt.tight_layout(pad=4)
                        if param == 'initialization':
                                plt.xticks(fontsize=6)
                        my_fig_name = "plots_of_" +str(analysis) + '_' + str(param) + '_' + str("{:%Y_%m_%d}".format(datetime.datetime.now())) + '_' +str(snps) +str(num)+ ".png"
                        plt.savefig(my_fig_name, dpi=300)
                        plt.show()
                        plt.clf()
                        plt.close()
                if stability_dict is not None:
                        for param in stability_dict:
                                sorted_stability_items = sorted(stability_dict[param].items(), key=operator.itemgetter(0))
                                ordered_stability_items = collections.OrderedDict(sorted_stability_items)
                                plt.boxplot(ordered_stability_items.values(), bootstrap=None,showmeans=False, meanline=False, notch=True,labels=ordered_stability_items.keys())
                                plt.xlabel(str(param).upper(), fontsize=10, fontweight='bold')
                                plt.ylabel('Delta Train-Test %s' % metric, fontsize=10,fontweight='bold')
                                plt.title('Stability Score vs %s' % param, fontsize=14, fontweight='bold')
                                if param == 'initialization':
                                        plt.xticks(fontsize=6)
                                my_fig_name = "stability_plot_of_" + str(run) + '_' + str(analysis) + '_' + str(param) + '_' + str("{:%Y_%m_%d}".format(datetime.datetime.now())) + '_' +str(snps) +str(num)+ ".png"
                                plt.savefig(my_fig_name, dpi=300)
                                plt.show()
                                plt.clf()
                                plt.close()


def make_goal_dict(whole_dict):
        print(whole_dict)
        goal_dict = {key:{} for key in whole_dict}
        for key in whole_dict:
                for item in whole_dict[key]:
                        goal_dict[key][item] = []
        time_dict = {key:{} for key in whole_dict} #both empty
        for key in whole_dict:
                for item in whole_dict[key]:
                        time_dict[key][item] = []
        return goal_dict, time_dict

x_train, y_train = load_data(data)
name_list = np.loadtxt(data, skiprows=1, usecols=(0,), dtype='str')

scaler = preprocessing.StandardScaler().fit(y_train)
#pickle.dump(scaler, open('scaler.pkl', 'wb'))
#scaler = pickle.load(open('scaler.pkl', 'rb'))

y_train = scaler.transform(y_train)

n_snps = x_train.shape[1]
metric_in_use = pearson_corr if binary == 'False' else sklearn.metrics.roc_auc_score
#################################################SVM####SVM#####SVM####################################################################
def ncv_results(analysis, ncv_object):
        print("Best Params of %s is %s " % (analysis, ncv_object.best_params))
        print("Outer scores of %s is %s and mean is %s" % (analysis, ncv_object.outer_scores, np.mean(ncv_object.outer_scores)))
        print("Outer scores (AUC probs if available) of %s is %s and mean is %s" % (analysis, ncv_object.outer_scores2, np.mean(ncv_object.outer_scores2)))
        print("Variance of %s is %s " % (analysis, ncv_object.variance))
        #print("Goal dict of %s is %s " % (analysis, ncv_object.goal_dict))
        make_param_box_plot(ncv_object.goal_dict, ncv_object.time_dict, str(analysis), stability_dict=ncv_object.stability_dict)
        with open('NCV_' + str(analysis) + '_' +  str(snps) + '_' + str(phenotype) + '_' + str(num) + '.pkl', 'wb') as ncvfile: #with open("fname.pkl", 'rb') as ncvfile:
                pickle.dump(ncv_object, ncvfile) #ncv_object = pickle.load(ncvfile)

def nn_results(analysis, ncv_object):
        print("Best Params of %s is %s " % (analysis, ncv_object.best_params))
        print("Outer scores of %s is %s and mean is %s" % (analysis, ncv_object.outer_scores, np.mean(ncv_object.outer_scores)))
        print("Outer scores (AUC probs if available) of %s is %s and mean is %s" % (analysis, ncv_object.outer_scores2, np.mean(ncv_object.outer_scores2)))
        print("Variance of %s is %s " % (analysis, ncv_object.variance))
        #print("Goal dict of %s is %s " % (analysis, ncv_object.goal_dict))
        make_param_box_plot(ncv_object.goal_dict, ncv_object.time_dict, str(analysis),stability_dict=ncv_object.stability_dict)
        nn_list = [ncv_object.best_inner_params_list, ncv_object.best_inner_score_list, ncv_object.best_params, ncv_object.metric, ncv_object.outer_scores, ncv_object.variance]
        with open('NCV_' + str(analysis) + '_' +  str(snps) + '_' + str(phenotype) + '_' + str(num) + '.pkl', 'wb') as ncvfile:
                pickle.dump(nn_list, ncvfile) #ncv_object = pickle.load(ncvfile)
        ncv_object.model.model.save("model_" + str(analysis) + '_' +  str(snps) + '_' + str(phenotype) + '_' + str(num) + ".h5")

print("Performing SVM")
c_param = [2e-2,2e-4,2e-8, 1,int(2e+2),int(2e+4),int(2e+8)] #can be negative #We found that trying exponentially growing sequences of C and γ is a practical method to identify good parameters https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
rbf_c = [2e-2,2e-4,2e-8, 1,int(2e+2),int(2e+4),int(2e+8)]
gamma_param = [0.002,0.2,0.5,0.01] #ValueError: gamma < 0
epsilon_param = [2e-5,2e-3,1,0]
loss_param = ['epsilon_insensitive', 'squared_epsilon_insensitive']
kernel_param = ['rbf', 'sigmoid'] #precompuited leads to square matrix error #temorarily removing poly for time reasons need to put it back in
tolerance=[1e-3,1e-5,1e-1]
shrinking=[True,False]
cache_size=[100,200,400]#Specify the size of the kernel cache (in MB).
degree = [1,2,3,0.1,100]
for i in range(1,5):
        globals()["svm_random_grid_"+str(i)] = {'gamma':gamma_param, 'C':rbf_c,'kernel':kernel_param, "degree":degree, 'epsilon':epsilon_param, "shrinking":shrinking,"tol":tolerance,"cache_size":cache_size}
        print(globals()["svm_random_grid_"+str(i)])
        globals()["svm_random_grid2_"+str(i)] = {'C' : c_param, 'loss':loss_param, 'epsilon':epsilon_param}
        print(globals()["svm_random_grid2_"+str(i)])

svm_random_grid = {1:svm_param_grid_1, 2:svm_param_grid_2, 3:svm_param_grid_3, 4:svm_param_grid_4}
svm_random_grid2 = {1:svm_param_grid2_1, 2:svm_param_grid2_2, 3:svm_param_grid2_3, 4:svm_param_grid2_4}
svm_goal_dict = {}
svm_time_dict = {}
rbg_goal_dict = {}
rbg_time_dict = {}
for i in range(1,5):
        globals()["svm_goal_dict"+str(i)], globals()["svm_time_dict"+str(i)] = make_goal_dict(globals()["svm_param_grid2_"+str(i)])
        svm_goal_dict[i] = globals()["svm_goal_dict"+str(i)]
        svm_time_dict[i] = globals()["svm_time_dict"+str(i)]
        globals()["rbg_goal_dict"+str(i)], globals()["rbg_time_dict"+str(i)] = make_goal_dict(globals()["svm_param_grid_"+str(i)])
        rbg_goal_dict[i] = globals()["svm_goal_dict"+str(i)]
        rbg_time_dict[i] = globals()["svm_time_dict"+str(i)]


SVM_NCV = NestedCV(model_name='LinearSVR', name_list=name_list, num=num, model=LinearSVR(), goal_dict=svm_goal_dict, time_dict=svm_time_dict, params_grid=svm_random_grid2, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
SVM_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='SVM', goal_dict=svm_goal_dict, time_dict=svm_time_dict)
ncv_results('SVM', SVM_NCV)

if binary == 'False' :
        print("Performing RBG")
        RBG_NCV = NestedCV(model_name='RBG', name_list=name_list, num=num, model=SVR(),  goal_dict=rbg_goal_dict, time_dict=rbg_time_dict,params_grid=svm_random_grid, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
        RBG_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='RBG', goal_dict=rbg_goal_dict, time_dict=rbg_time_dict)
        ncv_results('RBG', RBG_NCV)

print("Performing LASSO")
alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, -1, -10, -100]
max_iter=[1000,3000]
ridge_alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, -1, -10, -100]
tolerance=[1e-3,1e-5,1e-1]
selection=['cyclic','random']# default=’cyclic’
for i in range(1,5):
        globals()["alpha_dict_"+str(i)] = {'alpha':alpha,"max_iter":max_iter, "tol":tolerance, "selection":selection}
        print(globals()["alpha_dict_"+str(i)])
        globals()["ridge_alpha_dict_"+str(i)] = {'alpha':ridge_alpha, "tol":tolerance}
        print(globals()["ridge_alpha_dict_"+str(i)])

alpha_dict = {1:alpha_dict_1, 2:alpha_dict_2, 3:alpha_dict_3, 4:alpha_dict_4}
ridge_alpha_dict = {1:ridge_alpha_dict_1, 2:ridge_alpha_dict_2, 3:ridge_alpha_dict_3, 4:ridge_alpha_dict_4}

lass_goal_dict = {}
lass_time_dict = {}
for i in range(1,5):
        globals()["lass_goal_dict"+str(i)], globals()["lass_time_dict"+str(i)] = make_goal_dict(globals()["alpha_dict"+str(i)])
        lass_goal_dict[i] = globals()["lass_goal_dict"+str(i)]
        lass_time_dict[i] = globals()["lass_time_dict"+str(i)]


print(alpha_dict)
alpha_name_dict = {'alpha':"Alpha"}
if binary == 'False' :
        LASS_NCV = NestedCV(model_name='LASS', name_list=name_list, num=num, model=Lasso(), goal_dict=lass_goal_dict, time_dict=lass_time_dict, params_grid=alpha_dict, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
        LASS_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='LASS', goal_dict=lass_goal_dict, time_dict=lass_time_dict)
        ncv_results('LASS', LASS_NCV)

print("Performing Ridge")
lass_goal_dict = {}
lass_time_dict = {}
for i in range(1,5):
        globals()["lass_goal_dict"+str(i)], globals()["lass_time_dict"+str(i)] = make_goal_dict(globals()["ridge_alpha_dict"+str(i)])
        lass_goal_dict[i] = globals()["lass_goal_dict"+str(i)]
        lass_time_dict[i] = globals()["lass_time_dict"+str(i)]

if binary == 'True':
        RIDGE_NCV = NestedCV(model_name='RIDGE', name_list=name_list, num=num, model=RidgeClassifier(), goal_dict=lass_goal_dict, time_dict=lass_time_dict, params_grid=ridge_alpha_dict, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
else:
        RIDGE_NCV = NestedCV(model_name='RIDGE', name_list=name_list, num=num, model=Ridge(), goal_dict=lass_goal_dict, time_dict=lass_time_dict, params_grid=ridge_alpha_dict, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
RIDGE_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='RIDGE', goal_dict=lass_goal_dict, time_dict=lass_time_dict)
ncv_results('RIDGE', RIDGE_NCV)

print("Performing Random Forests")
n_estimators = [10,100,1000] # Number of features to consider at every split
max_features = ['sqrt', 'log2'] # Maximum number of levels in tree
max_depth = [1, 10, 50,100]
max_depth.append(None) # Minimum number of samples required to split a node
#min_samples_split = [int(x) for x in np.linspace(2, 2000, num = 100)]; min_samples_split.extend((5,10,20))
min_samples_split = [2, 10, 100, 1000] # Minimum number of samples required at each leaf node
#min_samples_leaf = [int(x) for x in np.linspace(1, 2000, num = 200)] ; min_samples_leaf.extend((2,4,8,16, 32, 64)) # Method of selecting samples for training each tree
min_samples_leaf = [1,2, 10, 100, 1000]
bootstrap = [False, True]
max_leaf_nodes = [10, 100, 500] #; max_leaf_nodes.append(x_train.shape[0])
max_samples = [0.5, 0.9, 0.1, 0.01]
#{'max_depth': 46, 'max_leaf_nodes': 695, 'n_estimators': 2778, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'min_samples_split': 2, 'bootstrap': False, 'max_samples': 0.5}
for i in range(1,5):
        globals()["random_grid_"+str(i)] = {'n_estimators': n_estimators,'max_features': max_features,'max_depth': max_depth,'min_samples_split': min_samples_split,'min_samples_leaf': min_samples_leaf,'bootstrap': bootstrap, 'max_samples':max_samples, 'max_leaf_nodes':max_leaf_nodes}
        print(globals()["random_grid_"+str(i)])

random_grid = {1:random_grid_1, 2:random_grid_2, 3:random_grid_3, 4:random_grid_4}
rf_goal_dict = {}
rf_time_dict = {}
for i in range(1,5):
        globals()["rf_goal_dict"+str(i)], globals()["rf_time_dict"+str(i)] = make_goal_dict(globals()["random_grid"+str(i)])
        rf_goal_dict[i] = globals()["rf_goal_dict"+str(i)]
        rf_time_dict[i] = globals()["rf_time_dict"+str(i)]

print(random_grid)
rf_name_dict = {"max_samples":"Maximum Fraction of Samples", "max_leaf_nodes":"Maximum Leaf Nodes", "n_estimators":"Number of Estimators", "n_snps":"Number of SNPs","max_features":"Maximum Number of Features", "max_depth":"Maximum Depth", "min_samples_split":"Minimum Number of Samples to Split", "min_samples_leaf":"Minimum Number of Samples in Leaf"}
rf_param_dict = {'n_snps':'n_features', 'n_estimators':'n_estimators'}
rf_param_list = ['n_estimators','max_features','max_depth','min_samples_split','min_samples_leaf','max_leaf_nodes', 'max_samples'] #dont have bootstrap here
if binary == 'True':
        RF_NCV = NestedCV(model_name='RF', name_list=name_list, num=num, model=RandomForestClassifier(), goal_dict=rf_goal_dict, time_dict=rf_time_dict, params_grid=random_grid, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
else:
        RF_NCV = NestedCV(model_name='RF', name_list=name_list, num=num, model=RandomForestRegressor(), goal_dict=rf_goal_dict, time_dict=rf_time_dict, params_grid=random_grid, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
RF_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='RF', goal_dict=rf_goal_dict, time_dict=rf_time_dict)
ncv_results('RF', RF_NCV)
#base_grid = {"fit_intercept":["True"]}


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))) #If a TensorFlow operation has both CPU and GPU implementations, by default the GPU will be used by default.
print("Performing Neural Network")
for i in range(1,5):
        globals()["param_grid_"+str(i)] = {'max_norm_reg':[True, False], 'decay_rate':[0.75, 0.5, 0.9], 'lr_schedule':[True, False], 'network_shape':['brick', 'funnel','long_funnel'], 'epochs' : [50,100,200],'batch_size' : [16,32, 128],'learning_rate' : [0.01, 0.001, 0.0001, 0.00001],'HP_L1_REG' : [1e-5,1e-6,1e-4, 1e-2, 0.1, 1e-3],'HP_L2_REG' : [1e-8, 1e-3, 1e-1, float(0)], 'kernel_initializer' : ['glorot_uniform', 'glorot_normal', 'random_normal', 'random_uniform', 'he_uniform', 'he_normal'],'activation' : ['tanh', 'relu', 'elu'],'HP_NUM_HIDDEN_LAYERS' : [2,3,5],'units' : [200, 100,1000], 'rate' : [float(0), 0.1, 0.3],'HP_OPTIMIZER' : ['Ftrl', 'RMSprop', 'Adadelta', 'Adamax', 'Adam', 'Adagrad', 'SGD']}
        print(globals()["param_grid_"+str(i)])

param_grid = {1:param_grid_1, 2:param_grid_2, 3:param_grid_3, 4:param_grid_4}
nn_goal_dict = {}
nn_time_dict = {}
for i in range(1,5):
        globals()["nn_goal_dict"+str(i)], globals()["nn_time_dict"+str(i)] = make_goal_dict(globals()["param_grid"+str(i)])
        nn_goal_dict[i] = globals()["nn_goal_dict"+str(i)]
        nn_time_dict[i] = globals()["nn_time_dict"+str(i)]


METRIC_ACCURACY = pearson_corr
dependencies = {'pearson_corr':pearson_corr}
custom_objects = {"pearson_corr":pearson_corr}
tf.config.threading.set_inter_op_parallelism_threads(32)
tf.config.threading.set_intra_op_parallelism_threads(32)
#tf.config.experimental_run_functions_eagerly(True) #needed to avoid error # tensorflow.python.eager.core._SymbolicException

callback = tf.keras.callbacks.EarlyStopping(monitor='pearson_corr', patience=20, mode='max', baseline=0.0) #min above 0 #this callkaci is throwing up and error Unknown metric function

if binary == 'True': #overwrite variables
        dependencies = {'auc':tf.keras.metrics.AUC}
        METRIC_ACCURACY = tf.keras.metrics.AUC
        callback = tf.keras.callbacks.EarlyStopping(monitor='auc', patience=20, mode='max', baseline=0.0) #min above 0


import tempfile
#https://github.com/tensorflow/tensorflow/issues/34697 #fixes an error that the early stopping callback throws up in the nested cv #something about the parralele fitting step needing everything to be pickle-able and the callback isnt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

make_keras_picklable()

def build_nn(input_shape, HP_OPTIMIZER, HP_NUM_HIDDEN_LAYERS, units, max_norm_reg, decay_rate, lr_schedule, activation, learning_rate, HP_L1_REG, HP_L2_REG, rate, kernel_initializer, network_shape):
        opt = HP_OPTIMIZER
        chosen_opt = getattr(tf.keras.optimizers,opt)
        if lr_schedule == True:
                learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,decay_steps=100000,decay_rate=decay_rate, staircase=True)
        if max_norm_reg == True:
                max_norm = tf.keras.constraints.max_norm(1.)
        else:
                max_norm = None
        reg = tf.keras.regularizers.l1_l2(l1=HP_L1_REG, l2=HP_L2_REG)
        long_funnel_count = 0 #keep widest shape for two layers
        model = Sequential()
        #input_shape = (x_train.shape[1]-1,) if snps == 'shuf' else (x_train.shape[1]-1,)
        model.add(Dense(units=units, kernel_constraint=max_norm, activation=activation, kernel_regularizer=reg, input_shape=input_shape))
        if rate != 0:
                model.add(Dropout(rate=rate))
        for i in range(HP_NUM_HIDDEN_LAYERS-1):
                if network_shape == 'funnel':
                        units = int(units*0.666)
                elif network_shape == 'long_funnel':
                        if long_funnel_count >= 1: #two wide layers (inclduing previous first layer)
                                units=int(units*0.666)
                        long_funnel_count += 1
                model.add(Dense(units=units, activation=activation, kernel_constraint=max_norm, kernel_regularizer=reg, kernel_initializer=kernel_initializer))
                if rate != 0:
                        model.add(Dropout(rate=rate))
        if binary == 'True' :
                model.add(Dense(1, activation='sigmoid', kernel_constraint=max_norm, kernel_regularizer=reg, kernel_initializer=kernel_initializer))
                model.compile(loss='binary_crossentropy',metrics=['accuracy', AUC(name='auc')],optimizer=chosen_opt(learning_rate=learning_rate))
        else:
                model.add(Dense(1, activation='linear', kernel_constraint=max_norm, kernel_regularizer=reg, kernel_initializer=kernel_initializer))
                model.compile(loss='mean_absolute_error',metrics=['accuracy', 'mae', pearson_corr],optimizer=chosen_opt(learning_rate=learning_rate))
        print(model.summary())
        return model


if binary == 'True':
        nn_model = KerasClassifier(build_fn = build_nn, verbose=0, callbacks=[callback])
else:
        nn_model = KerasRegressor(build_fn = build_nn, verbose=0, callbacks=[callback])


NN_NCV = NestedCV(model_name='nn_model', name_list=name_list, num=num, model=nn_model, goal_dict=nn_goal_dict, time_dict=nn_time_dict, params_grid=param_grid, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
NN_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='NN', goal_dict=nn_goal_dict, time_dict=nn_time_dict)
nn_results('NN', NN_NCV)
print("Performing a convulutional neural network")
for i in range(1,5):
        globals()["cnn_param_grid_"+str(i)] = {'decay_rate':[0.75, 0.5, 0.9], 'max_norm_reg':[True, False], 'lr_schedule':[True, False], 'network_shape':['brick', 'funnel','long_funnel'], 'epochs':[100, 50],'batch_size' : [16,64,128], 'learning_rate' : [0.01, 0.0001, 0.001],'HP_L1_REG' : [0.001, 0.0001,0.00001,0],'HP_L2_REG' : [0, 0.001,0.00001],'kernel_initializer' : ['glorot_normal', 'glorot_uniform', 'he_uniform', 'random_normal', 'random_uniform', 'he_normal'],'activation' : ['tanh', 'relu', 'elu'],'HP_NUM_HIDDEN_LAYERS' : [2,3, 5],'units' : [100,200,1000], 'rate' : [float(0), 0.1, 0.5],'HP_OPTIMIZER' : ['SGD','Ftrl', 'RMSprop', 'Adadelta', 'Adamax', 'Adam', 'Adagrad'], 'filters':[1,5],'strides':[1,2,3],'pool':[1,2,3],'kernel':[1,2,3]}
        print(globals()["cnn_param_grid_"+str(i)])

param_grid = {1:cnn_param_grid_1, 2:cnn_param_grid_2, 3:cnn_param_grid_3, 4:cnn_param_grid_4}
cnn_goal_dict = {}
cnn_time_dict = {}
for i in range(1,5):
        globals()["cnn_goal_dict"+str(i)], globals()["cnn_time_dict"+str(i)] = make_goal_dict(globals()["cnn_param_grid"+str(i)])
        cnn_goal_dict[i] = globals()["cnn_goal_dict"+str(i)]
        cnn_time_dict[i] = globals()["cnn_time_dict"+str(i)]


if binary == 'True':
        METRIC_ACCURACY = tf.metrics.AUC
else:
        METRIC_ACCURACY = 'pearson_corr'

print(x_train.shape)
#K.set_image_dim_ordering('th') #Negative dimension size caused by subtracting 2 from 1 for 'MaxPool - fixes error
K.set_image_data_format('channels_first') #prevents error
def conv_model(input_shape, HP_OPTIMIZER, HP_NUM_HIDDEN_LAYERS, lr_schedule, decay_rate, units, activation, learning_rate, HP_L1_REG, HP_L2_REG, max_norm_reg, rate, kernel_initializer,strides,pool,filters,kernel, network_shape):
        opt = HP_OPTIMIZER
        if HP_NUM_HIDDEN_LAYERS == 1 :
                print("HP_NUM_HIDDEN_LAYERS is equal to 1; this could cause building problems")
        chosen_opt = getattr(tf.keras.optimizers,opt)
        if lr_schedule == True:
                learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,decay_steps=100000,decay_rate=decay_rate, staircase=True)
        if max_norm_reg == True:
                max_norm = tf.keras.constraints.max_norm(1.)
        else:
                max_norm = None
        reg = tf.keras.regularizers.l1_l2(l1=HP_L1_REG, l2=HP_L2_REG)
        long_funnel_count = 0 #keep widest shape for two layers
        input_shape = (input_shape[0],1)
        #input_shape = (x_train.shape[1]-1,1) if snps == 'shuf' else (x_train.shape[1]-1,1)
        model = Sequential() # Only use dropout on fully-connected layers, and implement batch normalization between convolutions.
        model.add(Conv1D(filters=filters, strides=strides, input_shape=input_shape,  padding='same',data_format='channels_last', activation=activation, kernel_regularizer=reg, kernel_initializer=kernel_initializer, kernel_size=kernel))
        model.add(tf.keras.layers.MaxPool1D(pool_size=pool, strides=strides,padding='same',data_format='channels_last'))
        for i in range(HP_NUM_HIDDEN_LAYERS-1):
                if network_shape == 'funnel':
                        units = int(units*0.666)
                elif network_shape == 'long_funnel':
                        if long_funnel_count >= 1: #two wide layers (inclduing previous first layer)
                                units=int(units*0.666)
                        long_funnel_count += 1
                model.add(Conv1D(filters=filters, strides=strides, activation=activation,  padding='same',data_format='channels_last', kernel_regularizer=reg, kernel_initializer=kernel_initializer, kernel_size=kernel))
                model.add(tf.keras.layers.MaxPool1D(pool_size=pool, strides=strides,padding='same', data_format='channels_last'))
        model.add(Flatten())
        if binary == 'True':
                model.add(Dense(1, activation='sigmoid', kernel_constraint=max_norm, kernel_regularizer=reg, kernel_initializer=kernel_initializer))
                model.compile(loss='binary_crossentropy',metrics=['accuracy', AUC(name='auc')],optimizer=chosen_opt(learning_rate=learning_rate))
        else:
                model.add(Dense(1, activation='linear', kernel_constraint=max_norm, kernel_regularizer=reg, kernel_initializer=kernel_initializer))
                model.compile(loss='mean_absolute_error',metrics=['accuracy', 'mae', pearson_corr],optimizer=chosen_opt(learning_rate=learning_rate))
        print("Summary ", model.summary())
        return model


cnn_model = KerasRegressor(build_fn = conv_model,verbose=0, callbacks=[callback]) if binary == 'False' else KerasClassifier(build_fn = conv_model,verbose=0, callbacks=[callback])
CNN_NCV = NestedCV(model_name='CNN', name_list=name_list, num=num,model=cnn_model, goal_dict=cnn_goal_dict, time_dict=cnn_time_dict, params_grid=cnn_param_grid, outer_kfolds=4, inner_kfolds=4, n_jobs = 32,cv_options={'predict_proba':False,'randomized_search':True, 'randomized_search_iter':iterations, 'sqrt_of_score':False,'recursive_feature_elimination':False, 'metric':metric_in_use, 'metric_score_indicator_lower':False})
CNN_NCV.fit(x_train, y_train.ravel(), name_list=name_list, num=num, phenfile=phenfile, set_size=set_size, snps=snps, organism=organism, model_name='CNN', goal_dict=cnn_goal_dict, time_dict=cnn_time_dict)
nn_results('CNN', CNN_NCV)
