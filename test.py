from __future__ import division
import sklearn
from collections import Counter
import numpy as np
import pandas as pd
import nltk
import ngram
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import make_scorer
import jellyfish as jf
from sklearn import linear_model

def safe_unicode(obj, *args):
    """ return the unicode representation of obj """
    try:
        return unicode(obj, *args)
    except UnicodeDecodeError:
        # obj is byte string
        ascii_text = str(obj).encode('string_escape')
        return unicode(ascii_text)

Train_data = pd.read_csv("/media/sai/New Volume1/Practice/Relevance/data/train_complete.csv");
train_X = [];
train_Y = [];
for i,row in Train_data.iterrows():
	#print i;
	search_term = row["search_term"].lower();
	#if search_term in spell_check_dict:
	#	search_term = spell_check_dict[search_term];
	product_title = row["product_title"].lower();
	product_description = row["product_description"].lower();
	search_term = search_term.split(" ");
	product_title = product_title.split(" ");
	product_description = product_description.split(" ");
	m1 = 0;
	m2 = 0;
	m3 = 0;
	m4 = 0;
	m5 = 0;
	m6 = 0;
	m7 = 0;
	m8 = 0;
	for word in search_term:
		match_pd = ngram.NGram(product_description);
		match_pd = match_pd.search(word);
		if(len(match_pd)!=0):	
			m4 = m4 + match_pd[0][1];
		match_tl = ngram.NGram(product_title);
		match_tl = match_tl.search(word);
		if(len(match_tl)!=0):	
			m5 = m5 + match_tl[0][1];
		m1_tmp = 0;
		m2_tmp = 0;
		m3_tmp = 0;	
		word = unicode(word,'utf8');
		for txt in product_description:
			txt = unicode(txt,'utf8');
			a = jf.levenshtein_distance(word,txt);
			if(a>=m1_tmp):
				m1_tmp = a;	
			a = jf.damerau_levenshtein_distance(word,txt);
			if(a>=m2_tmp):
				m2_tmp = a;
			a = jf.hamming_distance(word,txt);
			if(a>=m3_tmp):
				m3_tmp = a;
		m1 = m1 + m1_tmp;
		m2 = m2 + m2_tmp;
		m3 = m3 + m3_tmp;		
		m6_tmp = 0;
		m7_tmp = 0;
		m8_tmp = 0;	
		#word = word.decode('utf-8');
		for txt in product_title:
			txt = safe_unicode(txt);
			a = jf.levenshtein_distance(word,txt);
			if(a>=m6_tmp):
				m6_tmp = a;	
			a = jf.damerau_levenshtein_distance(word,txt);
			if(a>=m7_tmp):
				m7_tmp = a;
			a = jf.hamming_distance(word,txt);
			if(a>=m8_tmp):
				m8_tmp = a;
		m6 = m6 + m6_tmp;
		m7 = m7 + m7_tmp;
		m8 = m8 + m8_tmp;					
	m = np.zeros(8);
	m[0] = m1;
	m[1] = m2/len(search_term);
	m[2] = m3/len(search_term);
	m[3] = m4/len(search_term);
	m[4] = m5/len(search_term);
	m[5] = m6/len(search_term);
	m[6] = m7/len(search_term);
	m[7] = m8/len(search_term);
	train_X.append(m);
	train_Y.append(row["relevance"]);
	#m = m/np.linalg.norm(m);
	#print "title match:" + str(m[0]) + ",description match:" + str(m[1])+",both:"+str(m[2])+",actual_match:"+str(row["relevance"]);		

print "Data prepared"
train_X = np.vstack(train_X);
train_Y = np.vstack(train_Y).ravel();
#train_Y = 3*train_Y-1;
train_X = train_X.astype('float64');
train_Y = train_Y.astype('float64');
train_X = preprocessing.normalize(train_X,axis=0);
ss = StandardScaler();
train_X = ss.fit_transform(train_X);
(trainX, testX, trainY, testY) = train_test_split(train_X, train_Y, test_size = 0.2, random_state = 32)
#rfc = RandomForestClassifier(n_jobs=-1, oob_score = True,max_features='auto',n_estimators=100,min_samples_leaf=10);
clf = linear_model.SGDRegressor();
#rfc = GradientBoostingClassifier(max_features='auto',n_estimators=400,min_samples_leaf=10);
param_grid = {
    #'kernel': ['linear'],
    #'C': [10**i for i in range(-3, 2)],
    'alpha': [10**i for i in range(-4, 4)],
    #'degree':[1,2,3,4,5,6,7]	
}

gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 3,scoring='mean_squared_error',verbose=5)
gs.fit(trainX, trainY)
#print rfc.score(trainX,trainY)
#rfc_param = gs.best_params_;
#print "Best parameters:", gs.best_params_
print "Best score:", gs.best_score_
joblib.dump(gs.best_estimator_, '/media/sai/New Volume1/Practice/beeImages/clf_beeNonbee.pkl', compress=9) 
