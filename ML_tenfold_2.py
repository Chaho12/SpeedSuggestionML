#atmcond	accdate	injury	travspd	spdlim				
# Load libraries
import time
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing


def readCSV():
	names = ['atmcond', 'atmcond2', 'travspd', 'travlimit', 'injury/class']
	dataset = pandas.read_csv('allData_no_2.csv', names = names)
	return dataset

def main():
	elapsed = time.time()

	dataset = readCSV()
	res = machineLearning(dataset)
	with open('allData_no_2.txt', 'a') as f:
		for i in res:
			print('', i, file=f)
		print('',"Done & elapsed time : {}".format(elapsed), file=f)
		print('\n\n', file=f)
	
	elapsed = time.time() - elapsed
	print("Done & elapsed time : {}".format(elapsed))

def machineLearning(dataset):
	# Split-out validation dataset
	array = dataset.values
	X = array[:,0:4]
	Y = array[:,4]
	validation_size = 0.20
	seed = 7
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_validation= scaler.transform(X_validation)
	
	# Test options and evaluation metric
	seed = 7
	scoring = 'accuracy'

	# Spot Check Algorithms
	models = []
	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	for i in range(1,21):
		models.append(("KNN_n_neighbors {}".format(i), KNeighborsClassifier(n_neighbors=i)))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(("pol_SVM".format(i), SVC(kernel='poly')))
	for i in range(1,20):
		models.append(("linear_SVM_C_{}".format(i*0.1), SVC(kernel='linear', C = i*0.1)))
		# models.append(("poly_SVM_C_{}".format(i*0.1), SVC(kernel='poly', C = i*0.1)))
		models.append(("rbf_SVM_C_{}".format(i*0.1), SVC(kernel='rbf', C = i*0.1)))
		models.append(("sig_SVM_C_{}".format(i*0.1), SVC(kernel='sigmoid', C = i*0.1)))
	models.append(('lin_SVM', SVC(kernel='linear')))
	models.append(('rbf_SVM', SVC(kernel='rbf')))

	for i in range(1,7):
		models.append(("RFC{}".format(i*5), RandomForestClassifier(n_estimators=i*5)))

	# evaluate each model in turn
	dataStore = []
	bestName = ''
	bestMean = 0
	bestAcc = 0
	bestStd = 1
	for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle = True)
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		
		ML = model
		ML.fit(X_train, Y_train)
		predictions = ML.predict(X_validation)
		# dataStore.append(accuracy_score(Y_validation, predictions))
		# dataStore.append(confusion_matrix(Y_validation, predictions))
		# dataStore.append(classification_report(Y_validation, predictions))
		acc = accuracy_score(Y_validation, predictions)
		mean = cv_results.mean()
		std = cv_results.std()
		msg = "%s: %f with - %f (%f) " % (name, acc, mean, std)
		print(msg)
		dataStore.append(msg)
				
		if (bestAcc < acc):
			bestName = name
			bestAcc = acc
			bestMean = mean
			bestStd = std
		if (bestAcc == acc):
			if(bestMean < mean):
				bestName = name
				bestAcc = acc
				bestMean = mean
				bestStd = std
			if(bestMean == mean):
				if(bestStd > std):
					bestName = name
					bestAcc = acc
					bestMean = mean
					bestStd = std

	msg = "Best prediction accuracy: {} using {} ({}) with {}".format(bestAcc, bestMean, bestStd, bestName)
	dataStore.append(msg)
	print(msg)
	
	return dataStore

if __name__ == "__main__":
	main()
	
