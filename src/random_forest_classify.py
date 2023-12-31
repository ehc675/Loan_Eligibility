import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import joblib

def cross_validation(train_file, test_file):
	# load train and test datasets
	train_df = pd.read_csv(train_file)
	test_df = pd.read_csv(test_file)

	train_x = train_df.drop("denial_reason_1", axis = 1)
	train_y = train_df["denial_reason_1"]
	test_x = test_df.drop("denial_reason_1", axis = 1)
	test_y = test_df["denial_reason_1"]

	cv_scores = []

	# Initialize a RandomForestClassifier
	model = RandomForestClassifier(random_state = 420)
	n_estimators_range = [50, 75, 100, 125, 150, 175, 200]
	for i in n_estimators_range:
		model.n_estimators = i
		scores = cross_val_score(model, train_x, train_y, cv = 2, scoring = 'accuracy')
		cv_scores.append(scores.mean())

	best_n_estimators = n_estimators_range[cv_scores.index(max(cv_scores))]

	print("Best n_estimators:", best_n_estimators)
	# The best is 175




def random_forest_classify(train_file, test_file):
	"""
	take in a train file path and a test file path,
	construct a standard random forest model,
	and evaluate the performance of the model
	"""

	# load train and test datasets
	train_df = pd.read_csv(train_file)
	test_df = pd.read_csv(test_file)

	train_x = train_df.drop("denial_reason_1", axis = 1)
	train_y = train_df["denial_reason_1"]
	test_x = test_df.drop("denial_reason_1", axis = 1)
	test_y = test_df["denial_reason_1"]

	# Initialize a RandomForestClassifier
	model = RandomForestClassifier(n_estimators = 1, random_state = 420)
	model.fit(train_x, train_y)

	# save the model to current directory
	joblib.dump(model, 'random_forest_model.pkl')


	# evaluate model's performance on train dataset
	train_predictions = model.predict(train_x)
	train_accuracy = accuracy_score(train_y, train_predictions)
	train_precision = precision_score(train_y, train_predictions, average = "weighted")
	train_recall = recall_score(train_y, train_predictions, average = "weighted")
	train_f1 = f1_score(train_y, train_predictions, average = "weighted")

	# evaluate model's performance on test dataset
	test_predictions = model.predict(test_x)
	test_accuracy = accuracy_score(test_y, test_predictions)
	test_precision = precision_score(test_y, test_predictions, average = "weighted")
	test_recall = recall_score(test_y, test_predictions, average = "weighted")
	test_f1 = f1_score(test_y, test_predictions, average = "weighted")

	print("Train Set Performance:")
	print("Train accuracy: {:.4f}".format(train_accuracy))
	print("Train precision: {:.4f}".format(train_precision))
	print("Train recall: {:.4f}".format(train_recall))
	print("Train f1: {:.4f}".format(train_f1))
	print("\n")

	print("Test Set Performance:")
	print("Test accuracy: {:.4f}".format(test_accuracy))
	print("Test precision: {:.4f}".format(test_precision))
	print("Test recall: {:.4f}".format(test_recall))
	print("Test f1: {:.4f}".format(test_f1))

	# plot out the feature importance graph
	feature_importances = model.feature_importances_
	feature_names = train_x.columns
	plt.figure(figsize=(10, 6))
	plt.barh(range(len(feature_importances)), feature_importances, align='center')
	plt.yticks(range(len(feature_importances)), feature_names)
	plt.xlabel('Feature Importance')
	plt.ylabel('Feature')
	plt.title('Random Forest Feature Importance')
	plt.show()

def inference(input):
	model = joblib.load('random_forest_model.pkl')
	predictions = model.predict(input)
	print(predictions)



if __name__ == '__main__':
	# cross_validation("../dataset/filled_train.csv", "../dataset/filled_test.csv")
	random_forest_classify("../dataset/filled_train.csv", "../dataset/filled_test.csv")
	# inference([[3, 4, 2, 1, 1, 1, 3000000000, 5, 240, 400000, 1, 6, 5]])
	inference([[3, 4, 5, 1, 1, 32, 0, 4.95, 0, 0, 1, 4, 0]])