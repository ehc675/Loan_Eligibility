import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

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
	model = BalancedRandomForestClassifier(n_estimators = 100, random_state = 420)
	model.fit(train_x, train_y)

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


if __name__ == '__main__':
	random_forest_classify("../dataset/filled_train.csv", "../dataset/filled_test.csv")