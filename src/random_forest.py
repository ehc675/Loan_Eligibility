import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def random_forest_classify(train_file, test_file):
	train_df = pd.read_csv(train_file)
	test_df = pd.read_csv(test_file)

	train_x = train_df.drop("Action_taken", axis = 1)
	train_y = train_df["Action_taken"]
	test_x = test_df.drop("Action_taken", axis = 1)
	test_y = test_df["Action_taken"]

	model = RandomForestClassifier(n_estimators = 100, random_state = 42)
	model.fit(train_x, train_y)

	train_predictions = model.predict(train_x)
	train_accuracy = accuracy_score(train_y, train_predictions)
	train_precision = precision_score(train_y, train_predictions)
	train_recall = recall_score(train_y, train_predictions)
	train_f1 = f1_score(train_y, train_predictions)

	test_predictions = model.predict(test_x)
	test_accuracy = accuracy_score(test_y, test_predictions)
	test_precision = precision_score(test_y, test_predictions)
	test_recall = recall_score(test_y, test_predictions)
	test_f1 = f1_score(test_y, test_predictions)

	print("Train Set Performance:")
	print("Train accuracy: {.4f}", format(train_accuracy))
	print("Train preision: {.4f}", format(train_precision))
	print("Train recall: {.4f}", format(train_recall))
	print("Train f1: {.4f}", format(train_f1))

	print("Test Set Performance:")
	print("Test accuracy: {.4f}", format(test_accuracy))
	print("Test preision: {.4f}", format(test_precision))
	print("Test recall: {.4f}", format(test_recall))
	print("Test f1: {.4f}", format(test_f1))

	feature_importances = model.feature_importances_
	plt.figure(figsize=(10, 6))
	plt.barh(range(len(feature_importances)), feature_importances, align='center')
	plt.yticks(range(len(feature_importances)), feature_names)
	plt.xlabel('Feature Importance')
	plt.ylabel('Feature')
	plt.title('Random Forest Feature Importance')
	plt.show()