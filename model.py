import os
import numpy as np
from preprocessing import MLSignalPipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


class Model:
    def __init__(self, train_data):
        self.model = self.create_model(train_data)

    def create_model(self, train_data):
        """
        Creates and trains logisitic regression model using cross-validation.

        :param train_data: PySpark DataFrame
        :return: best model from cross-validation
        """
        lr = LogisticRegression(maxIter=10)

        paramGrid_lr = ParamGridBuilder() \
            .addGrid(lr.regParam, np.linspace(0.3, 0.01, 10)) \
            .addGrid(lr.elasticNetParam, np.linspace(0.3, 0.8, 6)) \
            .build()

        crossval_lr = CrossValidator(estimator=lr,
                                     estimatorParamMaps=paramGrid_lr,
                                     evaluator=BinaryClassificationEvaluator(),
                                     numFolds=5)

        cv_model_lr = crossval_lr.fit(train_data)
        best_model_lr = cv_model_lr.bestModel
        return best_model_lr

    def predict(self, test_data):
        """
        Generates prediction given

        :param test_data: PySpark DataFrame
        :return: predictions
        """
        predictions_lr = self.model.transform(test_df)
        return predictions_lr


if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), "Petitions_dataset_2018_2019.csv")
    # Create our ML signal generation engine
    pipeline = MLSignalPipeline(file_path)

    # Let us tokenize and vectorize the text data
    pipeline.tokenize_data()
    pipeline.vectorize_data()

    # Save dataset to parquet file
    pipeline.save_parquet()

    # Split dataset into training and test splits
    train_df, test_df = pipeline.split_data(0)

    # Create and train our model using cross-validation
    model = Model(train_df)

    # Generate predictions on test set
    predictions = model.predict(test_df)
