from pyspark.sql import SparkSession
from pyspark.ml.feature import IDF
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import CountVectorizer
from string import ascii_letters
from pyspark.sql.functions import lower, trim, regexp_replace


class MLSignalPipeline:
    def __init__(self, filename):
        self.spark = SparkSession.builder.getOrCreate()
        self.df = self.load_data(filename)

    def load_data(self, filepath):
        """
        Load csv file into PySpark DataFrame given file path.
        :param filepath: str
        :return df: PySpark DataFrame
        """
        df = self.spark.read.csv("Petitions_dataset_2018_2019.csv", sep=",", inferSchema=True, header=True)

        # TODO: Clean up punctuation and whitespace from text
        # columns = df.schema.names
        # for column_name in columns:
        #    df = df.select(lower(trim(regexp_replace(column_name, '[^A-Za-z0-9 ]+', '')).alias("cleaned_" + column_name)))
        #    self.df.drop(column_name).collect()
        return df

    def tokenize_data(self):
        """
        Tokenize the text data from the PySpark DataFrame

        :return: transforms PySpark DataFrame to include tokenized text
        """
        columns = self.df.schema.names
        for column_name in columns:
            if column_name != "label":
                tokenizer = Tokenizer(inputCol=column_name, outputCol=column_name + "_words")
                self.df = tokenizer.transform(self.df)
                self.df.drop(column_name).collect()

    def vectorize_data(self):
        """
        Convert each list of tokens into vectors of token counts

        :return: vectors of token counts
        """
        columns = self.df.schema.names
        for column_name in columns:
            if "_words" in column_name:
                count = CountVectorizer(inputCol=column_name, outputCol=column_name+"_raw_features")
                model = count.fit(self.df)
                self.df = model.transform(self.df)
                self.df.drop(column_name).collect()

    def tf_idf(self):
        """
        Applies term frequency-inverse document frequency (TF-IDF) to vectorized data

        :return:
        """
        columns = self.df.schema.names
        for column_name in columns:
            if "_raw_features" in column_name:
                idf = IDF(inputCol=column_name, outputCol=column_name.replace("_raw_features", "_features"))
                idf_model = idf.fit(self.df)
                self.df = idf_model.transform(self.df)
                self.df.drop(column_name).collect()


    def split_data(self, seed):
        """
        Split dataset into training ans validation splits

        :return: train_df, test_df
        """
        train_df, test_df = self.df.randomSplit([0.8, 0.2], seed)
        return train_df, test_df

    def save_parquet(self):
        """
        Save PySpark DataFrame to parquet file for storage

        :return: locally saved parquet file
        """
        self.df.write.parquet("dataset.parquet")