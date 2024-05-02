import sys
import numpy as np
import pandas as pd
import pyspark
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import Imputer
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, accuracy_score
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import LogisticRegressionModel


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    model_path = "/data/trained_model/"

    # Initialize SparkSession
    spark = pyspark.SparkConf().setAppName('WinePrediction')
    spark_context = pyspark.SparkContext(conf=spark)
    spark_session = SparkSession.builder.appName("WinePrediction").getOrCreate()
    
    # Load trained model
    model = LogisticRegressionModel.load(spark_context, model_path)
    print("Model loaded")

    # Load CSV data
    data = spark_session.read.csv(csv_file, header=True, sep=";")
    data.printSchema()
    data.show()

    #data preprocessing

    #Converting the data to float to ensure compatibility with the algorithms
    for col_name in data.columns[1:-1]+['""""quality"""""']:
         data = data.withColumn(col_name, col(col_name).cast('float'))

    # Handling Missing Values: Impute missing values with mean
    imputer = Imputer(inputCols=data.columns[1:], outputCols=["{}_imputed".format(c) for c in data.columns[1:]])
    imputer_model = imputer.fit(data)
    data = imputer_model.transform(data)
    

    #rename the quality column for easier access 
    data = data.withColumnRenamed('""""quality"""""', "label")

    #separating the features from the label and converting it in to numpy array 
    features =np.array(data.select(data.columns[1:-1]).collect())
    label = np.array(data.select('label').collect())

    #creating the feature vector
    VectorAssembler = VectorAssembler(inputCols = data.columns[1:-1] , outputCol = 'features')
    transformed_data = VectorAssembler.transform(data)
    transformed_data = transformed_data.select(['features','label'])


    def to_labeled_point(sc, features, labels):
       labeled_points = [LabeledPoint(label, features) for label, features in zip(labels, features)]
       return sc.parallelize(labeled_points)

    # Create RDD of LabeledPoint
    dataset = to_labeled_point(spark_context, features, label)

    # Make predictions
    predictions = model.predict(dataset.map(lambda x: x.features))

    # Getting a RDD of label and predictions
    rdd_df = dataset.map(lambda lp: lp.label).zip(predictions)

    # Convert RDD to Spark DataFrame with default column names
    rdd_to_spark_df = rdd_df.toDF()

    # Convert RDD to Spark DataFrame with specified column names
    spark_df = rdd_df.toDF(["label", "Prediction"])

    # Convert Spark DataFrame to Pandas DataFrame
    label_predicted_df = spark_df.toPandas()

    # Calculate F1 score
    f1score = f1_score(label_predicted_df['label'], label_predicted_df['Prediction'], average='micro')
    print("F1 score:", f1score)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(label_predicted_df['label'], label_predicted_df['Prediction'])
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate accuracy
    accuracy = accuracy_score(label_predicted_df['label'], label_predicted_df['Prediction'])
    print("Accuracy:", accuracy)
