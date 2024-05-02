# Wine Prediction Model using Spark on AWS EMR

## Problem Statement

Develop a wine quality prediction machine learning model using Spark on AWS Elastic MapReduce (EMR). The project involves:

1. Training the model in parallel on multiple EC2 instances.
2. Deploying it to predict wine quality in an application running on a single EC2 instance.
3. Building a Docker container for the prediction application.

## Setting up the Cloud Environment

### Launch Amazon EMR Cluster

To set up the Amazon EMR, follow the step-by-step instructions on [AWS documentation](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-gs.html).

- Specify 1 for the master node and 3 for the core/task nodes to have a total of four instances.
- Review and launch the cluster.

### Set up Security Groups

1. Go to the EC2 dashboard, navigate to "Security Groups" under the "Network & Security" section.
2. Create a new security group, and configure inbound rules to allow SSH traffic from your IP address or a specific IP range.
3. Apply this security group to the EMR master and task nodes.

### SSH Access

1. Use an SSH client to connect to the master node of your EMR cluster.
2. Retrieve the public DNS name of the master node from the EMR dashboard.
3. Open a terminal or command prompt and use the SSH command to connect to the master node: `ssh -i <path_to_private_key_file> hadoop@<public_dns_name>`

## Running the Model Training

### Install Libraries on EMR Master Node

1. Connect to the master node of your Amazon EMR cluster using SSH.
2. Install the required Python libraries such as pandas, scikit-learn, and pyspark, findspark using pip.
3. Upload Training Dataset to Amazon S3 and make a note of the S3 URI.
4. Transfer the file containing your ML model to the master node of the EMR cluster using SCP.
5. Run Training Script on EMR Master Node: `spark-submit training.py`.
6. Save the Trained Model to Amazon S3.
7. Terminate the EMR cluster using the AWS Management Console.

## Running the Prediction App on a Single EC2 Instance

### Launch an EC2 Instance

1. Log in to the AWS Management Console.
2. Navigate to the EC2 dashboard.
3. Click on "Launch Instance" to start the instance launch wizard.
4. Choose an Amazon Machine Image (AMI).
5. Configure instance details such as network settings, storage (at least 30GB), and security groups.
6. Review and launch the instance.

### Connect to the EC2 Instance

1. Once the instance is running, obtain the public IP address of the instance from the EC2 dashboard.
2. Use an SSH client to connect to the instance.
3. Download your prediction app file and other files like the training model and Validation Dataset saved on S3 using the `cp` command (`aws s3 cp s3://cs643-njit-bucket/data/ValidationDataset.csv .`).
4. Install Python packages (numpy, pandas, scikit-learn, pyspark, findspark).
5. Run the Prediction App: Navigate to the directory where your prediction app files are located. Run the prediction app using the python command: `spark-submit app.py <path_to_csv_file>`.

## Running the Prediction App with Docker

1. Create a directory on your EC2 instance and copy all the files that you will need to run your prediction app (trained model, validation dataset, and the app script).
2. Inside the same directory create a file named Dockerfile: Write a Dockerfile that specifies the environment and dependencies required to run the prediction app.
3. Build Docker Image: Navigate to the directory containing your app.py file and the Dockerfile, then run the following command to build the Docker image: `docker build -t wineprediction`.
4. Run Docker Container: Once the Docker image is built, you can run it in a Docker container using the following command: `docker run wineprediction`.
5. Push the docker container in Docker Hub.
