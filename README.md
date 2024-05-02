# Wine Prediction Model using Spark on AWS EMR

Problem Statement: Develop a wine quality prediction machine learning model using Spark on AWS Elastic MapReduce (EMR). The project involves:
1. Training the model in parallel on multiple EC2 instances
2. Deploying it to predict wine quality in an application running on a single EC2 instance.
3. Build a Docker container for the prediction application. 

Setting up the cloud environment: 

Launch Amazon EMR Cluster:
1. To set up the Amazon EMR, follow the step by step instructions on aws documentation 
https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-gs.html
2. For the number of instances, specify 1 for the master node and 3 for the core/task 
nodes to have a total of four instances. 
3. Review and launch the cluster.
4. Set up Security Groups:
Go to the EC2 dashboard, navigate to "Security Groups" under the "Network & Security" 
Section. 
Create a new security group, and configure inbound rules to allow SSH traffic from your
IP address or a specific IP range.
Apply this security group to the EMR master and task nodes.
5. SSH Access:
Use an SSH client to connect to the master node of your EMR cluster.
Retrieve the public DNS name of the master node from the EMR dashboard.
Open a terminal or command prompt and use the SSH command to connect to the master node: ssh -i <path_to_private_key_file> hadoop@<public_dns_name>

Running the model training:

1. Install Libraries on EMR Master Node:
2. Connect to the master node of your Amazon EMR cluster using SSH.
3. Install the required Python libraries such as pandas, scikit-learn, and pyspark, findspark using pip
4. Upload Training Dataset to Amazon S3 and make a note of the S3 URI
5. Transfer the file containing your ML model to the master node of the EMR cluster using SCP.
6. Run Training Script on EMR Master Node: spark-submit training.py
7. Save the Trained Model to Amazon S3.
8. Terminate the EMR cluster using the AWS Management Console.

Running the prediction app on a single EC2 instance:

1. Launch an EC2 Instance:
Log in to the AWS Management Console.
Navigate to the EC2 dashboard.
Click on "Launch Instance" to start the instance launch wizard.
Choose an Amazon Machine Image (AMI).
Configure instance details such as network settings, storage(at least 30GB), and security groups.
Review and launch the instance.
2.  Connect to the EC2 Instance:
Once the instance is running, obtain the public IP address of the instance from the EC2 
dashboard.
Use an SSH client to connect to the instance 
3. Download your prediction app file and other files like training model and Validation Dataset saved on S3 using cp command (aws s3 cp s3://cs643-njit-bucket/data/ValidationDataset.csv .). 
4. Install Python packages (numpy, pandas, scikit-learn, pyspark,findspark). 
5. Run the Prediction App:
Navigate to the directory where your prediction app files are located.
Run the prediction app using the python command: spark-submit app.py <path_to_csv_file>

Running the prediction app with Docker:

1. Create a directory on your EC2 instance and copy all the files that you will need to run your prediction app(trained model, validation dataset and the app script)
2. Inside the same directory create a file named Dockerfile: Write a Dockerfile that specifies the environment and dependencies required to run the prediction app. 
3. Build Docker Image: Navigate to the directory containing your app.py file and the Dockerfile, then run the following command to build the Docker image: docker build -t wineprediction.
4. Run Docker Container: Once the Docker image is built, you can run it in a Docker container using the following command: docker run wineprediction 
5. Push the docker container in docker hub. 
