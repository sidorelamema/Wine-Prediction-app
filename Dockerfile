#Save the JDK version to be used in a variable
ARG OPENJDK_VERSION=8

#Get a base docker image based on JDK
FROM openjdk:${OPENJDK_VERSION}-jre-slim


#Configure spark version to be used
ARG SPARK_VERSION=3.5.1
ARG SPARK_EXTRAS=


#Set Label for container
LABEL org.opencontainers.image.title="Apache PySpark $SPARK_VERSION" \
      org.opencontainers.image.version=$SPARK_VERSION


#Since we are using miniconda3, setting up environemnt path before hand
ENV PATH="/opt/miniconda3/bin:${PATH}"
ENV PYSPARK_PYTHON="/opt/miniconda3/bin/python"


#Install dependencies for our docker image
RUN set -ex && \
        apt-get update && \
    apt-get install -y curl bzip2 --no-install-recommends && \
    curl -s -L --url "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" --output /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -f -p "/opt/miniconda3" && \
    rm /tmp/miniconda.sh && \
    conda config --set auto_update_conda true && \
    conda config --set channel_priority false && \
    conda update conda -y --force-reinstall && \
    conda install pip -y && \
    conda clean -tipy && \
    echo "PATH=/opt/miniconda3/bin:\${PATH}" > /etc/profile.d/miniconda.sh && \
    pip install --no-cache pyspark[$SPARK_EXTRAS]==${SPARK_VERSION} && \
    pip install findspark && \
    pip install numpy && \
    pip install scikit-learn && \
    pip install pandas && \   
    SPARK_HOME=$(python /opt/miniconda3/bin/find_spark_home.py) && \
    echo "export SPARK_HOME=$(python3 /opt/miniconda3/bin/find_spark_home.py)" > /etc/profile.d/spark.sh && \
    curl -s -L --url "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.8.9/aws-java-sdk-1.8.9.jar" --output $SPARK_HOME/jars/aws-java-sdk-1.8.9.jar && \
    curl -s -L --url "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.6/hadoop-aws-3.3.6.jar" --output $SPARK_HOME/jars/hadoop-aws-3.3.6.jar && \
    mkdir -p $SPARK_HOME/conf && \
    echo "spark.hadoop.fs.s3.impl=org.apache.hadoop.fs.s3a.S3AFileSystem" >> $SPARK_HOME/conf/spark-defaults.conf && \
    apt-get remove -y curl bzip2 && \
    apt-get autoremove -y && \
    apt-get clean
#Set Working env as /data
ENV PROG_DIR /data


#Set filename of ML Python file
ENV PROG_NAME app.py

#RUN mkdir -p /data/Modelfile
#Set Workdir as set in env variable PROG_DIR
WORKDIR ${PROG_DIR}
COPY app.py /data/
COPY ValidationDataset.csv /data/
COPY model.tar /data/
RUN tar -xvf /data/model.tar

#COPY * /data

#Copy python files, and datasets to work directory
ADD ${PROG_NAME} .
ENTRYPOINT ["spark-submit", "app.py"]
CMD ["/data/ValidationDataset.csv"]