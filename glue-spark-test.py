#########################################
### IMPORT LIBRARIES AND SET VARIABLES
#########################################

from datetime import datetime
import sys
import logging
import boto3
import json

from pyspark.context import SparkContext
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window
from pyspark.sql.types import *

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job


## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
spark_context = SparkContext() ### SparkContext.getOrCreate() 
glueContext = GlueContext(spark_context)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

print(">>> args:", args)

#Getting Job name
jobName = args['JOB_NAME']

#Logging parameters
MSG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(format=MSG_FORMAT, datefmt=DATETIME_FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#########################################
### EXTRACT DATA (READ DATA)
#########################################

#Log starting time
dt_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
logger.info(f"Job <{jobName}> (GLUE SPARK ETL) Started at {dt_start}")

read_data_path = 's3://banking-datalake-repo-staging/raw-data/temp/csv-data/load.csv'

users_df = spark.read.option("header",True) \
     .csv(read_data_path).sort(['id','update_date'], ascending=True)

logger.info(f"Data file(s) read from {read_data_path} successfully ...") 
logger.info("Created Spark Data Frame ...")
logger.info(f"Count: {users_df.count()}")
users_df.printSchema()
users_df.show()

#########################################
### READ SCHEMA
#########################################

s3_client = boto3.client('s3')
bucket = 'banking-datalake-repo-staging'
key = 'raw-data/temp/json-schema/types_mapping.json' 

data = s3_client.get_object(Bucket=bucket, Key=key)

json_data = data['Body'].read().decode('utf-8')
json_content = json.loads(json_data)
logger.info(f"json schema content: {json_content}")

#########################################
### TRANSFORM
#########################################

## Applying new json schema
def udf_mapping_type(type_col):
    return {
        'integer': IntegerType(),
        'timestamp': TimestampType(),
        'string': StringType()
    }.get(type_col, 'INVALID_TYPE')
    
for name_col, type_col in json_content.items():
    users_df = users_df.withColumn(name_col, f.col(name_col).cast(udf_mapping_type(type_col)))
users_df.printSchema()

## Removing older duplicates by update_date
w = Window.partitionBy("id").orderBy(f.desc("update_date"))
users_df = users_df.withColumn("rank", f.row_number().over(w)).filter("rank = 1").drop("rank").sort(['id'], ascending=True) #, inplace=True)
users_df.show()

#########################################
### LOAD (DATA SINK)
#########################################

write_path = "s3://banking-datalake-repo-staging/raw-data/temp/output-parquet/"
users_df.coalesce(1).write.format("parquet").mode("overwrite").save(write_path)
logger.info(f"Parquet file(s) saved on {write_path} successfully ...") 

## Show content saved
output_users_df = spark.read.parquet("s3://banking-datalake-repo-staging/raw-data/temp/output-parquet")

logger.info(f"Parquet data output file(s) read from {write_path} successfully ...") 
logger.info("Created Output Spark Data Frame ...")
logger.info(f"Count: {output_users_df.count()}")
output_users_df.printSchema()
logger.info(f"Result Dataframe ...") 
output_users_df.show()

#Log end time
dt_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger.info(f"Job <{jobName}> (GLUE SPARK ETL) Ended at {dt_end}")

job.commit()
logger.info("Job commited ...")
logger.info("Process Terminated ...")