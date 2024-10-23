# Read pdfs from s3 bucket
# Work In Progress *****
import os

import boto3
from dotenv import load_dotenv

#s3 client 
s3 = boto3.client(
    service_name="s3",
    region_name = "us-east-1",
    aws_access_key_id=os.getenv("ac_key"),
    aws_secret_access_key=os.getenv("sac_key"),
)


# Read an object from the bucket
bucket_name = "shel-underground-expressions"
obj_name = "KIRIRI GARDEN HOTEL ROI VERSION FINAL.pdf"
obj = s3.Bucket('shel-underground-expressions/public').Object('KIRIRI GARDEN HOTEL ROI VERSION FINAL.pdf').get()
    #response = s3.get_object(Bucket=bucket_name, Key=obj_name)
    #print(obj)


s3 = boto3.resource('s3')
bucket = s3.Bucket('shel-underground-expressions')
# Iterates through all the objects, doing the pagination for you. Each obj
# is an ObjectSummary, so it doesn't contain the body. You'll need to call
# get to get the whole body.
for obj in bucket.objects.all():
    key = obj.key
    body = obj.get()['Body'].read()
print(obj)