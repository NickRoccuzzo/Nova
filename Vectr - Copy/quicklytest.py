import boto3
import os

# Config
BUCKET_NAME = 'nova-93'
OBJECT_KEY = 'testtextfileforS3.txt'  # No leading slash
LOCAL_FILE = 'testtextfileforS3.txt'

ACCESS_KEY = 'AKIASJZJ27PFRBVTS2CA'
SECRET_KEY = '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'

# Create boto3 session
session = boto3.Session(
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)
s3 = session.client('s3')

# Check if the file exists locally already
if not os.path.exists(LOCAL_FILE):
    print("Downloading test file from S3...")
    s3.download_file(BUCKET_NAME, OBJECT_KEY, LOCAL_FILE)
    print(f"✅ Downloaded {LOCAL_FILE}")
else:
    print(f"✅ {LOCAL_FILE} already exists locally. Using cached version.")

# Read the file to verify contents
with open(LOCAL_FILE, 'r') as f:
    content = f.read()
    print("\nFile content:")
    print(content)