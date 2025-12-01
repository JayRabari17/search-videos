import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
)

user_query = "person dancing"

request_body = {"inputType": "text", "text": {"inputText": user_query}}

response = bedrock_runtime.invoke_model(
    modelId="us.twelvelabs.marengo-embed-3-0-v1:0",
    body=json.dumps(request_body),
    contentType="application/json",
    accept="application/json",
)

result = json.loads(response["body"].read())

if "data" in result and len(result["data"]) > 0:
    embedding = result["data"][0].get("embedding", [])
    print(f"Generated embedding with {len(embedding)} dimensions")
    print(embedding)
else:
    print("No embedding generated")
    print(f"Response: {result}")
