import boto3, json
from dotenv import load_dotenv

load_dotenv()
client = boto3.client("bedrock-runtime", region_name="us-west-2")

resp = client.invoke_model(
    modelId="amazon.titan-embed-text-v1",
    body='{"inputText":"hello"}'
)
print(json.loads(resp["body"].read()))

