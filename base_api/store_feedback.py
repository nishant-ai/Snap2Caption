from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3, base64, os
from botocore.exceptions import BotoCoreError, ClientError

# === FastAPI App ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request Schema ===
class FeedbackRequest(BaseModel):
    uid: str
    image: str  # base64 string (without data:image/... prefix)
    caption: str
    feedback: str

# === MinIO Config ===
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "129.114.25.254:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "snap2caption")

# === MinIO Client ===
s3_client = boto3.client(
    "s3",
    endpoint_url=f"http://{MINIO_ENDPOINT}",
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
)

# === Store Feedback Endpoint ===
@app.post("/store_feedback")
async def store_feedback(data: FeedbackRequest):
    uid = data.uid
    base_path = f"feedback/{uid}/"

    try:
        # Upload image
        image_bytes = base64.b64decode(data.image)
        s3_client.put_object(
            Bucket=MINIO_BUCKET_NAME,
            Key=f"{base_path}img_{uid}.jpg",
            Body=image_bytes,
            ContentType="image/jpeg"
        )

        # Upload caption
        s3_client.put_object(
            Bucket=MINIO_BUCKET_NAME,
            Key=f"{base_path}caption_{uid}.txt",
            Body=data.caption.encode("utf-8"),
            ContentType="text/plain"
        )

        # Upload feedback
        s3_client.put_object(
            Bucket=MINIO_BUCKET_NAME,
            Key=f"{base_path}feedback_{uid}.txt",
            Body=data.feedback.encode("utf-8"),
            ContentType="text/plain"
        )

        return {"status": "success", "message": f"Feedback saved to {base_path}"}

    except (BotoCoreError, ClientError, base64.binascii.Error) as e:
        raise HTTPException(status_code=500, detail=f"MinIO upload failed: {str(e)}")
