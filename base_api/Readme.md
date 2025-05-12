To Run Server:

    uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000

    OR

    python -m uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000

    source minio_config.sh && uvicorn store_feedback:app --host 0.0.0.0 --port 8010 (this one we are using)

Then Checkout: 

    http://127.0.0.1:8000/docs