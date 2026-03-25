import boto3
from rag.ingest import ingest_pdf_from_s3

BUCKET = "rag-project-thesis-pdfs"
PREFIX = "thesis/"
PROFILE = "rag-project"

if __name__ == "__main__":
    session = boto3.Session(profile_name=PROFILE)
    s3 = session.client("s3")

    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX)
    pdf_keys = [
        obj["Key"]
        for obj in response.get("Contents", [])
        if obj["Key"].endswith(".pdf")
    ]

    print(f"Found {len(pdf_keys)} PDF(s) in s3://{BUCKET}/{PREFIX}")

    for key in pdf_keys:
        print(f"Ingesting {key} ...")
        result = ingest_pdf_from_s3(BUCKET, key, PROFILE)
        print(f"  Done: {result}")