from supabase import create_client
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent / "Credentials.env"
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_to_storage(bucket_name: str, local_path: str, remote_path: str) -> str:
    with open(local_path, "rb") as f:
        supabase.storage.from_(bucket_name).upload(remote_path, f, {'upsert': 'true',})
    # ✅ Return public URL of uploaded file
    return f"{SUPABASE_URL}/storage/v1/object/public/{bucket_name}/{remote_path}"

def delete_from_storage(bucket_name: str, remote_path: str) -> bool:
    try:
        supabase.storage.from_(bucket_name).remove([remote_path])
        return True
    except Exception as e:
        print(f"Error deleting from storage: {str(e)}")
        return False

def download_from_storage(bucket_name: str, remote_path: str, local_path: str):
    """Download a file from Supabase Storage and save it locally."""
    try:
        res = supabase.storage.from_(bucket_name).download(remote_path)
        with open(local_path, "wb") as f:
            f.write(res)
        return True, f"Downloaded to {local_path}"
    except Exception as e:
        return False, str(e)