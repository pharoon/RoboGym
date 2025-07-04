from supabase import create_client
import os
from dotenv import load_dotenv
from pathlib import Path

class FileManager:
    def __init__(self, env_path=None):
        # Load environment variables from file if provided
        if env_path is None:
            env_path = Path(__file__).parent / "Credentials.env"
        load_dotenv(dotenv_path=env_path)

        self.supabase_url ="https://xljntaujspiljiczjzzh.supabase.co"
        self.supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhsam50YXVqc3BpbGppY3pqenpoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTAxMDU1NywiZXhwIjoyMDY2NTg2NTU3fQ.26xJZp3GbkZ-JqF6PEaF2ZofBV2u9WivMy_MqGPevnc"

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase credentials are missing or invalid.")

        self.client = create_client(self.supabase_url, self.supabase_key)

    def upload(self, bucket_name: str, local_path: str, remote_path: str) -> str:
        """Uploads a file to Supabase Storage and returns the public URL."""
        with open(local_path, "rb") as f:
            self.client.storage.from_(bucket_name).upload(remote_path, f, {'upsert': 'true'})
        return f"{self.supabase_url}/storage/v1/object/public/{bucket_name}/{remote_path}"

    def delete(self, bucket_name: str, remote_path: str) -> bool:
        """Deletes a file from Supabase Storage."""
        try:
            self.client.storage.from_(bucket_name).remove([remote_path])
            return True
        except Exception as e:
            print(f"Error deleting from storage: {str(e)}")
            return False

    def download(self, bucket_name: str, remote_path: str, local_path: str):
        """Downloads a file from Supabase Storage."""
        try:
            res = self.client.storage.from_(bucket_name).download(remote_path)
            with open(local_path, "wb") as f:
                f.write(res)
            return True, f"Downloaded to {local_path}"
        except Exception as e:
            return False, f"{str(e)}"
    def rename_file(self,bucket: str, old_path: str, new_path: str):
        # Step 1: Download old file content
        response = self.client.storage.from_(bucket).download(old_path)
        if not response:
            print("Failed to download existing file.")
            return False

        # Step 2: Upload the content to the new path
        upload_response =self.client.storage.from_(bucket).upload(new_path, response, {'upsert': 'true', 'content-type': 'application/zip'})
        if not upload_response:
            print("Failed to upload to new path.")
            return False

        # Step 3: Delete the old file
        delete_response = self.client.storage.from_(bucket).remove([old_path])
        if not delete_response:
            print("Uploaded to new path but failed to delete old path.")
        else:
            print("Old file deleted.")
        print("Supabase file renamed.")
        return True