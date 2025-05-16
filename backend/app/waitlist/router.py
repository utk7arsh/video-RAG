from fastapi import APIRouter, HTTPException
from .models import EmailIn, EmailResponse
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", "")
)

router = APIRouter(prefix="/waitlist", tags=["waitlist"])

@router.post("/", response_model=EmailResponse)
async def add_to_waitlist(email_in: EmailIn):
    logger.debug(f"Received email: {email_in.email}")
    try:
        # Insert email into Supabase
        result = supabase.table("waitlist_emails").insert({"email": email_in.email}).execute()
        logger.debug(f"Supabase response: {result}")
        return EmailResponse(message="Email added to waitlist")
    except Exception as e:
        logger.error(f"Error adding email to waitlist: {str(e)}")
        if "duplicate key" in str(e):
            raise HTTPException(status_code=400, detail="Email already registered")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}") 