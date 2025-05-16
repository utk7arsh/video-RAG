from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from .waitlist.router import router as waitlist_router
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Video RAG API",
    description="Backend API for Video RAG application",
    version="1.0.0"
)

# Get allowed origins from environment variable or use defaults
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080,https://videoinsights.vercel.app/"
).split(",")

logger.debug(f"Allowed origins: {ALLOWED_ORIGINS}")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(waitlist_router)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Request method: {request.method}")
    logger.debug(f"Request URL: {request.url}")
    logger.debug(f"Request headers: {request.headers}")
    response = await call_next(request)
    logger.debug(f"Response status: {response.status_code}")
    return response

@app.get("/")
async def root():
    return {"message": "Welcome to Video RAG API"} 