from pydantic import BaseModel, EmailStr

class EmailIn(BaseModel):
    email: EmailStr

class EmailResponse(BaseModel):
    message: str 