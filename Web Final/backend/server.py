from fastapi import FastAPI, APIRouter, HTTPException, Cookie, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import razorpay
import httpx
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
import base64

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    subscription_type: Optional[str] = None  # 'jee' or 'neet'
    subscription_status: str = 'inactive'  # 'active' or 'inactive'
    subscription_expires_at: Optional[datetime] = None
    created_at: datetime

class SessionData(BaseModel):
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime

class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    image_url: Optional[str] = None
    timestamp: datetime

class Chat(BaseModel):
    model_config = ConfigDict(extra="ignore")
    chat_id: str
    user_id: str
    title: str
    messages: List[Message] = []
    created_at: datetime
    updated_at: datetime

class Subscription(BaseModel):
    model_config = ConfigDict(extra="ignore")
    subscription_id: str
    user_id: str
    plan_type: str  # 'jee' or 'neet'
    amount: int  # in paise
    razorpay_order_id: Optional[str] = None
    razorpay_payment_id: Optional[str] = None
    status: str
    created_at: datetime

# Auth Helper
async def get_current_user(request: Request, session_token: Optional[str] = Cookie(None)) -> User:
    # Check cookie first, then Authorization header
    token = session_token
    if not token:
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
    
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    session = await db.user_sessions.find_one({"session_token": token}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    expires_at = session["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")
    
    user = await db.users.find_one({"user_id": session["user_id"]}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Convert datetime strings to datetime objects
    if isinstance(user.get('created_at'), str):
        user['created_at'] = datetime.fromisoformat(user['created_at'])
    if user.get('subscription_expires_at') and isinstance(user['subscription_expires_at'], str):
        user['subscription_expires_at'] = datetime.fromisoformat(user['subscription_expires_at'])
    
    return User(**user)

# AUTH ENDPOINTS
@api_router.post("/auth/session")
async def process_session(request: Request):
    data = await request.json()
    session_id = data.get('session_id')
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    
    # Call Emergent Auth API
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers={"X-Session-ID": session_id}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid session_id")
        
        auth_data = response.json()
    
    # Check if user exists
    user = await db.users.find_one({"email": auth_data['email']}, {"_id": 0})
    
    if user:
        user_id = user['user_id']
        # Update user info
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {
                "name": auth_data['name'],
                "picture": auth_data['picture']
            }}
        )
    else:
        # Create new user
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        new_user = {
            "user_id": user_id,
            "email": auth_data['email'],
            "name": auth_data['name'],
            "picture": auth_data['picture'],
            "subscription_type": None,
            "subscription_status": "inactive",
            "subscription_expires_at": None,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.users.insert_one(new_user)
        user = new_user
    
    # Create session
    session_token = auth_data['session_token']
    session_doc = {
        "user_id": user_id,
        "session_token": session_token,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.user_sessions.insert_one(session_doc)
    
    # Prepare user response
    response_user = {
        "user_id": user_id,
        "email": user['email'],
        "name": user['name'],
        "picture": user.get('picture'),
        "subscription_type": user.get('subscription_type'),
        "subscription_status": user.get('subscription_status', 'inactive')
    }
    
    response = JSONResponse(content={"user": response_user, "session_token": session_token})
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=7*24*60*60,
        path="/"
    )
    
    return response

@api_router.get("/auth/me")
async def get_me(request: Request, session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(request, session_token)
    return user

@api_router.post("/auth/logout")
async def logout(request: Request, session_token: Optional[str] = Cookie(None)):
    if session_token:
        await db.user_sessions.delete_one({"session_token": session_token})
    
    response = JSONResponse(content={"message": "Logged out"})
    response.delete_cookie(key="session_token", path="/")
    return response

# CHAT ENDPOINTS
@api_router.post("/chat/send")
async def send_message(
    request: Request,
    message: str = Form(...),
    image: Optional[UploadFile] = File(None),
    chat_id: Optional[str] = Form(None),
    session_token: Optional[str] = Cookie(None)
):
    user = await get_current_user(request, session_token)
    
    # Check subscription
    if user.subscription_status != 'active':
        raise HTTPException(status_code=403, detail="Active subscription required")
    
    # Get or create chat
    if chat_id:
        chat = await db.chats.find_one({"chat_id": chat_id, "user_id": user.user_id}, {"_id": 0})
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
    else:
        chat_id = f"chat_{uuid.uuid4().hex[:12]}"
        chat = {
            "chat_id": chat_id,
            "user_id": user.user_id,
            "title": message[:50],
            "messages": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        await db.chats.insert_one(chat)
    
    # Add user message
    user_message = {
        "role": "user",
        "content": message,
        "image_url": None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Process image if provided
    image_base64 = None
    if image:
        image_bytes = await image.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        user_message["image_url"] = f"data:image/jpeg;base64,{image_base64[:100]}..."  # Store preview
    
    # Call LLM
    llm_api_key = os.environ.get('EMERGENT_LLM_KEY')
    llm_chat = LlmChat(
        api_key=llm_api_key,
        session_id=chat_id,
        system_message=f"You are an expert tutor for {user.subscription_type.upper()} exam preparation. Help students solve doubts clearly and provide step-by-step explanations. Focus on concepts from Physics, Chemistry, and {'Biology' if user.subscription_type == 'neet' else 'Mathematics'}."
    ).with_model("openai", "gpt-5.2")
    
    # Prepare LLM message
    llm_message = UserMessage(text=message)
    if image_base64:
        llm_message.file_contents = [ImageContent(image_base64=image_base64)]
    
    # Get AI response
    ai_response = await llm_chat.send_message(llm_message)
    
    # Add assistant message
    assistant_message = {
        "role": "assistant",
        "content": ai_response,
        "image_url": None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Update chat in DB
    await db.chats.update_one(
        {"chat_id": chat_id},
        {
            "$push": {"messages": {"$each": [user_message, assistant_message]}},
            "$set": {"updated_at": datetime.now(timezone.utc).isoformat()}
        }
    )
    
    return {
        "chat_id": chat_id,
        "response": ai_response,
        "timestamp": assistant_message["timestamp"]
    }

@api_router.get("/chat/history")
async def get_chat_history(request: Request, session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(request, session_token)
    
    chats = await db.chats.find({"user_id": user.user_id}, {"_id": 0}).sort("updated_at", -1).to_list(100)
    return chats

@api_router.get("/chat/{chat_id}")
async def get_chat(chat_id: str, request: Request, session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(request, session_token)
    
    chat = await db.chats.find_one({"chat_id": chat_id, "user_id": user.user_id}, {"_id": 0})
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return chat

# TEST GENERATOR
@api_router.post("/test/generate")
async def generate_test(
    request: Request,
    session_token: Optional[str] = Cookie(None)
):
    user = await get_current_user(request, session_token)
    
    if user.subscription_status != 'active':
        raise HTTPException(status_code=403, detail="Active subscription required")
    
    data = await request.json()
    exam_type = data.get('exam_type', user.subscription_type)
    subject = data.get('subject', 'Physics')
    num_questions = data.get('num_questions', 10)
    
    # Call LLM to generate questions
    llm_api_key = os.environ.get('EMERGENT_LLM_KEY')
    llm_chat = LlmChat(
        api_key=llm_api_key,
        session_id=f"test_{uuid.uuid4().hex[:8]}",
        system_message=f"You are an expert question paper creator for {exam_type.upper()} exams. Generate high-quality questions similar to previous year papers."
    ).with_model("openai", "gpt-5.2")
    
    prompt = f"""Generate {num_questions} {exam_type.upper()} {subject} questions in JSON format.

Format:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
      "correct_answer": "A",
      "explanation": "Brief explanation",
      "difficulty": "easy/medium/hard"
    }}
  ]
}}

Ensure questions are similar to actual {exam_type.upper()} PYQs. Return ONLY valid JSON."""
    
    llm_message = UserMessage(text=prompt)
    response = await llm_chat.send_message(llm_message)
    
    # Parse JSON response
    import json
    try:
        # Extract JSON from response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            test_data = json.loads(json_str)
        else:
            test_data = json.loads(response)
    except:
        # Fallback if JSON parsing fails
        test_data = {
            "questions": [{
                "question": "Sample question - AI response parsing failed",
                "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                "correct_answer": "A",
                "explanation": "This is a sample question",
                "difficulty": "medium"
            }]
        }
    
    return test_data

# SUBSCRIPTION ENDPOINTS
@api_router.post("/subscription/create-order")
async def create_order(
    request: Request,
    session_token: Optional[str] = Cookie(None)
):
    user = await get_current_user(request, session_token)
    
    data = await request.json()
    plan_type = data.get('plan_type')  # 'jee' or 'neet'
    
    if plan_type not in ['jee', 'neet']:
        raise HTTPException(status_code=400, detail="Invalid plan_type")
    
    # Amount in paise
    amount = 10100 if plan_type == 'jee' else 5100
    
    # Create Razorpay order (Note: Using test credentials in demo)
    # In production, get these from environment
    subscription_id = f"sub_{uuid.uuid4().hex[:12]}"
    
    # Store subscription
    sub_doc = {
        "subscription_id": subscription_id,
        "user_id": user.user_id,
        "plan_type": plan_type,
        "amount": amount,
        "razorpay_order_id": None,
        "razorpay_payment_id": None,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.subscriptions.insert_one(sub_doc)
    
    return {
        "subscription_id": subscription_id,
        "amount": amount,
        "plan_type": plan_type,
        "currency": "INR"
    }

@api_router.post("/subscription/verify")
async def verify_payment(
    request: Request,
    session_token: Optional[str] = Cookie(None)
):
    user = await get_current_user(request, session_token)
    
    data = await request.json()
    subscription_id = data.get('subscription_id')
    payment_id = data.get('payment_id', 'demo_payment_' + uuid.uuid4().hex[:8])
    
    # Update subscription
    await db.subscriptions.update_one(
        {"subscription_id": subscription_id},
        {
            "$set": {
                "razorpay_payment_id": payment_id,
                "status": "completed"
            }
        }
    )
    
    # Update user subscription
    subscription = await db.subscriptions.find_one({"subscription_id": subscription_id}, {"_id": 0})
    
    expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    await db.users.update_one(
        {"user_id": user.user_id},
        {
            "$set": {
                "subscription_type": subscription['plan_type'],
                "subscription_status": "active",
                "subscription_expires_at": expires_at.isoformat()
            }
        }
    )
    
    return {
        "message": "Subscription activated",
        "expires_at": expires_at.isoformat()
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
