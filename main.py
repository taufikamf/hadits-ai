import os
import sys
import json
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, AsyncGenerator, Optional
from contextlib import asynccontextmanager

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Import internal modules
from utils import query_optimizer
# query_runner is imported in the lifespan function to avoid startup issues

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found. LLM features will be disabled.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class SessionQuestionRequest(BaseModel):
    question: str
    session_id: str = None

class HealthResponse(BaseModel):
    status: str
    message: str
    retrieval_status: str = "ok"
    llm_status: str = "unknown"

class ChatMessage(BaseModel):
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    
class ChatSession(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[ChatMessage] = []

class CreateSessionRequest(BaseModel):
    title: str = None

class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int

# Global variables to cache system components
retrieval_initialized = False

# Chat session storage configuration
SESSIONS_DIR = "data/sessions"
SESSIONS_FILE = os.path.join(SESSIONS_DIR, "sessions.json")

class ChatSessionManager:
    def __init__(self):
        # Ensure sessions directory exists
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        self.sessions = self._load_sessions()
    
    def _load_sessions(self) -> Dict[str, ChatSession]:
        """Load sessions from file"""
        if not os.path.exists(SESSIONS_FILE):
            return {}
        
        try:
            with open(SESSIONS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sessions = {}
                for session_id, session_data in data.items():
                    sessions[session_id] = ChatSession(**session_data)
                return sessions
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            return {}
    
    def _save_sessions(self):
        """Save sessions to file"""
        try:
            data = {}
            for session_id, session in self.sessions.items():
                data[session_id] = session.dict()
            
            with open(SESSIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def create_session(self, title: str = None) -> ChatSession:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        if not title:
            title = f"Chat {len(self.sessions) + 1}"
        
        session = ChatSession(
            session_id=session_id,
            title=title,
            created_at=timestamp,
            updated_at=timestamp,
            messages=[]
        )
        
        self.sessions[session_id] = session
        self._save_sessions()
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a specific session"""
        return self.sessions.get(session_id)
    
    def get_all_sessions(self) -> List[SessionResponse]:
        """Get all sessions summary"""
        sessions = []
        for session in self.sessions.values():
            sessions.append(SessionResponse(
                session_id=session.session_id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=len(session.messages)
            ))
        
        # Sort by updated_at descending
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return sessions
    
    def add_message(self, session_id: str, role: str, content: str) -> Optional[ChatMessage]:
        """Add a message to a session"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        message = ChatMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        
        session.messages.append(message)
        session.updated_at = datetime.now().isoformat()
        
        self._save_sessions()
        return message
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._save_sessions()
            return True
        return False
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update session title"""
        session = self.get_session(session_id)
        if session:
            session.title = title
            session.updated_at = datetime.now().isoformat()
            self._save_sessions()
            return True
        return False

# Initialize session manager
session_manager = ChatSessionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources"""
    global retrieval_initialized
    
    # Startup
    logger.info("Starting Hadis AI Backend...")
    
    # Test retrieval system initialization
    try:
        logger.info("Attempting to initialize retrieval system...")
        
        # Import here to avoid any potential import issues during startup
        from retriever import query_runner
        logger.info("Successfully imported query_runner module")
        
        # Test with a more robust initialization check
        logger.info("Running test query to verify retrieval system...")
        test_results = query_runner.query_hadits_return("shalat", top_k=1)
        
        if test_results and len(test_results) > 0:
            retrieval_initialized = True
            logger.info(f"Retrieval system initialized successfully - found {len(test_results)} results")
            logger.info(f"Sample result from kitab: {test_results[0].get('kitab', 'Unknown')}")
        else:
            logger.warning("Retrieval system returned no results for test query")
            retrieval_initialized = False
            
    except ImportError as e:
        logger.error(f"Failed to import retrieval modules: {e}")
        retrieval_initialized = False
    except Exception as e:
        logger.error(f"Failed to initialize retrieval system: {type(e).__name__}: {e}")
        logger.error("Full traceback:", exc_info=True)
        retrieval_initialized = False
    
    # Log final status
    if retrieval_initialized:
        logger.info("✅ Application startup completed successfully")
    else:
        logger.warning("⚠️ Application started with degraded functionality - retrieval system unavailable")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Hadis AI Backend...")

# Create FastAPI app
app = FastAPI(
    title="Hadis AI Backend",
    description="RAG-based chatbot for Islamic Hadith queries with streaming responses",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are a helpful assistant specializing in Islamic Hadith knowledge.
Use the following context as your learned knowledge, inside <context></context> XML tags.
<context>
{context}
</context>

When responding to the user: 
- Always search for relevant Hadiths from the context first.
- Begin your response with: "Berikut adalah hadits-hadits yang relevan dari pertanyaan yang Anda berikan:"
- Present each hadith in a **clear, structured format** using numbered lists instead of tables.
- For each hadith, include:
  * **Kitab**: Source book name
  * **ID**: Hadith identifier  
  * **Arab**: Arabic text (abbreviated if very long)
  * **Terjemah**: Indonesian translation
- After showing the hadiths, provide a **summary** of the relevant content in a clear and concise way.
- End your answer by asking the user a follow-up question, such as: 
  - "Apakah Anda ingin penjelasan lebih lanjut tentang hadits ini?" 
  - "Ingin mencari hadits dengan topik lain?"

Rules:
- Use formal and polite **Bahasa Indonesia** when responding. 
- If you don't know the answer, just say that you don't know. 
- If you're not sure, ask the user for clarification. 
- IMPORTANT: You MUST show ALL available hadiths in the context, not just one. Include every hadith provided to you. 
- If the question is too vague, ask the user to be more specific (e.g., "Topik apa yang Anda maksudkan?").
- NEVER use markdown tables as they can cause formatting issues.

Example response format:

**Hadits 1:**
- **Kitab**: Sunan Abu Daud
- **ID**: 1
- **Arab**: حَدَّثَنَا عَبْدُ اللَّهِ... (text abbreviated if too long)
- **Terjemah**: Telah menceritakan kepada kami...

**Hadits 2:**
- **Kitab**: Shahih Bukhari  
- **ID**: 123
- **Arab**: حَدَّثَنَا مُحَمَّدٌ...
- **Terjemah**: Telah menceritakan kepada kami...

**Ringkasan**: Nabi shallallahu 'alaihi wasallam mengajarkan adab tertentu ketika buang hajat, seperti menjauh dari manusia dan tidak menghadap kiblat. 

**Pertanyaan lanjut**: Apakah Anda ingin penjelasan lebih lanjut atau mencari hadits lainnya?"""

FOLLOW_UP_PROMPT = """You are a helpful assistant specializing in Islamic Hadith knowledge.
The user is asking for further explanation about hadiths that were previously shared. Use the following context from the previous response as your learned knowledge, inside <context></context> XML tags.
<context>
{context}
</context>

IMPORTANT: The user is asking for MORE DETAILED EXPLANATION about the hadiths above. You should:
- Provide a comprehensive analysis of each hadith
- Explain the historical context and circumstances (asbab al-hadith) if relevant
- Discuss the practical implications and lessons learned
- Explain key Arabic terms or concepts mentioned
- Connect the hadiths to broader Islamic principles
- Provide scholarly insights about the meaning and application

Begin your response with: "Berikut adalah penjelasan lebih lanjut tentang hadits-hadits tersebut:"

Use formal and polite **Bahasa Indonesia** when responding.
Provide detailed, educational content that helps the user understand the deeper meaning and significance of the hadiths.

End with: "Apakah ada aspek tertentu dari hadits-hadits ini yang ingin Anda pahami lebih dalam?"
"""

def is_follow_up_query(question: str) -> bool:
    """Detect if the user is asking for further explanation of previous hadiths"""
    follow_up_keywords = [
        "jelaskan lebih lanjut",
        "jelaskan lebih detail", 
        "jelaskan lebih dalam",
        "penjelasan lebih lanjut",
        "penjelasan lebih detail",
        "bisa dijelaskan lebih",
        "tolong jelaskan lebih",
        "mohon penjelasan lebih",
        "apa maksud",
        "apa arti",
        "bagaimana maksudnya",
        "bisa diperjelas",
        "perjelas",
        "detail",
        "lebih lanjut",
        "lebih dalam",
        "boleh dijelaskan",
        "boleh jelaskan lebih",
        "jelaskan tentang hadis tersebut",
        "jelaskan tentang hadits tersebut", 
        "hadis tadi",
        "hadits tadi",
        "hadis sebelumnya",
        "hadits sebelumnya",
        "hadis yang tadi",
        "hadits yang tadi"
    ]
    
    question_lower = question.lower().strip()
    
    # Check for exact matches or partial matches
    for keyword in follow_up_keywords:
        if keyword in question_lower:
            return True
    
    # Additional pattern matching for short queries that are likely follow-ups
    if len(question_lower.split()) <= 4:
        short_follow_ups = [
            "jelaskan",
            "dijelaskan", 
            "maksudnya",
            "artinya",
            "bagaimana",
            "kenapa",
            "mengapa"
        ]
        for keyword in short_follow_ups:
            if question_lower.startswith(keyword) or question_lower == keyword:
                return True
    
    return False

def extract_previous_hadith_context(session_messages: List[ChatMessage]) -> str:
    """Extract hadith context from the last assistant message"""
    if not session_messages:
        return ""
    
    # Find the last assistant message
    last_assistant_message = None
    for message in reversed(session_messages):
        if message.role == "assistant":
            last_assistant_message = message
            break
    
    if not last_assistant_message:
        return ""
    
    # Extract content that looks like hadith information
    content = last_assistant_message.content
    
    # Return the full content as context for follow-up
    return content

async def format_sse_message(event_type: str, data: str) -> str:
    """Format message according to SSE specification"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def prepare_context_from_hadiths(hadiths: List[Dict]) -> str:
    """Convert hadith search results into context string with length limits"""
    if not hadiths:
        return "Tidak ada hadits yang ditemukan untuk pertanyaan ini."
    
    context_parts = []
    total_length = 0
    max_total_length = 8000  # Ditingkatkan dari 4000 ke 8000 untuk menampilkan lebih banyak hadits
    
    for hadith in hadiths:
        # Truncate very long Arabic text to prevent formatting issues
        arab_text = hadith.get('arab', '')
        if len(arab_text) > 300:  # Dikurangi dari 500 ke 300 untuk menghemat ruang
            arab_text = arab_text[:300] + "... (teks diperpendek)"
            
        # Truncate very long translation text  
        terjemah_text = hadith.get('terjemah', '')
        if len(terjemah_text) > 800:  # Limit translation length
            terjemah_text = terjemah_text[:800] + "... (teks diperpendek)"
            
        context_part = f"""
Kitab: {hadith.get('kitab', 'Unknown')}
ID: {hadith.get('id', 'Unknown')}
Arab: {arab_text}
Terjemah: {terjemah_text}
---"""
        
        # Check if adding this hadith would exceed the total length limit
        if total_length + len(context_part) > max_total_length:
            logger.info(f"Context length limit reached, truncating to {len(context_parts)} hadiths")
            break
            
        context_parts.append(context_part)
        total_length += len(context_part)
    
    return "\n".join(context_parts)

async def generate_llm_response(question: str, context: str, is_follow_up: bool = False) -> AsyncGenerator[str, None]:
    
    if not GEMINI_API_KEY:
        yield await format_sse_message("error", "LLM service tidak tersedia. API key tidak dikonfigurasi.")
        return
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Choose prompt based on whether this is a follow-up query
        if is_follow_up:
            full_prompt = FOLLOW_UP_PROMPT.format(context=context) + f"\n\nPertanyaan follow-up: {question}"
        else:
            full_prompt = SYSTEM_PROMPT.format(context=context) + f"\n\nPertanyaan: {question}"
        
        response = model.generate_content(
            full_prompt,
            stream=True,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,  # Lower temperature for more stable output
                max_output_tokens=3000,  # Increased for detailed follow-up explanations
                top_p=0.9,
                top_k=40,
            )
        )
        
        for chunk in response:
            if chunk.text:
                # Clean up the chunk text to remove excessive whitespace
                cleaned_text = chunk.text.strip()
                if cleaned_text:  # Only yield non-empty chunks
                    yield await format_sse_message("message", cleaned_text)
                    await asyncio.sleep(0.01)  
        
        yield await format_sse_message("complete", "Response generation completed")
        
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        yield await format_sse_message("error", f"Terjadi kesalahan dalam generate respons: {str(e)}")

async def stream_hadits_response(question: str, session_id: str = None) -> AsyncGenerator[str, None]:
    """Main streaming function that handles the complete RAG pipeline"""
    
    full_response = ""  
    
    try:
        if session_id:
            session_manager.add_message(session_id, "user", question)
            yield await format_sse_message("session_updated", session_id)
        
        yield await format_sse_message("status", "Memproses pertanyaan Anda...")
        
        # Check if this is a follow-up query
        is_follow_up = is_follow_up_query(question)
        context = ""
        
        if is_follow_up and session_id:
            # For follow-up queries, use previous hadith context from session
            yield await format_sse_message("status", "Mengambil konteks hadits sebelumnya...")
            
            session = session_manager.get_session(session_id)
            if session and session.messages:
                context = extract_previous_hadith_context(session.messages)
                
                if context:
                    logger.info(f"Follow-up detected. Using previous context: {len(context)} characters")
                    yield await format_sse_message("retrieval_complete", "Menggunakan hadits dari konteks sebelumnya")
                else:
                    # Fallback to regular search if no previous context found
                    is_follow_up = False
                    logger.info("Follow-up detected but no previous context found, falling back to regular search")
            else:
                is_follow_up = False
                logger.info("Follow-up detected but no session found, falling back to regular search")
        
        if not is_follow_up:
            # Regular hadith search
            yield await format_sse_message("status", "Mengoptimalkan query pencarian...")
            optimized_query, required_keywords = query_optimizer.optimize_query(question, return_keywords=True)
            
            logger.info(f"Original question: {question}")
            logger.info(f"Optimized query: {optimized_query}")
            logger.info(f"Keywords: {required_keywords}")
            
            yield await format_sse_message("status", "Mencari hadits yang relevan...")
            
            # Import query_runner here since it's not imported globally
            from retriever import query_runner
            
            hadiths = query_runner.query_hadits_return(
                raw_query=question,
                optimized_query=optimized_query,
                top_k=5,
                required_keywords=required_keywords,
                min_match=2
            )
            
            logger.info(f"Found {len(hadiths)} hadiths")
            
            context = prepare_context_from_hadiths(hadiths)
            
            # Log jumlah hadits dalam konteks
            context_hadits_count = context.count("Kitab:")
            logger.info(f"Jumlah hadits dalam konteks: {context_hadits_count} dari total {len(hadiths)}")
            
            yield await format_sse_message("retrieval_complete", f"Ditemukan {len(hadiths)} hadits relevan")
        
        if GEMINI_API_KEY:
            if is_follow_up:
                yield await format_sse_message("status", "Menyusun penjelasan detail...")
            else:
                yield await format_sse_message("status", "Menyusun respons...")
            
            async for llm_chunk in generate_llm_response(question, context, is_follow_up=is_follow_up):
                if llm_chunk.startswith("event: message"):
                    lines = llm_chunk.split('\n')
                    for line in lines:
                        if line.startswith('data:'):
                            try:
                                data = json.loads(line[5:].strip())
                                # Only add to full_response if data is meaningful
                                if data and data.strip() and not data.isspace():
                                    full_response += data
                            except json.JSONDecodeError:
                                # Skip malformed JSON data
                                continue
                            except Exception:
                                # Skip any other parsing errors
                                continue
                yield llm_chunk
        else:
            yield await format_sse_message("status", "LLM tidak tersedia, menampilkan hasil pencarian...")
            
            if is_follow_up:
                if context:
                    full_response = f"Konteks hadits sebelumnya:\n\n{context}\n\nMohon maaf, LLM tidak tersedia untuk memberikan penjelasan detail. Silakan gunakan konteks hadits di atas."
                    yield await format_sse_message("message", full_response)
                else:
                    full_response = "Maaf, tidak ada konteks hadits sebelumnya yang dapat dijelaskan dan LLM tidak tersedia."
                    yield await format_sse_message("message", full_response)
            else:
                hadiths = []  # Initialize in case not set in follow-up path
                if 'hadiths' in locals():
                    if hadiths:
                        response_text = f"Ditemukan {len(hadiths)} hadits untuk pertanyaan: {question}\n\n"
                        for i, hadith in enumerate(hadiths, 1):
                            response_text += f"{i}. **{hadith.get('kitab', 'Unknown')}** (ID: {hadith.get('id', 'Unknown')})\n"
                            response_text += f"   - Arab: {hadith.get('arab', 'Tidak tersedia')}\n"
                            response_text += f"   - Terjemah: {hadith.get('terjemah', 'Tidak tersedia')}\n\n"
                        
                        full_response = response_text
                        yield await format_sse_message("message", response_text)
                    else:
                        full_response = "Maaf, tidak ditemukan hadits yang relevan untuk pertanyaan Anda."
                        yield await format_sse_message("message", full_response)
            
            yield await format_sse_message("complete", "Pencarian selesai")
        
        if session_id and full_response.strip():
            session_manager.add_message(session_id, "assistant", full_response.strip())
            yield await format_sse_message("session_updated", session_id)
    
    except Exception as e:
        logger.error(f"Error in stream_hadits_response: {e}")
        error_message = f"Terjadi kesalahan: {str(e)}"
        yield await format_sse_message("error", error_message)
        
        if session_id:
            session_manager.add_message(session_id, "assistant", error_message)

# Routes

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    
    llm_status = "available" if GEMINI_API_KEY else "unavailable"
    retrieval_status = "ok" if retrieval_initialized else "error"
    
    overall_status = "healthy" if retrieval_initialized else "degraded"
    
    return HealthResponse(
        status=overall_status,
        message="Hadis AI Backend is running",
        retrieval_status=retrieval_status,
        llm_status=llm_status
    )

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Main endpoint for hadith questions with streaming SSE response (without session)"""
    
    if not retrieval_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Retrieval system not available. Please check system status."
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    # Return SSE stream
    return EventSourceResponse(
        stream_hadits_response(request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/sessions/{session_id}/ask")
async def ask_question_in_session(session_id: str, request: QuestionRequest):
    """Ask a question within a specific chat session"""
    
    if not retrieval_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Retrieval system not available. Please check system status."
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    # Check if session exists
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    # Return SSE stream with session context
    return EventSourceResponse(
        stream_hadits_response(request.question, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/sessions")
async def get_sessions():
    """Get all chat sessions"""
    sessions = session_manager.get_all_sessions()
    return {
        "sessions": sessions,
        "total": len(sessions)
    }

@app.post("/sessions")
async def create_session(request: CreateSessionRequest):
    """Create a new chat session"""
    session = session_manager.create_session(request.title)
    return {
        "session_id": session.session_id,
        "title": session.title,
        "created_at": session.created_at
    }

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific session with all messages"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    return session

@app.put("/sessions/{session_id}")
async def update_session(session_id: str, request: CreateSessionRequest):
    """Update session title"""
    if not request.title:
        raise HTTPException(
            status_code=400,
            detail="Title is required"
        )
    
    success = session_manager.update_session_title(session_id, request.title)
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    return {"message": "Session updated successfully"}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    return {"message": "Session deleted successfully"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Hadis AI Backend",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 