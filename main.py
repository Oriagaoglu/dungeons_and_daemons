from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
import asyncio
import json
import time
import logging
import psutil
import GPUtil
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel
import redis
import asyncpg
from contextlib import asynccontextmanager
import httpx
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import uvicorn
import os 
from dotenv import load_dotenv
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('dnd_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('dnd_request_duration_seconds', 'Request duration')
ACTIVE_SESSIONS = Gauge('dnd_active_sessions', 'Active WebSocket sessions')
GPU_MEMORY_USAGE = Gauge('dnd_gpu_memory_mb', 'GPU memory usage in MB')
MODEL_INFERENCE_TIME = Histogram('dnd_model_inference_seconds', 'Model inference time')

class ContextManager:
    """Intelligent context management for memory efficiency"""
    
    def __init__(self, db_pool, max_tokens: int = 1024, window_size: int = 8):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")        
        self.redis_client = redis.from_url(redis_url)
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.db_pool = db_pool
    
    async def get_context(self, session_id: str) -> List[Dict]:
        """Get optimized context for the session"""
        # Get from Redis cache first
        cached = self.redis_client.get(f"context:{session_id}")
        if cached:
            return json.loads(cached)
        
        # Fallback to database
        return await self.load_from_db(session_id)
    
    async def add_message(self, session_id: str, message: Dict):
        """Add message and manage context size"""
        context = await self.get_context(session_id)
        context.append(message)
        
        # Apply sliding window + compression
        if len(context) > self.window_size:
            context = await self.compress_context(context)
        
        # Cache the updated context
        self.redis_client.setex(
            f"context:{session_id}", 
            3600,  # 1 hour TTL
            json.dumps(context)
        )
        
        return context
    
    async def compress_context(self, messages: List[Dict]) -> List[Dict]:
        """Compress older messages while preserving important information"""
        if len(messages) <= self.window_size:
            return messages
        
        recent_messages = messages[-self.window_size:]
        older_messages = messages[:-self.window_size]
        
        # Extract key information from older messages
        character_info = self.extract_character_info(older_messages)
        world_state = self.extract_world_state(older_messages)
        important_events = self.extract_important_events(older_messages)
        
        # Create compressed summary
        summary = {
            "role": "system",
            "content": f"""Character: {character_info}
World State: {world_state}
Key Events: {important_events}"""
        }
        
        return [summary] + recent_messages
    
    async def load_from_db(self, session_id: str) -> List[Dict]:
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT messages FROM game_sessions WHERE session_id = $1",
                session_id
            )
            return row["messages"] if row and row["messages"] else []

    def extract_character_info(self, messages: List[Dict]) -> str:
        # Extract character sheets, stats, equipment
        return "Thistle Swiftfoot - Halfling Rogue, Level 1, HP: 8/8"
    
    def extract_world_state(self, messages: List[Dict]) -> str:
        # Extract current location, time, weather, etc.
        return "Location: Harmony's Edge village, Time: Afternoon, Weather: Clear"
    
    def extract_important_events(self, messages: List[Dict]) -> str:
        # Extract major story beats, combat results, decisions
        return "Dragon attacked village, Crystal stolen, Character created"

class GameState:
    """Manages persistent game state"""
    
    def __init__(self):
        self.db_pool = None

   

    async def init_db(self):
        """Initialize database connection pool"""
        DATABASE_URL = os.getenv("DATABASE_URL")
        self.db_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=5,
            max_size=20
        )
    
    async def save_character(self, session_id: str, character_data: Dict):
        """Save character to database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO characters (
                    session_id, name, race, class_name, level, stats, equipment, backstory, hp, max_hp, updated_at
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                )
                ON CONFLICT (session_id) DO UPDATE SET
                    name = $2,
                    race = $3,
                    class_name = $4,
                    level = $5,
                    stats = $6,
                    equipment = $7,
                    backstory = $8,
                    hp = $9,
                    max_hp = $10,
                    updated_at = $11
            """, 
            session_id,
            character_data.get("name"),
            character_data.get("race"),
            character_data.get("class_name"),
            character_data.get("level", 1),
            character_data.get("stats", {}),        # ← fixed
            character_data.get("equipment", []),    # ← fixed
            character_data.get("backstory", ""),
            character_data.get("hp", 8),
            character_data.get("max_hp", 8),
            datetime.utcnow()
            )

    
    async def get_character(self, session_id: str) -> Optional[Dict]:
        """Load character from database"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT session_id, name, race, class_name, level, stats, equipment, backstory, hp, max_hp
                FROM characters
                WHERE session_id = $1
            """, session_id)

            if not row:
                return None

            return {
                "session_id": row["session_id"],
                "name": row["name"],
                "race": row["race"],
                "class_name": row["class_name"],
                "level": row["level"],
                "stats": row["stats"],
                "equipment": row["equipment"],
                "backstory": row["backstory"],
                "hp": row["hp"],
                "max_hp": row["max_hp"]
            }


class SystemMonitoring:
    """System performance monitoring"""
    
    @staticmethod
    def get_gpu_stats():
        """Get GPU memory usage"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # RTX 2060
                GPU_MEMORY_USAGE.set(gpu.memoryUsed)
                return {
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "utilization": gpu.load * 100
                }
        except:
            pass
        return None
    
    @staticmethod
    def get_system_stats():
        """Get system resource usage"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3)
        }

class LLMClient:
    """Enhanced LLM client with monitoring and error handling"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("VLLM_URL", "http://localhost:8000")

        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def generate_response(self, messages: List[Dict], session_id: str) -> str:
        """Generate response with timing and error handling"""
        start_time = time.time()
        
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ",
                    "messages": messages,
                    "temperature": 0.8,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            data = response.json()
            inference_time = time.time() - start_time
            MODEL_INFERENCE_TIME.observe(inference_time)
            
            logger.info(f"Generated response for session {session_id} in {inference_time:.2f}s")
            
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"LLM generation failed for session {session_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="AI service temporarily unavailable")

# Pydantic models
class ChatMessage(BaseModel):
    content: str
    session_id: str

class CharacterSheet(BaseModel):
    name: str
    race: str
    class_name: str
    level: int = 1
    stats: Dict[str, int]
    equipment: List[str] = []
    backstory: str = ""

# Global instances
game_state = GameState()
# context_manager = ContextManager()
system_monitor = SystemMonitoring()
llm_client = LLMClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting D&D AI system...")
    await game_state.init_db()

    global context_manager
    context_manager = ContextManager(game_state.db_pool)
    # Background monitoring task
    async def monitor_system():
        while True:
            system_monitor.get_gpu_stats()
            await asyncio.sleep(30)  # Update every 30 seconds
    
    monitor_task = asyncio.create_task(monitor_system())
    
    yield
    
    # Shutdown
    monitor_task.cancel()
    if game_state.db_pool:
        await game_state.db_pool.close()
    await llm_client.client.aclose()
    logger.info("D&D AI system shut down")

# FastAPI app
app = FastAPI(
    title="D&D AI System",
    description="Advanced AI-powered Dungeons & Dragons experience",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        ACTIVE_SESSIONS.set(len(self.active_connections))
        logger.info(f"Session {session_id} connected")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            ACTIVE_SESSIONS.set(len(self.active_connections))
            logger.info(f"Session {session_id} disconnected")
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

manager = ConnectionManager()

# Routes
@app.get("/")
async def read_root():
    return {"message": "D&D AI System - Ready for Adventure!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_stats = system_monitor.get_gpu_stats()
    system_stats = system_monitor.get_system_stats()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "gpu": gpu_stats,
        "system": system_stats,
        "active_sessions": len(manager.active_connections)
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type", "chat")
            
            if message_type == "chat":
                user_message = data["content"]
                
                # Add user message to context
                await context_manager.add_message(session_id, {
                    "role": "user",
                    "content": user_message
                })
                
                # Get current context
                context = await context_manager.get_context(session_id)
                
                # Generate AI response
                ai_response = await llm_client.generate_response(context, session_id)
                
                # Add AI response to context
                await context_manager.add_message(session_id, {
                    "role": "assistant", 
                    "content": ai_response
                })
                
                # Send response to client
                await manager.send_message(session_id, {
                    "type": "chat",
                    "content": ai_response,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            elif message_type == "dice_roll":
                # Handle dice rolling
                dice_result = {
                    "type": "dice_result",
                    "roll": data["dice"],
                    "result": 15,  # Would implement actual dice logic
                    "timestamp": datetime.utcnow().isoformat()
                }
                await manager.send_message(session_id, dice_result)
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)

@app.post("/character")
async def create_character(character: CharacterSheet, session_id: str):
    """Create or update character sheet"""
    REQUEST_COUNT.labels(method="POST", endpoint="/character").inc()
    
    await game_state.save_character(session_id, character.dict())
    
    return {"message": "Character saved successfully", "character": character}

@app.get("/character/{session_id}")
async def get_character(session_id: str):
    """Get character sheet"""
    REQUEST_COUNT.labels(method="GET", endpoint="/character").inc()
    
    character = await game_state.get_character(session_id)
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    return character

@app.post("/batch/world-update")
async def trigger_world_update(background_tasks: BackgroundTasks):
    """Trigger batch world state update"""
    background_tasks.add_task(process_world_events)
    return {"message": "World update queued"}

async def process_world_events():
    """Background task to process world events"""
    logger.info("Processing world events...")
    
    # Simulate world updates
    await asyncio.sleep(5)
    
    # Update weather, NPCs, economy, etc.
    world_updates = {
        "weather": "Thunderstorm approaching",
        "npc_movements": ["Merchant arrived", "Guards changed shift"],
        "economic_changes": ["Sword prices increased 10%"]
    }
    
    # Broadcast to all active sessions
    for session_id in manager.active_connections:
        await manager.send_message(session_id, {
            "type": "world_update",
            "updates": world_updates,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    logger.info("World events processed")

# Static file serving for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )