import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List
import asyncpg
import redis
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BackgroundWorker:
    """Background task processor for D&D AI system"""
    
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        parsed_url = urlparse(redis_url)

        self.redis_client = redis.Redis(
            host=parsed_url.hostname,
            port=parsed_url.port,
            db=int(parsed_url.path.lstrip("/")),
            decode_responses=True
        )
        self.db_pool = None
        self.should_run = True
    
    async def init_db(self):
        """Initialize database connection pool"""
        try:
            self.db_pool = await asyncpg.create_pool(
                os.getenv('DATABASE_URL', 'postgresql://dnd_user:dnd_secure_pass_2024@localhost/dnd_game'),
                min_size=2,
                max_size=10
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def process_world_events(self):
        """Process pending world events"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get unprocessed events
                events = await conn.fetch("""
                    SELECT id, event_type, description, data 
                    FROM world_events 
                    WHERE processed = FALSE 
                    ORDER BY created_at ASC
                    LIMIT 10
                """)
                
                for event in events:
                    await self.handle_world_event(conn, event)
                    
                    # Mark as processed
                    await conn.execute(
                        "UPDATE world_events SET processed = TRUE WHERE id = $1",
                        event['id']
                    )
                    
                logger.info(f"Processed {len(events)} world events")
                
        except Exception as e:
            logger.error(f"Error processing world events: {e}")
    
    async def handle_world_event(self, conn, event):
        """Handle individual world event"""
        event_type = event['event_type']
        data = json.loads(event['data']) if event['data'] else {}
        
        if event_type == 'weather_change':
            await self.update_weather(conn, data)
        elif event_type == 'merchant_arrival':
            await self.spawn_merchant(conn, data)
        elif event_type == 'dragon_sighting':
            await self.handle_dragon_event(conn, data)
        
        logger.info(f"Handled {event_type}: {event['description']}")
    
    async def update_weather(self, conn, data):
        """Update weather across all active sessions"""
        weather_update = {
            'weather': data.get('severity', 'clear'),
            'temperature': data.get('temperature', 'moderate'),
            'visibility': data.get('visibility', 'clear')
        }
        
        # Update all active sessions
        await conn.execute("""
            UPDATE game_sessions 
            SET world_state = world_state || $1::jsonb,
                updated_at = CURRENT_TIMESTAMP
            WHERE updated_at > NOW() - INTERVAL '24 hours'
        """, json.dumps(weather_update))
    
    async def spawn_merchant(self, conn, data):
        """Spawn merchant in active sessions"""
        merchant_data = {
            'npcs': {
                'traveling_merchant': {
                    'name': 'Gareth the Wanderer',
                    'goods': data.get('goods', []),
                    'gold': data.get('gold', 100),
                    'location': 'village_square'
                }
            }
        }
        
        await conn.execute("""
            UPDATE game_sessions 
            SET world_state = world_state || $1::jsonb,
                updated_at = CURRENT_TIMESTAMP
            WHERE updated_at > NOW() - INTERVAL '12 hours'
        """, json.dumps(merchant_data))
    
    async def handle_dragon_event(self, conn, data):
        """Handle dragon sighting event"""
        dragon_data = {
            'threats': {
                'red_dragon': {
                    'location': 'northern_mountains',
                    'threat_level': data.get('threat_level', 'medium'),
                    'last_seen': datetime.utcnow().isoformat()
                }
            }
        }
        
        await conn.execute("""
            UPDATE game_sessions 
            SET world_state = world_state || $1::jsonb,
                updated_at = CURRENT_TIMESTAMP
        """, json.dumps(dragon_data))
    
    async def cleanup_old_sessions(self):
        """Clean up old inactive sessions"""
        try:
            async with self.db_pool.acquire() as conn:
                # Archive sessions older than 7 days
                archived = await conn.execute("""
                    DELETE FROM game_sessions 
                    WHERE updated_at < NOW() - INTERVAL '7 days'
                """)
                
                # Clean up orphaned characters
                orphaned = await conn.execute("""
                    DELETE FROM characters 
                    WHERE session_id NOT IN (SELECT session_id FROM game_sessions)
                """)
                
                logger.info(f"Cleaned up {archived} old sessions and {orphaned} orphaned characters")
                
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
    
    async def generate_random_events(self):
        """Generate random world events"""
        import random
        
        events = [
            {
                'type': 'weather_change',
                'description': 'Weather patterns shift across the realm',
                'data': {'severity': random.choice(['light', 'moderate', 'heavy'])}
            },
            {
                'type': 'merchant_arrival', 
                'description': 'A new merchant arrives in town',
                'data': {'gold': random.randint(50, 200)}
            },
            {
                'type': 'beast_migration',
                'description': 'Wild creatures migrate through the area', 
                'data': {'creature_type': random.choice(['wolves', 'deer', 'bears'])}
            }
        ]
        
        # 20% chance to generate an event each cycle
        if random.random() < 0.2:
            event = random.choice(events)
            
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO world_events (event_type, description, data)
                        VALUES ($1, $2, $3)
                    """, event['type'], event['description'], json.dumps(event['data']))
                    
                logger.info(f"Generated random event: {event['description']}")
                
            except Exception as e:
                logger.error(f"Error generating random event: {e}")
    
    async def update_metrics(self):
        """Update system metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Count active sessions
                active_sessions = await conn.fetchval("""
                    SELECT COUNT(*) FROM game_sessions 
                    WHERE updated_at > NOW() - INTERVAL '1 hour'
                """)
                
                # Count total characters
                total_characters = await conn.fetchval("SELECT COUNT(*) FROM characters")
                
                # Store metrics in Redis for Prometheus
                self.redis_client.set('metric:active_sessions', active_sessions)
                self.redis_client.set('metric:total_characters', total_characters)
                
                logger.info(f"Updated metrics: {active_sessions} active sessions, {total_characters} characters")
                
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def run_worker_cycle(self):
        """Run one cycle of background tasks"""
        try:
            await self.process_world_events()
            await self.generate_random_events()
            await self.update_metrics()
            
            # Less frequent tasks
            import random
            if random.random() < 0.1:  # 10% chance
                await self.cleanup_old_sessions()
                
        except Exception as e:
            logger.error(f"Error in worker cycle: {e}")
    
    async def run(self):
        """Main worker loop"""
        logger.info("Starting background worker...")
        
        await self.init_db()
        
        cycle_count = 0
        while self.should_run:
            try:
                cycle_count += 1
                logger.info(f"Starting worker cycle {cycle_count}")
                
                await self.run_worker_cycle()
                
                # Wait 30 seconds between cycles
                await asyncio.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                self.should_run = False
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
        
        # Cleanup
        if self.db_pool:
            await self.db_pool.close()
        
        logger.info("Background worker stopped")

async def main():
    """Main entry point"""
    worker = BackgroundWorker()
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())