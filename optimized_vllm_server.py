# optimized_vllm_server.py - RTX 2060 6GB Optimized Configuration
import subprocess
import os
import sys
import time
import logging
import psutil
import threading
from typing import Optional
import signal

os.makedirs(os.path.dirname("./app/logs/vllm_server.log"), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./app/logs/vllm_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VLLMServerManager:
    """Intelligent vLLM server management with memory optimization"""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.should_restart = True
        self.restart_count = 0
        self.max_restarts = 5
        
        # RTX 2060 6GB Optimized Settings
        self.config = {
            "model": "TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ",
            "quantization": "awq",
            "dtype": "auto",
            "max_model_len": 256,  # Reduced for 6GB VRAM
            "gpu_memory_utilization": 0.55,  # Conservative for stability
            "swap_space": 2,  # Enable CPU swap for overflow
            "disable_custom_all_reduce": True,  # Reduce memory overhead
            "enforce_eager": True,  # Disable CUDA graphs for memory savings
            "max_num_seqs": 8,  # Limit concurrent sequences
            "max_num_batched_tokens": 256,  # Batch size optimization
            "enable_chunked_prefill": True,  # Memory efficient prefill
            "max_num_on_the_fly": 1,  # Limit on-the-fly sequences
        }
    
    def setup_environment(self):
        """Configure environment for optimal memory usage"""
        # CUDA memory allocation strategies
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"  # RTX 2060 architecture
        
        # Python memory optimizations
        os.environ["PYTHONHASHSEED"] = "0"
        os.environ["OMP_NUM_THREADS"] = str(min(4, psutil.cpu_count()))
        
        # Reduce PyTorch memory fragmentation
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def get_memory_info(self):
        """Get current system and GPU memory usage"""
        memory = psutil.virtual_memory()
        info = {
            "system_memory_percent": memory.percent,
            "system_memory_available_gb": memory.available / (1024**3)
        }

        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                info.update({
                    "gpu_memory_used": gpu.memoryUsed,
                    "gpu_memory_total": gpu.memoryTotal,
                    "gpu_utilization": gpu.load * 100,
                    "gpu_temperature": gpu.temperature
                })
        except ImportError:
            logger.warning("GPUtil not available for GPU monitoring")
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
        
        return info
    
    def check_system_resources(self) -> bool:
        """Check if system has enough resources to run the model"""
        memory_info = self.get_memory_info()
    
        
        # Check system memory (need at least 2GB free)
        if memory_info["system_memory_available_gb"] < 2.0:
            logger.error(f"Insufficient system memory: {memory_info['system_memory_available_gb']:.1f}GB available")
            return False
        
        return True
    
    def build_command(self) -> list:
        """Build vLLM server command with optimized parameters"""
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config["model"],
            "--quantization", self.config["quantization"],
            "--dtype", self.config["dtype"],
            "--max-model-len", str(self.config["max_model_len"]),
            "--gpu-memory-utilization", str(self.config["gpu_memory_utilization"]),
            "--swap-space", str(self.config["swap_space"]),
            "--max-num-seqs", str(self.config["max_num_seqs"]),
            "--max-num-batched-tokens", str(self.config["max_num_batched_tokens"]),
            "--port", "8000",
            "--host", "0.0.0.0",
            "--served-model-name", self.config["model"],
            "--disable-log-stats",  # Reduce logging overhead
        ]
        
        # Add conditional flags
        # if self.config["disable_custom_all_reduce"]:
        #     cmd.append("--disable-custom-all-reduce")
        
        if self.config["enforce_eager"]:
            cmd.append("--enforce-eager")
        
        # if self.config["enable_chunked_prefill"]:
        #     cmd.append("--enable-chunked-prefill")
        
        return cmd
    
    def start_server(self):
        """Start the vLLM server with monitoring"""
        if not self.check_system_resources():
            logger.error("System resources insufficient, cannot start server")
            return False
        
        cmd = self.build_command()
        logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Start output monitoring thread
            monitor_thread = threading.Thread(
                target=self.monitor_output,
                daemon=True
            )
            monitor_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            return False
    
    def monitor_output(self):
        """Monitor server output for errors and memory issues"""
        if not self.process:
            return
        
        while self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    
                    # Log important messages
                    if any(keyword in line.lower() for keyword in ['error', 'warning', 'cuda', 'memory']):
                        logger.warning(f"vLLM: {line}")
                    elif 'INFO' in line:
                        logger.info(f"vLLM: {line}")
                    
                    # Check for OOM errors
                    if 'out of memory' in line.lower() or 'cuda error' in line.lower():
                        logger.error(f"Memory error detected: {line}")
                        self.handle_memory_error()
                        
            except Exception as e:
                logger.error(f"Error monitoring output: {e}")
                break
    
    def handle_memory_error(self):
        """Handle out of memory errors by reducing parameters"""
        logger.warning("Handling memory error by reducing model parameters")
        
        # Reduce memory usage parameters
        self.config["max_model_len"] = max(256, self.config["max_model_len"] - 256)
        self.config["gpu_memory_utilization"] = max(0.5, self.config["gpu_memory_utilization"] - 0.1)
        self.config["max_num_seqs"] = max(1, self.config["max_num_seqs"] - 2)
        self.config["max_num_batched_tokens"] = max(128, self.config["max_num_batched_tokens"] - 128)
        
        logger.info(f"Reduced parameters: max_len={self.config['max_model_len']}, "
                   f"gpu_util={self.config['gpu_memory_utilization']}, "
                   f"max_seqs={self.config['max_num_seqs']}")
        
        # Restart server with reduced parameters
        self.restart_server()
    
    def restart_server(self):
        """Restart the server with current configuration"""
        if self.restart_count >= self.max_restarts:
            logger.error(f"Maximum restarts ({self.max_restarts}) reached, giving up")
            return False
        
        logger.info(f"Restarting server (attempt {self.restart_count + 1}/{self.max_restarts})")
        self.restart_count += 1
        
        # Stop current process
        self.stop_server()
        
        # Wait a moment for cleanup
        time.sleep(5)
        
        # Clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        # Start with updated configuration
        return self.start_server()
    
    def stop_server(self):
        """Gracefully stop the vLLM server"""
        if self.process:
            logger.info("Stopping vLLM server...")
            
            try:
                # Try graceful shutdown first
                self.process.terminate()
                
                # Wait up to 30 seconds for graceful shutdown
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    logger.warning("Graceful shutdown timed out, forcing kill")
                    self.process.kill()
                    self.process.wait()
                
                logger.info("vLLM server stopped")
                
            except Exception as e:
                logger.error(f"Error stopping server: {e}")
            
            finally:
                self.process = None
    
    def is_healthy(self) -> bool:
        """Check if the server is running and healthy"""
        if not self.process or self.process.poll() is not None:
            return False
        
        try:
            import requests
            response = requests.get("http://localhost:8000/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def run_with_monitoring(self):
        """Run server with automatic monitoring and restart"""
        logger.info("Starting vLLM server with monitoring...")
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.should_restart = False
            self.stop_server()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        while self.should_restart:
            try:
                # Start server
                if not self.start_server():
                    logger.error("Failed to start server")
                    break
                
                # Monitor server health
                while self.should_restart and self.process:
                    time.sleep(30)  # Check every 30 seconds
                    
                    # Log memory usage
                    memory_info = self.get_memory_info()
                    logger.info(f"Memory status: {memory_info}")
                    
                    # Check if process is still running
                    if self.process.poll() is not None:
                        logger.error("Server process died unexpectedly")
                        break
                    
                    # Check health endpoint
                    if not self.is_healthy():
                        logger.error("Server health check failed")
                        break
                
                # If we get here, server died or became unhealthy
                if self.should_restart:
                    logger.warning("Server needs restart")
                    time.sleep(10)  # Wait before restart
                    
            except Exception as e:
                logger.error(f"Unexpected error in monitoring loop: {e}")
                time.sleep(10)
        
        logger.info("Server monitoring stopped")

class MemoryProfiler:
    """Memory profiling and optimization utilities"""
    
    @staticmethod
    def log_memory_usage():
        """Log detailed memory usage"""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                    memory_cached = torch.cuda.memory_cached(i) / 1024**3       # GB
                    
                    logger.info(f"GPU {i} Memory - Allocated: {memory_allocated:.2f}GB, "
                               f"Reserved: {memory_reserved:.2f}GB, Cached: {memory_cached:.2f}GB")
        except Exception as e:
            logger.error(f"Error logging GPU memory: {e}")
        
        # System memory
        memory = psutil.virtual_memory()
        logger.info(f"System Memory - Used: {memory.used / 1024**3:.2f}GB, "
                   f"Available: {memory.available / 1024**3:.2f}GB, "
                   f"Percent: {memory.percent:.1f}%")
    
    @staticmethod
    def optimize_for_rtx2060():
        """Apply RTX 2060 specific optimizations"""
        logger.info("Applying RTX 2060 6GB optimizations...")
        
        # Set optimal CUDA settings
        os.environ.update({
            "CUDA_LAUNCH_BLOCKING": "0",
            "CUDA_CACHE_DISABLE": "0",
            "CUDA_CACHE_MAXSIZE": "134217728",  # 128MB
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
            "TORCH_CUDNN_V8_API_ENABLED": "1",
        })
        
        try:
            import torch
            if torch.cuda.is_available():
                # Enable memory pool and optimize settings
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                
                # Set memory fraction (leave some for system)
                torch.cuda.set_per_process_memory_fraction(0.8)
                
                logger.info("RTX 2060 optimizations applied successfully")
        except Exception as e:
            logger.error(f"Error applying optimizations: {e}")

def main():
    """Main entry point"""
    logger.info("Starting optimized vLLM server for RTX 2060 6GB")
    
    # Apply RTX 2060 optimizations
    MemoryProfiler.optimize_for_rtx2060()
    
    # Initialize server manager
    server_manager = VLLMServerManager()
    server_manager.setup_environment()
    
    # Log initial memory state
    MemoryProfiler.log_memory_usage()
    
    try:
        # Run server with monitoring
        server_manager.run_with_monitoring()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        server_manager.stop_server()
        logger.info("vLLM server manager shutdown complete")

if __name__ == "__main__":
    main()

# Additional utility functions for memory management

class ContextWindow:
    """Smart context window management for memory efficiency"""
    
    def __init__(self, max_tokens: int = 1024):
        self.max_tokens = max_tokens
        self.compression_ratio = 0.7  # Compress to 70% when needed
    
    def compress_messages(self, messages: list) -> list:
        """Compress message history while preserving important context"""
        if not messages:
            return messages
        
        # Estimate token count (rough approximation)
        total_tokens = sum(len(msg.get('content', '').split()) * 1.3 for msg in messages)
        
        if total_tokens <= self.max_tokens:
            return messages
        
        # Keep system message and recent messages
        system_msgs = [msg for msg in messages if msg.get('role') == 'system']
        user_assistant_msgs = [msg for msg in messages if msg.get('role') in ['user', 'assistant']]
        
        # Calculate how many recent messages to keep
        target_tokens = int(self.max_tokens * self.compression_ratio)
        keep_count = self.calculate_keep_count(user_assistant_msgs, target_tokens)
        
        recent_msgs = user_assistant_msgs[-keep_count:] if keep_count > 0 else []
        
        # Create summary of older messages
        if len(user_assistant_msgs) > keep_count:
            older_msgs = user_assistant_msgs[:-keep_count]
            summary = self.create_summary(older_msgs)
            summary_msg = {
                'role': 'system',
                'content': f"Previous conversation summary: {summary}"
            }
            return system_msgs + [summary_msg] + recent_msgs
        
        return system_msgs + recent_msgs
    
    def calculate_keep_count(self, messages: list, target_tokens: int) -> int:
        """Calculate how many recent messages to keep within token budget"""
        current_tokens = 0
        keep_count = 0
        
        # Count from the end
        for msg in reversed(messages):
            msg_tokens = len(msg.get('content', '').split()) * 1.3
            if current_tokens + msg_tokens <= target_tokens:
                current_tokens += msg_tokens
                keep_count += 1
            else:
                break
        
        return keep_count
    
    def create_summary(self, messages: list) -> str:
        """Create a concise summary of message history"""
        # Extract key information
        character_info = []
        locations = []
        actions = []
        
        for msg in messages:
            content = msg.get('content', '').lower()
            
            # Extract character information
            if any(word in content for word in ['character', 'name', 'class', 'race', 'level']):
                character_info.append(content[:100])
            
            # Extract location information
            if any(word in content for word in ['village', 'town', 'dungeon', 'forest', 'mountain']):
                locations.append(content[:100])
            
            # Extract important actions
            if any(word in content for word in ['attack', 'cast', 'move', 'search', 'talk']):
                actions.append(content[:100])
        
        # Build summary
        summary_parts = []
        
        if character_info:
            summary_parts.append(f"Character: {'; '.join(character_info[:2])}")
        
        if locations:
            summary_parts.append(f"Locations: {'; '.join(locations[:2])}")
        
        if actions:
            summary_parts.append(f"Recent actions: {'; '.join(actions[-3:])}")
        
        return " | ".join(summary_parts) if summary_parts else "Previous conversation occurred."

# Export key classes for use in other modules
__all__ = ['VLLMServerManager', 'MemoryProfiler', 'ContextWindow']