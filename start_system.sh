echo "🐲 Starting D&D AI System..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

# Create necessary directories
mkdir -p logs models nginx/ssl monitoring/grafana/dashboards monitoring/grafana/datasources

# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  No NVIDIA GPU detected - falling back to CPU inference"
fi

# Function to wait for service
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s $url > /dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts - waiting 5s..."
        sleep 5
        ((attempt++))
    done
    
    echo "❌ $service_name failed to start within timeout"
    return 1
}

# Stop any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose down

# Start core services first
echo "🗄️  Starting database services..."
docker-compose up -d postgres redis

# Wait for databases
wait_for_service "http://localhost:5432" "PostgreSQL" || exit 1
wait_for_service "http://localhost:6379" "Redis" || exit 1

# Start AI services
echo "🤖 Starting AI services..."
docker-compose up -d vllm_server

# Wait for vLLM
wait_for_service "http://localhost:8000/health" "vLLM Server" || exit 1

# Start application services
echo "🚀 Starting application services..."
docker-compose up -d api_server worker

# Wait for API
wait_for_service "http://localhost:8080/health" "API Server" || exit 1

# Start frontend
echo "🎨 Starting frontend..."
docker-compose up -d frontend

# Start monitoring
echo "📊 Starting monitoring services..."
docker-compose up -d prometheus grafana node_exporter nvidia_gpu_exporter

# Start logging
echo "📝 Starting logging services..."
docker-compose up -d elasticsearch kibana filebeat

# Start reverse proxy
echo "🌐 Starting reverse proxy..."
docker-compose up -d nginx

echo ""
echo "🎉 D&D AI System is starting up!"
echo ""
echo "Services will be available at:"
echo "  🎮 Game UI:      http://localhost:3000"
echo "  📡 API Docs:     http://localhost:8080/docs"
echo "  📊 Grafana:      http://localhost:3001 (admin/admin123)"
echo "  🔍 Prometheus:   http://localhost:9090"
echo "  📋 Kibana:       http://localhost:5601"
echo ""
echo "💡 Run 'docker-compose logs -f api_server' to watch the logs"
echo "💡 Run 'docker-compose ps' to check service status"

# Show final status
sleep 10
echo ""
echo "🔍 Final service status:"
docker-compose ps