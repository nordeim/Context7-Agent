# 🐳 Context7 AI Agent - Docker Deployment Guide

## 📋 Overview

This guide provides comprehensive instructions for deploying the Context7 AI Agent using Docker in various environments, from local development to production clusters.

## 🎯 Quick Start (30 seconds)

```bash
# 1. Pull and run
docker run -it \
  -e OPENAI_API_KEY=your-key-here \
  -v $(pwd)/data:/app/data \
  context7/agent:latest

# 2. That's it! The agent starts immediately
```

## 🏗️ Deployment Options

### 1. 🚀 Local Development

#### Single Container
```bash
# Build locally
docker build -t context7:dev -f docker/Dockerfile .

# Run with development settings
docker run -it \
  --name context7-dev \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/src:/app/src:ro \
  context7:dev
```

#### Development Stack
```bash
# Use compose for full development environment
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f
```

### 2. 🐳 Production Single Instance

#### Basic Production
```bash
# Run production image
docker run -d \
  --name context7-prod \
  --restart unless-stopped \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v context7_data:/app/data \
  -v context7_logs:/app/logs \
  context7/agent:2.0.0
```

#### With Health Check
```bash
docker run -d \
  --name context7-prod \
  --health-cmd="python -c 'import sys; sys.path.insert(0, \"src\"); from config import config; print(\"OK\")'" \
  --health-interval=30s \
  --health-retries=3 \
  --health-timeout=10s \
  context7/agent:2.0.0
```

### 3. ☁️ Cloud Deployments

#### AWS ECS

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name context7-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://docker/ecs-task-definition.json

# Run service
aws ecs create-service \
  --cluster context7-cluster \
  --service-name context7-service \
  --task-definition context7-task \
  --desired-count 2
```

#### Kubernetes

```bash
# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Create secret
kubectl create secret generic openai-secret \
  --from-literal=api-key=your-key-here

# Check status
kubectl get pods -l app=context7-agent
```

### 4. 🏭 Enterprise Deployment

#### High Availability Setup
```bash
# Production compose
docker-compose -f docker/docker-compose.prod.yml up -d

# Scale horizontally
docker-compose -f docker/docker-compose.prod.yml up -d --scale context7-agent=3
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | ✅ |
| `CONTEXT7_DEFAULT_THEME` | UI theme | cyberpunk | ❌ |
| `CONTEXT7_MAX_HISTORY` | Max conversation history | 50 | ❌ |
| `CONTEXT7_ANIMATION_SPEED` | Animation speed | 0.05 | ❌ |
| `LOG_LEVEL` | Logging level | INFO | ❌ |
| `REDIS_URL` | Redis connection | redis://redis:6379 | ❌ |

### Volume Mounts

| Path | Purpose | Persistence |
|------|---------|-------------|
| `/app/data` | Conversation history | ✅ Permanent |
| `/app/logs` | Application logs | ✅ Permanent |
| `/app/config` | Configuration files | ✅ Permanent |

## 🔒 Security Best Practices

### 1. Non-root User
The container runs as `context7` user (UID 1000) for security.

### 2. Secret Management

#### Docker Secrets
```bash
# Create secret
echo "your-api-key" | docker secret create openai_key -

# Use in compose
secrets:
  - openai_key

secrets:
  openai_key:
    external: true
```

#### Environment Files
```bash
# Create secure env file
cat > .env.prod << EOF
OPENAI_API_KEY=your-key-here
CONTEXT7_MAX_HISTORY=100
EOF

chmod 600 .env.prod
```

### 3. Network Security
```yaml
networks:
  context7_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## 📊 Monitoring & Observability

### Health Checks
```bash
# Manual health check
docker exec context7-prod python -c "import sys; sys.path.insert(0, 'src'); from config import config; print('OK')"

# Container stats
docker stats context7-prod
```

### Logging
```bash
# View logs
docker logs -f context7-prod

# Structured logging
docker logs --tail 100 --timestamps context7-prod | jq
```

### Metrics with Prometheus
```yaml
# Add to compose
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
```

## 🚀 Performance Tuning

### Resource Limits
```yaml
services:
  context7-agent:
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
```

### Caching Strategy
```bash
# Enable Redis caching
docker-compose -f docker/docker-compose.prod.yml up -d redis
```

### Load Balancing
```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
```

## 🔄 CI/CD Integration

### GitHub Actions
```yaml
name: Docker Build & Deploy
on:
  push:
    branches: [main]
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          push: true
          tags: context7/agent:${{ github.sha }}
```

### GitLab CI
```yaml
stages:
  - build
  - deploy

docker-build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

## 🐛 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker logs context7-agent

# Verify environment
docker exec context7-agent env | grep OPENAI

# Check permissions
docker exec context7-agent ls -la /app/data/
```

#### Slow Performance
```bash
# Check resource usage
docker stats

# Optimize with limits
docker update --memory=1g --cpus=1 context7-agent
```

#### Network Issues
```bash
# Test connectivity
docker exec context7-agent curl -I https://api.openai.com

# Check DNS
docker exec context7-agent nslookup api.openai.com
```

## 🧪 Testing Deployments

### Local Testing
```bash
# Test with minimal config
docker run --rm \
  -e OPENAI_API_KEY=test-key \
  context7/agent:latest \
  python -c "from config import config; print('Config loaded')"
```

### Integration Testing
```bash
# Run tests in container
docker run --rm \
  -v $(pwd)/tests:/app/tests \
  context7/agent:latest \
  pytest tests/integration/
```

## 📞 Support & Resources

### Quick Commands
```bash
# Get help
docker run --rm context7/agent:latest --help

# Check version
docker run --rm context7/agent:latest python -c "import src; print(src.__version__)"
```

### Community Resources
- [GitHub Issues](https://github.com/nordeim/Context7-MCP-RAG-Agent/issues)
- [Discord Community](https://discord.gg/context7)
- [Documentation](https://context7-agent.readthedocs.io)

---

## 🎉 Success Checklist

After deployment, verify:

- [ ] Container starts successfully
- [ ] Environment variables loaded
- [ ] Data volume persists
- [ ] Health checks pass
- [ ] Logs are accessible
- [ ] Monitoring is active
- [ ] Backups configured
- [ ] Security scan passed

---

**Ready to deploy?** 🚀

Choose your deployment path:
- [Local Development](docker/docker-compose.yml)
- [Production](docker/docker-compose.prod.yml)
- [Cloud Kubernetes](k8s/)
- [AWS ECS](aws-ecs/)
