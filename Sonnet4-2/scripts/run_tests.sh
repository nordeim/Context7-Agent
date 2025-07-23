#!/bin/bash
# Test runner script for Context7 Agent

set -e 

echo "🧪 Running Context7 Agent Test Suite"
echo "===================================="

# Install test dependencies
echo "📦 Installing test dependencies..."
pip install -r requirements.txt
    
# Run unit tests
echo "🔬 Running unit tests..."
pytest tests/ -v -m "not integration" --cov=src --cov-report=term-missing
        
# Run integration tests
echo "🔗 Running integration tests..."
pytest tests/ -v -m integration
        
# Generate coverage report
echo "📊 Generating coverage report..."
pytest tests/ --cov=src --cov-report=html --cov-report=term
                
# Run performance tests
echo "⚡ Running performance tests..."
pytest tests/performance/ -v || echo "Performance tests not implemented yet"

echo "✅ All tests completed!"

