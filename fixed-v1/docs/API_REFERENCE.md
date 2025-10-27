# API Reference - Enhanced Hadith AI Fixed V1

Complete API documentation for the Enhanced Hadith AI REST API server.

## 🌐 Base URL

```
http://localhost:5000
```

## 📡 API Endpoints

### Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Service health check |
| POST | `/session/create` | Create new chat session |
| POST | `/chat` | Process chat query with session |
| POST | `/query` | Simple query without session |
| GET | `/session/{session_id}/stats` | Get session statistics |
| GET | `/greeting` | Get greeting message |

## 📋 Response Format

All API responses follow this standardized format:

```json
{
  "success": true|false,
  "timestamp": "2024-01-01T12:00:00.000000",
  "data": {...},
  "message": "Success message",
  "error": "Error message if applicable"
}
```

## 🔍 Detailed Endpoint Documentation

### 1. API Information

**GET** `/`

Get basic information about the API service.

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-01T12:00:00",
  "data": {
    "service": "Hadith AI API - Fixed V1",
    "version": "1.0.0",
    "description": "Enhanced Hadith retrieval and question answering API",
    "endpoints": { ... }
  },
  "message": "Hadith AI API is running",
  "error": ""
}
```

**Example:**
```bash
curl http://localhost:5000/
```

### 2. Health Check

**GET** `/health`

Check the health status of the service and all components.

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-01T12:00:00",
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-01T12:00:00",
    "components": {
      "retrieval_system": "healthy",
      "query_preprocessor": "healthy",
      "session_management": "healthy"
    },
    "statistics": {
      "active_sessions": 5,
      "total_sessions_created": 42,
      "analytics_enabled": true
    },
    "configuration": {
      "max_results_display": 5,
      "session_timeout_minutes": 30,
      "min_confidence_threshold": 0.1
    }
  },
  "message": "Service is healthy",
  "error": ""
}
```

**Status Codes:**
- `200`: Service is healthy
- `503`: Service is degraded or unhealthy

**Example:**
```bash
curl http://localhost:5000/health
```

### 3. Create Session

**POST** `/session/create`

Create a new chat session for conversation tracking.

**Request Body:** None required

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-01T12:00:00",
  "data": {
    "session_id": "session_1704110400000",
    "created_at": "2024-01-01T12:00:00"
  },
  "message": "Session created successfully",
  "error": ""
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/session/create
```

### 4. Chat Query (with Session)

**POST** `/chat`

Process a chat query with session management and context tracking.

**Request Body:**
```json
{
  "query": "apa hukum shalat jumat bagi wanita?",
  "session_id": "session_1704110400000",
  "max_results": 5
}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | User query in Indonesian |
| `session_id` | string | No | - | Session ID for context tracking |
| `max_results` | integer | No | 5 | Maximum results to return (1-20) |

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-01T12:00:00",
  "data": {
    "success": true,
    "message": "Berikut hadits yang sesuai dengan pertanyaan Anda:\n\nDitemukan 3 hadits:\n\n[1] Dari Jabir bin Abdullah ra, Rasulullah saw bersabda: \"Shalat Jumat adalah wajib bagi setiap muslim yang telah baligh...\"\n    💡 Kata kunci: shalat jumat, wanita\n    ✅ Hasil dengan kepercayaan tinggi",
    "results": [
      {
        "document_id": "bukhari_123",
        "document": {
          "id": "bukhari_123",
          "terjemah": "Dari Jabir bin Abdullah ra, Rasulullah saw bersabda: \"Shalat Jumat adalah wajib bagi setiap muslim yang telah baligh...\"",
          "source": "Shahih Bukhari",
          "chapter": "Kitab Shalat Jumat"
        },
        "score": 0.85,
        "semantic_score": 0.82,
        "keyword_score": 0.90,
        "literal_overlap_score": 0.75,
        "matched_keywords": ["shalat jumat", "wanita", "hukum"],
        "rank": 1
      }
    ],
    "query_analysis": {
      "original_query": "apa hukum shalat jumat bagi wanita?",
      "processed_query": "hukum shalat jumat wanita",
      "key_terms": ["hukum", "shalat", "jumat", "wanita"],
      "islamic_terms": ["shalat", "jumat"],
      "has_question": true,
      "has_action_intent": false,
      "islamic_context_strength": 0.75,
      "query_length": 6,
      "processed_length": 4
    },
    "response_time_ms": 234.5,
    "session_id": "session_1704110400000",
    "metadata": {
      "total_candidates": 45,
      "filtered_candidates": 12,
      "query_length": 6,
      "avg_result_score": 0.78
    }
  },
  "message": "Query processed successfully",
  "error": ""
}
```

**Error Responses:**
```json
{
  "success": false,
  "timestamp": "2024-01-01T12:00:00",
  "data": null,
  "message": "",
  "error": "Missing required fields: query"
}
```

**Status Codes:**
- `200`: Query processed successfully
- `400`: Invalid request (missing fields, invalid parameters)
- `500`: Internal server error

**Example:**
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "bagaimana cara berwudhu yang benar?",
    "session_id": "session_1704110400000",
    "max_results": 3
  }'
```

### 5. Simple Query (without Session)

**POST** `/query`

Process a simple query without session management.

**Request Body:**
```json
{
  "query": "hukum zakat fitrah",
  "max_results": 5
}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | User query in Indonesian |
| `max_results` | integer | No | 5 | Maximum results to return (1-20) |

**Response:**
Same format as `/chat` but without `session_id` in response.

**Example:**
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "puasa ramadan bagi muslimah",
    "max_results": 3
  }'
```

### 6. Session Statistics

**GET** `/session/{session_id}/stats`

Get detailed statistics for a specific session.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | Yes | Session ID in URL path |

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-01T12:00:00",
  "data": {
    "session_id": "session_1704110400000",
    "created_at": "2024-01-01T11:00:00",
    "query_count": 5,
    "total_results_returned": 15,
    "avg_results_per_query": 3.0,
    "context_history_size": 5,
    "last_activity": "2024-01-01T12:00:00"
  },
  "message": "Session statistics retrieved successfully",
  "error": ""
}
```

**Error Response:**
```json
{
  "success": false,
  "timestamp": "2024-01-01T12:00:00",
  "data": null,
  "message": "",
  "error": "Session not found or expired"
}
```

**Status Codes:**
- `200`: Statistics retrieved successfully
- `404`: Session not found or expired

**Example:**
```bash
curl http://localhost:5000/session/session_1704110400000/stats
```

### 7. Greeting Message

**GET** `/greeting`

Get the greeting message for new conversations.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `session_id` | string | No | Optional session ID |

**Response:**
```json
{
  "success": true,
  "timestamp": "2024-01-01T12:00:00",
  "data": {
    "success": true,
    "message": "Assalamu'alaikum! Saya siap membantu Anda mencari hadits. Silakan ajukan pertanyaan tentang Islam.",
    "results": [],
    "query_analysis": null,
    "response_time_ms": 0,
    "session_id": "session_1704110400000",
    "metadata": {
      "event": "greeting"
    }
  },
  "message": "Greeting retrieved successfully",
  "error": ""
}
```

**Example:**
```bash
curl "http://localhost:5000/greeting?session_id=session_1704110400000"
```

## 🔐 Authentication

Currently, the API does not require authentication. For production deployment, consider adding:

- API key authentication
- JWT token-based authentication
- Rate limiting per user/IP
- CORS configuration for specific domains

## 📊 Response Data Models

### Chat Response Object

```typescript
interface ChatResponse {
  success: boolean;
  message: string;
  results: RetrievalResult[];
  query_analysis?: QueryAnalysis;
  response_time_ms: number;
  session_id?: string;
  metadata: Record<string, any>;
}
```

### Retrieval Result Object

```typescript
interface RetrievalResult {
  document_id: string;
  document: {
    id: string;
    terjemah: string;
    source?: string;
    chapter?: string;
    [key: string]: any;
  };
  score: number;
  semantic_score: number;
  keyword_score: number;
  literal_overlap_score: number;
  matched_keywords: string[];
  rank: number;
}
```

### Query Analysis Object

```typescript
interface QueryAnalysis {
  original_query: string;
  processed_query: string;
  key_terms: string[];
  islamic_terms: string[];
  has_question: boolean;
  has_action_intent: boolean;
  islamic_context_strength: number;
  query_length: number;
  processed_length: number;
}
```

## 🚀 Usage Examples

### JavaScript/Node.js

```javascript
const axios = require('axios');

const API_BASE = 'http://localhost:5000';

// Create session and process query
async function chatWithHadithAI() {
  try {
    // Create session
    const sessionResponse = await axios.post(`${API_BASE}/session/create`);
    const sessionId = sessionResponse.data.data.session_id;
    
    // Get greeting
    const greetingResponse = await axios.get(`${API_BASE}/greeting?session_id=${sessionId}`);
    console.log('Greeting:', greetingResponse.data.data.message);
    
    // Process query
    const queryResponse = await axios.post(`${API_BASE}/chat`, {
      query: 'apa hukum minuman keras dalam islam?',
      session_id: sessionId,
      max_results: 3
    });
    
    console.log('Response:', queryResponse.data.data.message);
    console.log('Results:', queryResponse.data.data.results.length);
    
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
}

chatWithHadithAI();
```

### Python

```python
import requests

API_BASE = 'http://localhost:5000'

def chat_with_hadith_ai():
    try:
        # Create session
        session_response = requests.post(f'{API_BASE}/session/create')
        session_id = session_response.json()['data']['session_id']
        
        # Process query
        query_response = requests.post(f'{API_BASE}/chat', json={
            'query': 'bagaimana cara shalat yang benar?',
            'session_id': session_id,
            'max_results': 5
        })
        
        data = query_response.json()['data']
        print(f"Response: {data['message']}")
        print(f"Found {len(data['results'])} results")
        print(f"Response time: {data['response_time_ms']:.1f}ms")
        
    except requests.RequestException as e:
        print(f"Error: {e}")

chat_with_hadith_ai()
```

### cURL Examples

```bash
# Health check
curl http://localhost:5000/health

# Simple query
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "hukum riba", "max_results": 3}'

# Session-based conversation
SESSION_ID=$(curl -X POST http://localhost:5000/session/create | jq -r '.data.session_id')
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"zakat fitrah\", \"session_id\": \"$SESSION_ID\"}"
```

## ⚠️ Error Handling

### Common Error Codes

| Status Code | Description | Common Causes |
|-------------|-------------|---------------|
| 400 | Bad Request | Missing required fields, invalid parameters |
| 404 | Not Found | Invalid endpoint, expired session |
| 405 | Method Not Allowed | Wrong HTTP method |
| 500 | Internal Server Error | System error, configuration issues |
| 503 | Service Unavailable | System unhealthy, maintenance mode |

### Error Response Format

```json
{
  "success": false,
  "timestamp": "2024-01-01T12:00:00",
  "data": null,
  "message": "",
  "error": "Detailed error description"
}
```

### Best Practices

1. **Always check the `success` field** before processing response data
2. **Handle network timeouts** - some queries may take several seconds
3. **Implement retry logic** for transient errors (5xx codes)
4. **Validate inputs** before sending to API
5. **Use sessions** for conversational interfaces
6. **Monitor health endpoint** for system status

## 🔧 Configuration

### API Server Configuration

The API server can be configured via environment variables or command line arguments:

```bash
# Environment variables
export API_HOST=0.0.0.0
export API_PORT=5000
export DEBUG_MODE=False
export MAX_RESULTS_DEFAULT=5

# Command line
python service/api_server.py \
  --host 0.0.0.0 \
  --port 5000 \
  --max-results 5
```

### Service Configuration

For advanced configuration, modify the `ServiceConfig` in your code:

```python
from service import ServiceConfig, RetrievalConfig

config = ServiceConfig(
    max_results_display=10,
    enable_sessions=True,
    session_timeout_minutes=60,
    min_confidence_threshold=0.15,
    retrieval_config=RetrievalConfig(
        top_k=50,
        semantic_weight=0.7,
        keyword_weight=0.2,
        literal_overlap_weight=0.1
    )
)
```

This API reference provides everything needed to integrate with the Enhanced Hadith AI Fixed V1 system.
