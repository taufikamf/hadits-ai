# Hadits AI Backend API Documentation

**Base URL:** `http://localhost:8000`  
**Version:** 1.0.0  
**Description:** FastAPI backend for Hadits AI chatbot with RAG capabilities and chat session management

## Endpoints Overview

### 1. Health Check
```
GET /health
```
**Response (200):**
```json
{
  "status": "healthy",
  "message": "Hadis AI Backend is running",
  "retrieval_status": "ok", 
  "llm_status": "available"
}
```

### 2. Get All Sessions
```
GET /sessions
```
**Response (200):**
```json
{
  "sessions": [
    {
      "session_id": "5769bd75-34be-433e-832f-393075d255e7",
      "title": "Chat 2",
      "created_at": "2025-07-16T04:35:49.285785",
      "updated_at": "2025-07-16T04:37:21.326267", 
      "message_count": 4
    }
  ],
  "total": 1
}
```

### 3. Create New Session
```
POST /sessions
Content-Type: application/json

{
  "title": "Test Chat Session"  // optional, defaults to "New Chat"
}
```
**Response (200):**
```json
{
  "session_id": "af3b3597-d19a-449e-a2a4-52b1c3600622",
  "title": "Test Chat Session",
  "created_at": "2025-07-16T05:04:49.462409"
}
```

### 4. Get Specific Session
```
GET /sessions/{session_id}
```
**Response (200):**
```json
{
  "session_id": "af3b3597-d19a-449e-a2a4-52b1c3600622",
  "title": "Test Chat Session",
  "created_at": "2025-07-16T05:04:49.462409",
  "updated_at": "2025-07-16T05:04:49.462409",
  "messages": [
    {
      "message_id": "msg-123",
      "role": "user",
      "content": "apa itu riba?",
      "timestamp": "2025-07-16T05:05:00.000000"
    },
    {
      "message_id": "msg-124", 
      "role": "assistant",
      "content": "Riba adalah pengambilan tambahan...",
      "timestamp": "2025-07-16T05:05:15.000000"
    }
  ]
}
```
**Response (404):**
```json
{
  "detail": "Session not found"
}
```

### 5. Update Session Title
```
PUT /sessions/{session_id}
Content-Type: application/json

{
  "title": "Updated Chat Session"
}
```
**Response (200):**
```json
{
  "message": "Session updated successfully"
}
```

### 6. Delete Session
```
DELETE /sessions/{session_id}
```
**Response (200):**
```json
{
  "message": "Session deleted successfully"
}
```

### 7. Ask Question in Session (SSE Streaming)
```
POST /sessions/{session_id}/ask
Content-Type: application/json

{
  "question": "apa itu riba?"
}
```
**Response (200):**
```
Content-Type: text/event-stream

data: {"type": "status", "message": "Memproses pertanyaan..."}

data: {"type": "status", "message": "Mencari hadits terkait..."}

data: {"type": "status", "message": "Menggenerate respons..."}

data: {"type": "content", "content": "Berdasarkan hadits-hadits yang ditemukan:\n\n"}

data: {"type": "content", "content": "## Pengertian Riba\n\n"}

data: {"type": "content", "content": "Riba adalah pengambilan tambahan..."}

data: {"type": "done"}
```

### 8. Legacy Ask Endpoint
```
POST /ask
Content-Type: application/json

{
  "question": "apa hukum riba?"
}
```
**Response:** Same SSE format as session ask endpoint

## Data Models

### ChatMessage
```json
{
  "message_id": "string (UUID)",
  "role": "user | assistant", 
  "content": "string",
  "timestamp": "string (ISO 8601)"
}
```

### ChatSession
```json
{
  "session_id": "string (UUID)",
  "title": "string",
  "created_at": "string (ISO 8601)",
  "updated_at": "string (ISO 8601)",
  "messages": "array of ChatMessage"
}
```

### SessionSummary
```json
{
  "session_id": "string (UUID)",
  "title": "string",
  "created_at": "string (ISO 8601)", 
  "updated_at": "string (ISO 8601)",
  "message_count": "number"
}
```

## SSE Streaming Events

### Event Types
1. **status** - Processing status updates
   ```json
   {"type": "status", "message": "Memproses pertanyaan..."}
   ```

2. **content** - Streaming response content
   ```json
   {"type": "content", "content": "Response text chunk..."}
   ```

3. **done** - End of streaming
   ```json
   {"type": "done"}
   ```

### Implementation Notes
- Use EventSource API or fetch with `text/event-stream`
- Parse each SSE event as JSON
- Handle 'status' events for UI loading indicators
- Accumulate 'content' events for response text
- Close connection on 'done' event

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request format"
}
```

### 404 Not Found
```json
{
  "detail": "Session not found"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "question"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

### 503 Service Unavailable
```json
{
  "detail": "Retrieval system not available"
}
```

## Technical Details

### CORS Configuration
- **Allowed Origins:** `*` (all origins in development)
- **Allowed Methods:** GET, POST, PUT, DELETE
- **Allowed Headers:** `*`

### Data Storage
- **Session Storage:** JSON files in `data/sessions/`
- **Session Persistence:** Across server restarts
- **UUID Format:** UUID4 for session and message IDs
- **DateTime Format:** ISO 8601 with timezone

### Limitations
- No authentication required
- No rate limiting implemented
- No explicit message length limits
- No timeout on SSE connections

## Frontend Implementation Recommendations

1. **SSE Client:** Use EventSource or fetch API with proper error handling
2. **State Management:** Consider React Query, SWR, or similar for session management
3. **UI Patterns:** Implement typing indicators during SSE status events
4. **Error Boundaries:** Wrap SSE components in error boundaries for connection failures
5. **Offline Support:** Cache sessions locally for offline viewing
6. **Accessibility:** Ensure SSE events are accessible to screen readers

## Quick Test Commands

```bash
# Health check
curl -X GET http://localhost:8000/health

# Get all sessions
curl -X GET http://localhost:8000/sessions

# Create session
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Chat"}'

# Get session
curl -X GET http://localhost:8000/sessions/{session_id}

# Update session
curl -X PUT http://localhost:8000/sessions/{session_id} \
  -H "Content-Type: application/json" \
  -d '{"title": "New Title"}'

# Delete session
curl -X DELETE http://localhost:8000/sessions/{session_id}

# Ask question with SSE
curl -X POST http://localhost:8000/sessions/{session_id}/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "apa itu riba?"}' \
  -N
``` 