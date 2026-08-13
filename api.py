"""
用 FastAPI 把 streaming.py 暴露成 HTTP端点
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
import uuid
from streaming import stream_research, stream_resume

from fastapi.responses import FileResponse

app = FastAPI(title="Mini Research Agent")

@app.post("/research")
async def start_research(question: str = Query(...)):
    """发起研究，返回 SSE 流"""
    thread_id = str(uuid.uuid4())
    return StreamingResponse(
        stream_research(question, thread_id),
        media_type="text/event-stream",
        headers={"X-Thread-ID": thread_id},   # 前端从这个 header 拿到 thread_id
    )

@app.post("/research/{thread_id}/resume")
async def resume_research(thread_id: str, answer: str = Query(...)):
    """回复 clarify 追问，继续研究"""
    return StreamingResponse(
        stream_resume( thread_id, answer),
        media_type="text/event-stream",
    ) 

@app.get("/")
async def index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
