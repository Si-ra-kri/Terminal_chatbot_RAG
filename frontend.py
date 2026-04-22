from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from main import rag_query

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


class QueryRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/query")
async def query_endpoint(body: QueryRequest):
    if not body.query.strip():
        return JSONResponse(status_code=400, content={"error": "empty query"})
    
    answer = rag_query(body.query)
    
    if answer is None:
        return JSONResponse(status_code=500, content={"error": "something went wrong"})
    
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("frontend:app", host="0.0.0.0", port=8000, reload=True)