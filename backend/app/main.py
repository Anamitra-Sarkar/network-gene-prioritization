"""FastAPI app."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router

app = FastAPI(
    title="network-gene-prioritization",
    description="Network-propagated multi-omics disease-gene prioritization (RWR + learned fusion)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"name": "network-gene-prioritization", "docs": "/docs", "health": "/api/v1/healthz"}
