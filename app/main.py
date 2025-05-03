from langfuse.decorators import observe, langfuse_context
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    langfuse_context.flush()

app = FastAPI(lifespan=lifespan) 