from fastapi import FastAPI
from config import engine
import models
import router

# generate model to table postgresql
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.get('/')
async def Home():
    return "Welcome Home"


app.include_router(router.router)