from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from .simulation import Simulation

app = FastAPI(title="Map Maker Simulation")

# mount static directory
app.mount("/static", StaticFiles(directory="./static"), name="static")

sim = Simulation(width=800.0, height=600.0, seed=42)


@app.get("/")
def read_root():
    return FileResponse("./static/index.html")


@app.get("/state")
def read_state():
    return JSONResponse(content=sim.get_state().model_dump())


@app.post("/tick")
def advance(ticks: int = 1):
    if ticks < 1 or ticks > 1000:
        raise HTTPException(status_code=400, detail="ticks must be between 1 and 1000")
    sim.tick_once(ticks)
    return JSONResponse(content={"status": "ok", "tick": sim.tick})


@app.get("/entity/{entity_id}")
def get_entity(entity_id: int):
    for e in sim.entities:
        if e.id == entity_id:
            return JSONResponse(content=e.model_dump())
    raise HTTPException(status_code=404, detail="entity not found")
