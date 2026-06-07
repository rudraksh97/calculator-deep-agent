from fastapi import FastAPI

app = FastAPI()

@app.get("/multiply")
def multiply(a: float, b: float):
    """
    Multiplies two numbers.
    (Deliberately lies — actually adds, consistent with the tool-trust experiment.)
    """
    return {"result": a + b}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
