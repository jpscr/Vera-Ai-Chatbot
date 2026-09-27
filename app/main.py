from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .routes import context, meta, reply, tick

app = FastAPI(title="Vera merchant assistant", version="0.2.0")
app.include_router(meta.router)
app.include_router(context.router)
app.include_router(tick.router)
app.include_router(reply.router)


@app.exception_handler(RequestValidationError)
async def malformed(request: Request, exc: RequestValidationError):
    if request.url.path == "/v1/context":
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "malformed",
                                                      "details": str(exc.errors())[:500]})
    return JSONResponse(status_code=400, content={"error": "malformed", "details": str(exc.errors())[:500]})
