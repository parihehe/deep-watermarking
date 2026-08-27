"""Phase 7 — FastAPI application exposing the frozen DWT-SVD baseline to a browser.

Run locally:
    .venv/Scripts/python.exe -m uvicorn src.app.main:app --reload --port 8000

Then open http://127.0.0.1:8000/ .

This layer only marshals HTTP <-> ``src.app.service``; it contains no
watermarking logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.app import service

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="DWT-SVD Invisible Watermarking — Demo",
    description="Interactive front end for the frozen Phase 6 DWT-SVD baseline.",
    version="0.7.0",
)


@app.get("/api/health")
def health() -> dict:
    info: dict = {"status": "ok"}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001 - health must never fail on an optional import
        info["torch"] = None
        info["torch_error"] = repr(exc)
    return info


@app.get("/api/options")
def get_options() -> dict:
    return service.options()


@app.post("/api/watermark/embed")
async def watermark_embed(
    image: Annotated[UploadFile, File()],
    alpha: Annotated[float, Form()] = 0.010,
    bit_length: Annotated[int, Form()] = 64,
    wavelet: Annotated[str, Form()] = "haar",
    subband: Annotated[str, Form()] = "LL",
    extra_subbands: Annotated[str, Form()] = "",
    payload_source: Annotated[str, Form()] = "message",
    payload_text: Annotated[str | None, Form()] = None,
    payload_uuid: Annotated[str | None, Form()] = None,
    payload_bits: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    data = await image.read()
    extra = tuple(s for s in (x.strip() for x in extra_subbands.split(",")) if s)
    req = service.EmbedRequest(
        alpha=alpha,
        bit_length=bit_length,
        wavelet=wavelet,
        subband=subband,
        extra_subbands=extra,
        payload_source=payload_source,
        payload_text=payload_text,
        payload_uuid=payload_uuid,
        payload_bits=payload_bits,
    )
    try:
        result = service.run_embed(data, req)
    except service.ServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(result)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
