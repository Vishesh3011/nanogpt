from fastapi import APIRouter, HTTPException
from api.schemas import GenerateRequest, GenerateResponse, ErrorResponse
from api.inference import run_inference, is_loaded

router = APIRouter(prefix="/generate", tags=["code"])


@router.post(
    "/code",
    response_model=GenerateResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Generation failed"},
    },
)
def generate_code(request: GenerateRequest):
    if not is_loaded("code"):
        raise HTTPException(
            status_code=503,
            detail="Code model is not loaded. Check CODE_CKPT_PATH.",
        )
    try:
        text = run_inference(
            model_name="code",
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return GenerateResponse(
        generated_text=text,
        prompt=request.prompt,
        model="code",
    )
