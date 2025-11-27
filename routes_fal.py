import os
import asyncio
import logging
from typing import List
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv
from llm_responses import generate_flux_prompt
import fal_client


# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Env + App Setup ---
load_dotenv()

if not os.getenv("FAL_KEY"):
    print("⚠️ WARNING: Missing FAL_KEY env var.")

os.environ["FAL_KEY"] = os.getenv("FAL_KEY", "")

router = APIRouter(prefix="/fal", tags=["fal"])


# --- Models ---
class BannerRequest(BaseModel):
    """
    Request body for banner generation.
    'prompt' is used AS-IS for image/banner generation.
    """
    raw_data: str
    variant_count: int = 1  # how many banner variations to generate



class BannerResult(BaseModel):
    index: int
    prompt: str
    image_url: str






# -----------------------------------------------------------
# IMAGE GENERATION
# -----------------------------------------------------------
async def generate_single_banner_image(prompt: str, index: int) -> BannerResult:
    try:
        logger.info(f"🖼️ Generating banner {index + 1} with prompt: {prompt}")

        handler = await fal_client.submit_async(
            "rundiffusion-fal/juggernaut-flux/pro",
            arguments={"prompt": prompt},
        )

        result = await handler.get()

        images = result.get("images") or []
        if not images:
            raise ValueError("No images returned from FAL")

        image_url = images[0].get("url")
        if not image_url:
            raise ValueError("No image URL in FAL response")

        logger.info(f"✅ Banner {index + 1} ready: {image_url}")

        return BannerResult(
            index=index + 1,
            prompt=prompt,
            image_url=image_url,
        )

    except Exception as e:
        logger.error(f"❌ [Banner {index + 1}] Failed: {e}", exc_info=True)
        return BannerResult(index=index + 1, prompt=prompt, image_url="FAILED")


# -----------------------------------------------------------
# MAIN ENDPOINT
# -----------------------------------------------------------
@router.post("/generate-banner", response_model=dict)
async def generate_banner(request: BannerRequest):
    print("reaching here ")
    raw_data = request.raw_data

    logger.info(f"\n🚀 STARTING BANNER JOB: Raw Data = {raw_data}")

    # 🔥 Step 1: Rewrite user’s raw JSON → 2-line Flux prompt
    flux_prompt = await generate_flux_prompt(raw_data)

    # Multiple variants = identical prompts
    prompts: List[str] = [flux_prompt for _ in range(request.variant_count)]

    # Fire parallel tasks
    tasks = [
        generate_single_banner_image(prompt, i)
        for i, prompt in enumerate(prompts)
    ]

    results = await asyncio.gather(*tasks)

    logger.info("🎉 All banners completed.")

    return {
        "status": "success",
        "final_prompt_used": flux_prompt,
        "total_images": len(results),
        "results": results,
    }