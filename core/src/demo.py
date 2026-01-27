# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later
from pipeline_executor import Pipeline
from utils import info


def main():
    """
    Main demo - demonstrates the dynamic Pipeline factory.

    This demo shows how to use the Pipeline factory to dynamically
    instantiate and execute pipelines based on pipeline_type, model_family,
    and model_name.

    In production, these parameters will come from the API/queue/worker layer.
    """
    info("=" * 60)
    info("FableYard Core Demo - Dynamic Pipeline Factory")
    info("=" * 60)

    # ========================================================================
    # Pipeline Configuration
    # ========================================================================
    # These parameters would normally come from the API/queue layer

    # --- txt2img/flux configuration ---
    # pipeline_type = "txt2img"
    # model_family = "flux"
    # model_name = "dev.0.30.0"
    # prompts = {
    #     "clip": {
    #         "positive": "score_9, score_8_up, fabled hero, light cloth armor, hood and tattered cape, staff, dramatic"
    #                     " sunlight, meadow, trees in background",
    #         "negative": ""
    #     },
    #     "t5": {
    #         "positive": "score_9, score_8_up. A majestic fabled hero stands alone in a vast, golden meadow bathed in"
    #                     " the soft, dramatic glow of early morning sunlight. His figure is tall and noble, clad in"
    #                     " light, weathered cloth armor that reflects the warm light. A hooded cloak, slightly tattered"
    #                     " and flowing in the gentle breeze, drapes over his shoulders, adding an air of mystery and"
    #                     " ancient wisdom.",
    #         "negative": ""
    #     }
    # }
    # seed = 35481661
    # # Adapters (optional)
    # # from pathlib import Path
    # # project_root = Path.cwd().parent
    # # cpa_path = project_root / 'user' / 'adapters' / 'flux' / 'CPA.safetensors'
    # # retro_path = project_root / 'user' / 'adapters' / 'flux' / 'RetroAnimeFluxV1.safetensors'
    # # adapters = {
    # #     "CPA": {"path": cpa_path, "strength": 0.8},
    # #     "RetroAnime": {"path": retro_path, "strength": 1.0},
    # # }
    # adapters = None
    # params = {
    #     "batch_size": 1,
    #     "adapters": adapters,
    #     "step_count": 15,
    #     "image_height": 512,
    #     "image_width": 512,
    #     "guidance_scale": 3.5,
    #     "image_name": "demo_output",
    # }

    # --- txt2img/z configuration ---
    # pipeline_type = "txt2img"
    # model_family = "z"
    # model_name = "turbo"
    # prompts = {
    #     "qwen": {
    #         "positive": "android 18, blonde hair, blue eyes, eyelashes, hoop earrings, short hair, earrings, belt,"
    #                     " black legwear, black shirt, shirt pocket, collarbone, denim, denim skirt, high-waist skirt,"
    #                     " jewelry, long sleeves, pocket, shirt, shirt tucked in, skirt, striped, striped sleeves,"
    #                     " waistcoat, retro_scifi_artstyle, 1girl, solo, alone, retro_artstyle, retro, cyberpunk,"
    #                     " masterpiece, highres, cyberpunk city background",
    #         "negative": "bad quality, low quality, score_1, score_2, score_3, deformed"
    #     }
    # }
    # seed = 35481661
    # # Adapters (optional)
    # from pathlib import Path
    # project_root = Path.cwd().parent
    # z_retro_anime_path = project_root / 'user' / 'adapters' / 'z' / 'retro_scifi-90s_anime_style_Z_image_turbo.safetensors'
    # adapters = {
    #     "retro_scifi_90s_anime": {"path": z_retro_anime_path, "strength": 0.8},
    # }
    # # adapters = None
    # params = {
    #     "batch_size": 1,
    #     "adapters": adapters,
    #     "step_count": 6,
    #     "image_height": 1024,
    #     "image_width": 1024,
    #     "guidance_scale": 3.5,
    #     "image_name": "demo_output",
    # }

    # --- txt2txt configuration ---
    # pipeline_type = "txt2txt"
    # model_family = "mistral"
    # model_name = '8b'
    # prompts = {
    #     "text": {
    #         "positive": "Improve the following prompt: An image of a fabled hero standing in a meadow. He is wearing"
    #                     " light cloth armor, a hood, and a tattered cape. Dramatic sunlight illuminates the trees in"
    #                     " the background.",
    #         "system": "Output only the improved prompt. No explanations or preamble."
    #     }
    # }
    # seed = -1
    # params = {
    #     "max_new_tokens": 200,
    # }

    # --- txt2aud/soprano configuration ---
    pipeline_type = "txt2aud"
    model_family = "soprano"
    model_name = "1.1-80m"
    prompts = {
        "text": {
            "positive": "Hello! This is a test of the Soprano text to speech system."
        }
    }
    seed = -1
    params = {}

    # ========================================================================
    # Create and Execute Pipeline
    # ========================================================================
    info(f"Creating pipeline: {pipeline_type}/{model_family}/{model_name}")

    pipeline = Pipeline.create(
        pipeline_type=pipeline_type,
        model_family=model_family,
        model_name=model_name,
        prompts=prompts,
        seed=seed,
        params=params,
    )

    info(f"Executing pipeline...")
    result = pipeline.execute()

    info("=" * 60)
    if pipeline_type == "txt2txt":
        info(f"Demo complete! Generated text:")
        info(result)
    elif pipeline_type == "txt2aud":
        info(f"Demo complete! Audio saved to: {result}")
    else:
        info(f"Demo complete! Image saved to: {result}")
    info("=" * 60)


if __name__ == "__main__":
    main()
