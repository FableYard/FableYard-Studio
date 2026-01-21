# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

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
    pipeline_type = "txt2img"
    # model_family = 'flux'
    # model_name = 'dev.0.30.0'
    model_family = "z"
    model_name = "turbo"

    # Runtime parameters
    batch_size = 1
    prompts = {
        # "clip": {
        #     "positive": "score_9, score_8_up, fabled hero, light cloth armor, hood and tattered cape, staff, dramatic"
        #                 " sunlight, meadow, trees in background",
        #     "negative": ""
        # },
        # "t5": {
        #     "positive": "score_9, score_8_up. An image of a fabled hero standing in a meadow. He is wearing light cloth"
        #                 " armor, a hood, and a tattered cape. Dramatic sunlight illuminates the trees in the "
        #                 "background.",
        #     "negative": ""
        # }
        "qwen": {
            "positive": "android 18, blonde hair, blue eyes, eyelashes, hoop earrings, short hair, earrings, belt,"
                        " black legwear, black shirt, shirt pocket, collarbone, denim, denim skirt, high-waist skirt,"
                        " jewelry, long sleeves, pocket, shirt, shirt tucked in, skirt, striped, striped sleeves,"
                        " waistcoat, retro_scifi_artstyle, 1girl, solo, alone, retro_artstyle, retro, cyberpunk,"
                        " masterpiece, highres, cyberpunk city background",
            "negative": "bad quality, low quality, score_1, score_2, score_3, deformed"
        }
    }
    step_count = 15
    height = 512
    width = 512
    seed = 35481661
    guidance_scale = 3.5
    image_name = "demo_output"

    # ========================================================================
    # Adapters configuration
    # ========================================================================
    project_root = Path.cwd().parent
    # cpa_path = project_root / 'user' / 'adapters' / 'flux' / 'CPA.safetensors'
    # retro_path = project_root / 'user' / 'adapters' / 'flux' / 'RetroAnimeFluxV1.safetensors'
    # glass_path = project_root / 'user' / 'adapters' / 'flux' / 'glass-sculptures-flux.safetensors'
    z_retro_anime_path = project_root / 'user' / 'adapters' / 'z' / 'retro_scifi-90s_anime_style_Z_image_turbo.safetensors'

    # info(f"cpa_path: {cpa_path}")
    adapters = {
        # Example structure: "adapter_name": {"path": "path/to/adapter.safetensors", "strength": 1.0}
        # "CPA": {"path": cpa_path, "strength": 0.8},
        # "RetroAnime": {"path": retro_path, "strength": 1.0},
        # "glass-sculptures-flux": {"path": glass_path, "strength": 0.8}
        "retro_scifi_90s_anime": {"path": z_retro_anime_path, "strength": 0.8}
    }
    # adapters = None

    # ========================================================================
    # Create and Execute Pipeline
    # ========================================================================
    info(f"Creating pipeline: {pipeline_type}/{model_family}/{model_name}")

    pipeline = Pipeline.create(
        pipeline_type=pipeline_type,
        model_family=model_family,
        model_name=model_name,
        batch_size=batch_size,
        prompts=prompts,
        adapters=adapters,
        step_count=step_count,
        image_height=height,
        image_width=width,
        seed=seed,
        guidance_scale=guidance_scale,
        image_name=image_name
    )

    info(f"Executing pipeline...")
    result_path = pipeline.execute()

    info("=" * 60)
    info(f"Demo complete! Image saved to: {result_path}")
    info("=" * 60)


if __name__ == "__main__":
    main()
