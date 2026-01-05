import enum
from uuid import UUID

from pydantic import BaseModel
from typing import Optional, Dict


class PipelineType(enum.Enum):
    txt2img = "Text to Image"
    txt2txt = "Text to Text"
    txt2vid = "Text to Video"
    txt2aud = "Text to Audio"
    img2img = "Image to Image"
    img2txt = "Image to Text"
    img2vid = "Image to Video"
    img2aud = "Image to Audio"
    vid2img = "Video to Image"
    vid2txt = "Video to Text"
    vid2vid = "Video to Video"
    vid2aud = "Video to Audio"
    aud2img = "Audio to Image"
    aud2txt = "Audio to Text"
    aud2vid = "Audio to Video"
    aud2aud = "Audio to Audio"


class MediaRequest(BaseModel):
    pipelineType: str
    model: str
    prompts: Dict[str, Dict[str, str]]  # Flexible prompt structure: {"encoder": {"positive": "...", "negative": "..."}}
    stepCount: int = 20
    imageWidth: int = 512
    imageHeight: int = 512
    lora: Optional[str] = None


class MediaResponse(BaseModel):
    runId: str
    timestamp: str
    queuePosition: int
    pipelineType: str
    model: str


class LorasResponse(BaseModel):
    loras: list[str]


class ModelsResponse(BaseModel):
    models: list[str]


class OutputImage(BaseModel):
    filename: str
    timestamp: float


class OutputsResponse(BaseModel):
    outputs: list[OutputImage]
