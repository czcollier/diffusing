import abc
from dataclasses import dataclass
import torch
from IPython.display import display, Markdown
import time

from diffusers import (
    StableDiffusion3Pipeline,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    FluxPipeline,
    DiffusionPipeline
)

import piexif
import imagehash

from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd3, get_weighted_text_embeddings_flux1
from generation_model import (
    GenerationConfig,
    Subject,
    Adapter
)

@dataclass
class PicGenerator(abc.ABC):
  pipe: DiffusionPipeline
  rnd: torch._C.Generator = torch.Generator(device='cuda')
  seed: int = None

  def load_adapters(self, adapters: list[Adapter]):
    self.pipe.unload_lora_weights()

    for adp in adapters:
      self.pipe.load_lora_weights(
          adp.path,
          weight_name=adp.weights_file,
          adapter_name=adp.name)

    names = [a.name for a in adapters]
    weights = [a.scale for a in adapters]
    self.pipe.set_adapters(names, adapter_weights=weights)

  @abc.abstractmethod
  def generate(self):
    pass

  @abc.abstractmethod
  def build_pipe(self):
    pass


@dataclass
class FluxGenerator(PicGenerator):
  def build_pipe(self):
    return FluxPipeline.from_pretrained(
      model_id,
      torch_dtype=torch.bfloat16,
      safety_checker=None,
      use_safetensors=True,
      device="cuda"
    ).to("cuda")

  def generate(
      self,
      subject: Subject,
      config: GenerationConfig = GenerationConfig()):

    prompt = subject.build_prompt()

    (prompt_embeds, pooled_prompt_embeds) = get_weighted_text_embeddings_flux1(
      self.pipe,
      prompt = prompt
    )

    images = self.pipe(
      prompt_embeds=prompt_embeds,
      pooled_prompt_embeds=pooled_prompt_embeds,
      height=config.height(),
      width=config.width(),
      num_images_per_prompt=config.num_samples,
      num_inference_steps=config.steps,
      guidance_scale=config.guidance_scale,
      generator=self.rnd,
    ).images

    return images, prompt

class StableDiffusionGenerator(PicGenerator):
  def build_pipe(self):
    return StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16).to("cuda")

  def generate(
      self,
      subject: Subject,
      config: GenerationConfig = GenerationConfig()):

    prompt = subject.build_prompt()

    (prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds) = get_weighted_text_embeddings_sd3(
      self.pipe,
      neg_prompt = subject.neg_prompt,
      prompt = prompt
    )

    images = self.pipe(
      prompt_embeds=prompt_embeds,
      pooled_prompt_embeds=pooled_prompt_embeds,
      negative_prompt_embeds=neg_prompt_embeds,
      negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
      height=config.height(),
      width=config.width(),
      num_images_per_prompt=config.num_samples,
      num_inference_steps=config.steps,
      guidance_scale=config.guidance_scale,
      generator=self.rnd,
    ).images

    return images, prompt

class StableDiffusionMedGenerator(StableDiffusionGenerator):
  def build_pipe(self):
    return StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16).to("cuda")

