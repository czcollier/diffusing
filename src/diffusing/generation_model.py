from dataclasses import dataclass

@dataclass
class Subject:
  person: str
  pose: str = ""
  scene: str = ""
  clothing: str = ""
  photo_quality: str = ""
  neg_prompt: str = None

  def build_prompt(self):
    descr = ", ".join([self.person, self.pose, self.scene, self.clothing])
    prompt = ", ".join([descr, self.photo_quality])
    return prompt

@dataclass
class Adapter:
  path: str
  weights_file: str
  name: str
  scale: float = 1.0

@dataclass
class GenerationConfig:
  guidance_scale: float = 7.5
  steps: int = 30
  small_dim: int = 896
  large_dim: int = 1152
  landscape: bool = False
  num_samples: int = 1

  def height(self):
    return self.small_dim if self.landscape else self.large_dim

  def width(self):
    return self.large_dim if self.landscape else self.small_dim
