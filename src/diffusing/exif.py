import sys
import piexif
import imagehash
import piexif.helper
from datetime import datetime

import json
from PIL import Image
from io import BytesIO
from diffusing.generation_model import GenerationConfig, Subject


def save_with_exif(img, gen_config, subject, model_id, path):
  prompt = subject.build_prompt()
  comment_str = f"""model ID: {model_id}
  num_inference_steps: {gen_config.steps}
  guidance_scale: {gen_config.guidance_scale}
  height: {gen_config.height()}
  width: {gen_config.width()}
  prompt: {prompt}""".replace("\n", "; ")

  comment = piexif.helper.UserComment.dump(comment_str)
  date_now = datetime.now().strftime("%Y:%m:%d %H:%M:%S")

  exif_exif_dict = {
      piexif.ExifIFD.UserComment: comment,
      piexif.ExifIFD.DateTimeOriginal: date_now,
      piexif.ExifIFD.DateTimeDigitized: date_now,
  }
  exif_dict = {
      '0th': { piexif.ImageIFD.DateTime: date_now },
      "Exif": {
        piexif.ExifIFD.UserComment: comment,
        piexif.ExifIFD.DateTimeOriginal: date_now,
        piexif.ExifIFD.DateTimeDigitized: date_now,
      },
      "1st": {},
      "thumbnail": None,
      "GPS": {}}

  exif_dat = piexif.dump(exif_dict)

  image_hash = str(imagehash.average_hash(img))
  full_path = f"{path}/{image_hash}.jpg"
  print(f"saving as: {full_path}")
  img.save(full_path,  exif=exif_dat)


def add_exif(filename, gen_config, subject, model_id, path):
  img = Image.open(filename)
  return save_with_exif(img, gen_config, subject, model_id, path)


def extract_exif(img_bytes_or_filename):
  #img = Image.open(BytesIO(img_bytes))
  exif_dict = piexif.load(img_bytes_or_filename)
  return exif_dict["Exif"][piexif.ExifIFD.UserComment]

if __name__ == "__main__":
  mode = sys.argv[1]
  filename = sys.argv[2]
  model_id = "some model"
  if mode == "show":
    print(extract_exif(filename).decode("utf=8"))
  elif mode == "write":
    with open("gen_config.json") as gc_file:
      gen_config = json.load(gc_file, object_hook=lambda d: GenerationConfig(**d))
    with open("prompt.json") as prompt_file:
      prompt = json.load(prompt_file)
    with open("subject.json") as subj_file:
      subject = json.load(subj_file, object_hook=lambda d: Subject(**d))
    prompt_str = prompt["text"]
    add_exif(filename, gen_config, subject, model_id, "./")
