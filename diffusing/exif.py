#@title EXIF and Saving helpers

import piexif
import imagehash
import piexif.helper
from datetime import datetime
from generation import GenerationConfig

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
