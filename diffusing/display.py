#@title Image Display Helpers
import io
import math
from IPython.display import display
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from ipywidgets import widgets, HBox, Layout
from PIL import Image


def display_images(images, max_size=500):
  def grid_dims(i):
    x = math.floor(math.sqrt(i))
    return (x, math.ceil(i / x))

  def to_byte_array(img):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


  cols, rows = grid_dims(len(images))

  img_widgets = []
  for img in images:
    ratio = img.width / img.height
    max_width, max_height = (max_size, max_size * ratio) if img.width > img.height else (max_size, max_size / ratio)
    max_width_px = f"{max_width}px"
    max_height_px = f"{max_height}px"
    img_widgets.append(widgets.Image(value=to_byte_array(img), layout=Layout(max_height=max_height_px, max_width=max_width_px)))

  #img_widgets = [widgets.Image(value=to_byte_array(img)) for img in images]
  grid = np.reshape(img_widgets, (rows, cols))

  for row in grid:
    img_row = HBox(row.tolist(), layout=Layout(width='90%'))
    display(img_row)


#@title Create and seed random generator
#from datetime import datetime
#import torch
#g_cuda = torch.Generator(device='cuda')
#ms = g_cuda.manual_seed(datetime.now().microsecond)
