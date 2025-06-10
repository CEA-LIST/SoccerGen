import torch
import numpy as np
import supervision as sv
import einops
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import cv2
import json
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Optional

## Utils

def resize_for_letterbox(image, target_size, background_color=(0, 0, 0)):
  """
  Resizes an image for letterbox mode, maintaining aspect ratio with padding.

  Args:
      image_path: Path to the image file.
      target_size: Tuple representing the target width and height for the letterbox.
      background_color: Optional background color for padding (default black).

  Returns:
      A resized PIL Image object with letterbox padding.
  """
  width, height = image.size
  target_width, target_height = target_size

  # Calculate new image dimensions to maintain aspect ratio
  scale = min(target_width / width, target_height / height)
  new_width = int(width * scale)
  new_height = int(height * scale)

  # Create a background image with target size and chosen background color
  background = Image.new('RGB', target_size, background_color)

  # Paste the resized image onto the background, centered
  left = (target_width - new_width) // 2
  top = (target_height - new_height) // 2
  background.paste(image.resize((new_width, new_height)), (left, top))

  return background


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def annotate_bboxes(
    image: np.ndarray,
    detections: sv.Detections,
    labels: Optional[List[str]] = None
) -> np.ndarray:
    """
    Annotates an image with bounding boxes and labels based on provided detections.

    Parameters:
        image (np.ndarray): The image to be annotated. It should be in a format compatible with sv.BoundingBoxAnnotator
            and sv.LabelAnnotator, typically a NumPy array.
        detections (sv.Detections): A collection of detections, each typically containing information like
            bounding box coordinates, class IDs, etc., to be used for annotation.
        labels (Optional[List[str]]): A list of strings representing the labels for each detection. If not
            provided, labels are automatically generated as sequential numbers.

    Returns:
        np.ndarray: An annotated version of the input image, with bounding boxes and labels drawn over it.

    """
    bounding_box_annotator = sv.BoundingBoxAnnotator(
        color=sv.Color.BLACK,
        color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(
        color=sv.Color.BLACK,
        text_color=sv.Color.WHITE,
        color_lookup=sv.ColorLookup.CLASS,
        text_scale=0.7)

    if labels is None:
        labels = [str(i) for i in range(len(detections))]

    annotated_image = image.copy()
    annotated_image = bounding_box_annotator.annotate(
        annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(
        annotated_image, detections=detections, labels=labels)

    return annotated_image

##

parser = ArgumentParser()
parser.add_argument('--config', type=str, help='Configuration file path.')
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--dataset', type=str, default='football')
parser.add_argument('--input_json', type=str, help='JSON file containing inference inputs.')
parser.add_argument('--control_mode', type=str, choices=['calib', 'posxcol', 'calibxposxcol'], default='calibxposxcol')
parser.add_argument('--control_index', type=int, default=None, help='Index of the control image in the validation set of the Football dataset.')
parser.add_argument('--resolution', type=int, default=512, help='Resolution of the image.')
parser.add_argument('--batch_size', type=int, default=4, help='The number of images to sample.')
parser.add_argument('--ddim_steps', type=int, default=50)
parser.add_argument('--ddim_eta', type=float, default=0.)
parser.add_argument('--cfg_scale', type=float, default=9.0)
parser.add_argument('--cfg_rw', action="store_true", help="Whether to add classifier-free guidance reweighting.")
parser.add_argument('--negative_prompts', type=str, default="lowres, cropped, worst quality, low quality, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured")
parser.add_argument('--save_path', type=str, default=None, help="Path to the folder where to save the images.")
args = parser.parse_args()

dest = Path("." if args.save_path is None else args.save_path)
if not dest.exists():
    dest.mkdir(parents=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = OmegaConf.load(args.config)
model = instantiate_from_config(config.model)

if args.cfg_rw:
    model.control_scales = [args.cfg_scale * (0.825 ** float(12 - i)) for i in range(13)]

if args.ckpt_path is not None:
    model.init_from_ckpt(args.ckpt_path)

model = model.to(device)


inputs = json.load(Path(args.input_json).open())
control_keys = model.control_keys

cond = dict()
prompt = "A TV broacast image of a professional soccer game. Players of the two teams are scattered around the field on a green grass, some running, some kicking the ball. White lines mark the boundaries of the field. Players of the same team have the same jersey and the two teams' jerseys are different."
text_embedding = model.get_learned_conditioning([prompt]*args.batch_size)
cond["c_crossattn"] = [text_embedding]
valid_keys = [k for k in inputs.keys() if k in control_keys]
assert len(valid_keys) != 0, f"Error: no valid control key found in the input. Valid keys are: {control_keys}"
shape = (args.batch_size, 3, args.resolution, args.resolution)
for key in control_keys:
    if key in inputs:
        control_image = np.array(Image.open(inputs[key]).convert("RGB"))
        control_image = cv2.resize(control_image, (args.resolution, args.resolution)).astype(np.float32) / 127.5 - 1.0
        control = torch.FloatTensor(control_image).unsqueeze(0).expand((args.batch_size, -1, -1, -1)).to(device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format)
        cond[key] = [control]
    else:
        cond[key] = [torch.zeros(shape, device=device)]


use_ddim = args.ddim_steps is not None
sampling_kwargs = dict(cond=cond, batch_size=args.batch_size, ddim=use_ddim, ddim_steps=args.ddim_steps, eta=args.ddim_eta)
if args.cfg_scale > 1.0:
    negative_prompt = args.negative_prompts
    uc_cross = model.get_learned_conditioning([negative_prompt]*args.batch_size)
    uc_full = {**cond}
    uc_full["c_crossattn"] = [uc_cross]
    sampling_kwargs['unconditional_guidance_scale'] = args.cfg_scale
    sampling_kwargs['unconditional_conditioning'] = uc_full

samples, _ = model.sample_log(**sampling_kwargs)
generated_images = model.decode_first_stage(samples).detach().cpu()
generated_images = np.ascontiguousarray((255 * ((torch.clamp(generated_images, -1., 1.) + 1.0) / 2.0).permute(0, 2, 3, 1).numpy()).astype(np.uint8))
for i in range(generated_images.shape[0]):
    Image.fromarray(generated_images[i]).save(dest / f'sample{i}.jpg')