import argparse
import os

from tqdm import tqdm

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import json

BICUBIC = InterpolationMode.BICUBIC

import torch

from diffusers import(
    DiffusionPipeline,
    StableDiffusionPipeline,
)
from profusion_diffusers import (
    StableDiffusionPromptNetPipeline,
    DDIMScheduler
)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--customization_model",
        required=True,
        choices=["DreamBooth", "TextualInversion", "CustomDiffusion", "ProFusion"],
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt_file_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input_img_path",
        type=str,
        help="Used in only ProFusion. It used to make ref_image_latents and ref_image_embed"
    )
    parser.add_argument(
        "--save_img_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--mapping_json_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--guidance_scale",
        default=None,
        help="It used to DreamBooth, TextualInversion and CustomDiffusion"
    )
    parser.add_argument(
        "--eta",
        default=None,
        help="USed to CustomDiffusion"
    )
    parser.add_argument(
        "--cfg",
        default=None,
        help="Used to ProFusion. Increase this if you want more information from the input image."
    )
    parser.add_argument(
        "--ref_cfg",
        default=None,
        help="Used to ProFusion. Increase this if you want more information from the prompt."
    )
    parser.add_argument(
        "--refine_step",
        default=None,
        help="If you consider conditions to be independent, input refine_step=0. It activates on fusion=True setting."
    )
    parser.add_argument(
        "--refine_emb_scale",
        default=None,
        help=(
        "Increase this if you want some more information from the input image",
        "Decrease this if text information is not correctly generated.",
        "Normally 0.4~0.9 should work."
        )
    )
    parser.add_argument(
        "--refine_cfg",
        default=None,
        help="Guidance for fusion step sampling"
    )
    parser.add_argument(
        "--refine_eta",
        default=None,
    )
    parser.add_argument(
        "--FusionSampling",
        action="store_true",
    )
    parser.add_argument(
        "--residual",
        default=None,
    )
    parser.add_argument(
        "--modifier_token_bin_name",
        default="<new1>.bin",
    )
    parser.add_argument(
        "--output_img_num",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    if args.prompt is None and args.prompt_file_path is None:
        parser.error("Specify either --prompt or --prompt_file_path")

    return args

args = parse_args()

if args.customization_model == "ProFusion":

    def process_img(img_file, random=False):
        if type(img_file) == str:
            img_file = [img_file]
            
        input_img = []
        for img in img_file:
            image = Image.open(img).convert('RGB')
            w, h = image.size
            crop = min(w, h)
            if random:
                image = T.Resize(560, interpolation=T.InterpolationMode.BILINEAR)(image)
                image = T.RandomCrop(512)(image)
                image = T.RandomHorizontalFlip()(image)
            else:
                image = image.crop(((w - crop) // 2, (h - crop) // 2, (w + crop) // 2, (h + crop) // 2))
            input_img_ = image = image.resize((512, 512), Image.LANCZOS)
            input_img.append(ToTensor()(image).unsqueeze(0))
        input_img = torch.cat(input_img).to("cuda").to(vae.dtype)
        img_latents = vae.encode(input_img * 2.0 - 1.0).latent_dist.sample()
        img_latents = img_latents * vae.config.scaling_factor

        img_4_clip = processor(input_img)
        vision_embeds = openclip.vision_model(img_4_clip, output_hidden_states=True)
        vision_hidden_states = vision_embeds.last_hidden_state
        return img_latents, vision_hidden_states, input_img_

    processor = Compose([
        Resize(224, interpolation=BICUBIC),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
# Load the trained model #################################################

if args.customization_model == "DreamBooth" or \
    args.customization_model == "TextualInversion":
    pipeline = StableDiffusionPipeline
elif args.customization_model == "CustomDiffusion":
    pipeline = DiffusionPipeline
elif args.customization_model == "ProFusion":
    pipeline = StableDiffusionPromptNetPipeline

if args.customization_model == "ProFusion":
    # must use DDIM when refine_step > 0
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = pipeline.from_pretrained(args.pretrained_model_name_or_path, scheduler = scheduler, torch_dtype=torch.float16).to("cuda")
elif args.customization_model == "CustomDiffusion":
    pipeline = pipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
else:
    pipeline = pipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16).to("cuda")

if args.customization_model == "CustomDiffusion":
    pipeline.unet.load_attn_procs(
        args.pretrained_model_name_or_path,
        weight_name="pytorch_custom_diffusion_weights.bin"
    )
    pipeline.load_textual_inversion(
        args.pretrained_model_name_or_path,
        weight_name=args.modifier_token_bin_name,
    )
############################################################################

pipeline.set_progress_bar_config(disable=True)

if args.customization_model == "ProFusion":
    
    openclip = pipeline.openclip
    vae = pipeline.vae

    gt_latents, vision_hidden_states, input_img_ = process_img(args.input_img_path)


kwargs={}

if args.customization_model == "DreamBooth" or \
    args.customization_model == "TextualInversion":
    kwargs["num_inference_steps"] = 50 if args.num_inference_steps is None else args.num_inference_steps
    kwargs["guidance_scale"] = 7.5 if args.guidance_scale is None else args.guidance_scale

elif args.customization_model == "CustomDiffusion":
    kwargs["num_inference_steps"] = 100 if args.num_inference_steps is None else args.num_inference_steps
    kwargs["guidance_scale"] = 6.0 if args.guidance_scale is None else args.guidance_scale
    kwargs["eta"] = 1.0 if args.eta is None else args.eta

elif args.customization_model == "ProFusion":

    kwargs["num_inference_steps"] = 50 if args.num_inference_steps is None else args.num_inference_steps
    kwargs["ref_image_latent"] = gt_latents
    kwargs["ref_image_embed"] = vision_hidden_states
    
    if not args.FusionSampling:
        kwargs["guidance_scale"] = 5.0 if args.cfg is None else args.cfg
        kwargs["res_prompt_scale"] = 0.0 if args.residual is None else args.residual
        kwargs["ref_prompt"] = None
        kwargs["guidance_scale_ref"] = 0
        kwargs["refine_step"] = 0
    
    elif args.FusionSampling:
        kwargs["guidance_scale"] = 7.0 if args.cfg is None else args.cfg
        kwargs["guidance_scale_ref"] = 5.0 if args.ref_cfg is None else args.ref_cfg
        kwargs["res_prompt_scale"] = 0.0 if args.residual is None else args.residual
        kwargs["refine_step"] = 1 if args.refine_step is None else args.refine_step
        kwargs["refine_eta"] = 1 if args.refine_eta is None else args.refine_eta
        kwargs["refine_emb_scale"] = 0.6 if args.refine_emb_scale is None else args.refine_emb_scale
        kwargs["refine_guidance_scale"] = 7.0 if args.refine_cfg is None else args.refine_cfg


if args.prompt is not None:
    
    if args.customization_model == "DreamBooth" or \
    args.customization_model == "TextualInversion" or \
    args.customization_model == "CustomDiffusion":
        kwargs["prompt"] = args.prompt
    elif args.customization_model == "ProFusion":
        kwargs["prompt"] = "a holder " + args.prompt
        if args.FusionSampling:
            kwargs["ref_prompt"] = "a person " + args.prompt

elif args.prompt is None:

    with open(args.prompt_file_path, 'r') as file:
        
        prompts = [line.strip() for line in file.readlines()]

if os.path.exists(args.mapping_json_path) and os.path.getsize(args.mapping_json_path) > 0:
    with open(args.mapping_json_path, 'r') as f:
        existing_data = json.load(f)
else:
    existing_data = {}

if args.prompt is not None:
    step = args.output_img_num
else:
    step = args.output_img_num * len(prompts)

progress_bar = tqdm(
    range(0, step),
    desc="Steps",
    dynamic_ncols=True,
)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.prompt is not None:
    for idx in range(args.output_img_num):
        
        image = pipeline(
            **kwargs,
        ).images[0]

        idx += 1
        idx = str(idx).zfill(4)
        
        save_img_name = args.save_img_name
        if args.output_img_num > 1:
            save_img_name = save_img_name + "_" + str(idx)

        image.save(f"{args.save_path}/{save_img_name}.jpg")
        progress_bar.update(1)

        existing_data[save_img_name] = args.prompt

elif args.prompt is None:
    idx = 0
    for prompt in prompts:
        for _ in range(args.output_img_num):
            
            idx += 1

            image = pipeline(
                prompt = prompt,
                **kwargs,
            ).images[0]
            
            save_img_name = args.save_img_name
            if args.output_img_num > 1:
                save_img_name = save_img_name + "_" + str(idx)

            image.save(f"{args.save_path}/{save_img_name}.jpg")
            progress_bar.update(1)

            existing_data[save_img_name] = prompt

with open(args.mapping_json_path, 'w') as f:
    json.dump(existing_data, f, indent = 4)
