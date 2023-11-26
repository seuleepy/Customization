import os
os.environ["MKL_NUM_THREADS"] = "2" # export MKL_NUM_THREADS=2
os.environ["NUMEXPR_NUM_THREADS"] = "2" # export NUMEXPR_NUM_THREADS=2
os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=2

import argparse
import json

def parse_args():
    parser=argparse.ArgumentParser()
    available_metrics = ['ImageImageCLIP', 'ImageImageDINO', 'ImageImageKID', "ImageImageFace", "ImagePromptCLIP"]
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        default=available_metrics,
        help="Select metrics from available metrics list",
    )
    parser.add_argument(
        "--input_image_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--gen_image_dir",
        type=str,
    )
    parser.add_argument(
        '--image_prompt_json',
        type=str,
        help = "json mapping from image_id -> prompt."
    )
    parser.add_argument(
        "--input_batch_size",
        type=int,
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
    )
    parser.add_argument(
        "--result_path",
        type=str,
        required=True,
    )
    available_clip = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    parser.add_argument(
        "--CLIP_model",
        type=str,
        nargs='+',
        default=available_clip,
        help="Select model(s) from available clip models list",
    )
    available_face = ["AdaFace", "VGG-Face", "Facenet", "OpenFace", "DeepFace", "ArcFace", "SFace"]
    parser.add_argument(
        "--Face_model",
        type=str,
        nargs="+",
        default=available_face,
        help="Select model(s) from available face models list"
    )
    parser.add_argument(
        "--Description",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if "ImageImageCLIP" in args.metrics or "ImageImageDINO" in args.metrics:
        if not args.gen_image_dir:
            parser.error("--gen_image_dir is required when using ImageImageCLIP or ImageImageDINO metric")
        if not args.input_batch_size:
            parser.error("--input_batch_size is required when using ImageImageCLIP or ImageImageDINO metrics")
        if not args.gen_batch_size:
            parser.error("--gen_batch_size is required when using ImageImageCLIP or ImageImageDINO metrics")
    if "ImageImageKID" in args.metrics or "ImageImageFace" in args.metrics:
        if not args.gen_image_dir:
            parser.error("--gen_image_dir is required when using ImageImageKID or ImageImageFace metrics")
        if not args.input_batch_size:
            parser.error("--input_batch_size is required when using ImageImageFace AdaFace metrics")
        if not args.gen_batch_size:
            parser.error("--gen_batch_size is required when sing ImageImageFace AdaFace metrics")
    if "ImagePromptCLIP" in args.metrics:
        if not args.image_prompt_json:
            parser.error("--image_prompt_json is required when using ImagePromptCLIP metrics")
        if not args.input_batch_size:
            parser.error("--input_batch_size is required when using ImagePromptCLIP metrics")
        
    return args 

args = parse_args()

if os.path.exists(f"{args.result_path}/score.json"):
    if os.path.getsize(f"{args.result_path}/score.json") > 0:
        with open(f"{args.result_path}/score.json", 'r') as file:
            existing_data = json.load(file)
else:
    existing_data = {}

if "ImageImageCLIP" in args.metrics:
    from metrics.clip_image_model import clip_image_score
    for model in args.CLIP_model:
        existing_data[f"ImageImageCLIP({model}) {args.Description}"] = clip_image_score(args, model)

if "ImageImageDINO" in args.metrics:
    from metrics.dino_model import dino_score
    existing_data[f"ImageImageDINO(ViT-S/16) {args.Description}"] = dino_score(args)

if "ImageImageKID" in args.metrics:
    from cleanfid import fid
    existing_data[f"ImageImageKID {args.Description}"] = fid.compute_kid(args.input_image_dir, args.gen_image_dir)

if "ImageImageFace" in args.metrics:
    from metrics.face_recognition_model import face_score
    for model in args.Face_model:
        existing_data[f"ImageImageFace({model}) {args.Description}"] = face_score(args, model)

if "ImagePromptCLIP" in args.metrics:
    from metrics.clip_text_model import clip_text_score
    for model in args.CLIP_model:
        existing_data[f"ImagePromptCLIP({model}) {args.Description}"] = clip_text_score(args, model)

with open(f"{args.result_path}/score.json", 'w') as file:
    json.dump(existing_data, file, indent=4)