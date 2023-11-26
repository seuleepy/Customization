import os
import json

import clip
import torch
from PIL import Image
import pathlib
from tqdm import tqdm

def clip_text_score(args, model):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_paths = [os.path.join(args.input_image_dir, path) for path in os.listdir(args.input_image_dir) # ['./clipscore/example/images/image2.jpg', './clipscore/example/images/image1.jpg']
                if path.endswith(('.png', '.jpg', 'jpeg', 'tiff'))] 
    image_ids = [pathlib.Path(path).stem for path in image_paths] #['image2', 'image1']

    with open(args.image_prompt_json) as f:
        image_prompt = json.load(f)
    prompts = [image_prompt[img_id] for img_id in image_ids] #['a black dog wearing headphones looks at the camera as an orange cat walks in the background.', 'an orange cat and a grey cat are lying together.']

    model, preprocess = clip.load(model, device)

    image_input = []
    text_input = []

    for path, prompt in zip(image_paths, prompts):
        image = Image.open(path)
        image_input_ = preprocess(image).unsqueeze(0).to(device)
        text_input_ = torch.cat([clip.tokenize(prompt)]).to(device)
        image_input.append(image_input_)
        text_input.append(text_input_)

    num_iter = len(image_input) // args.input_batch_size
    remainder = len(image_input) % args.input_batch_size

    if remainder > 0:
        raise ValueError("The --input_batch_size encounters a remainder")

    progress_bar = tqdm(
        range(0, num_iter),
        desc="Steps",
        dynamic_ncols=True,
    )
    score = 0
    num_score = 0

    with torch.no_grad():

        for i in range(num_iter):
            start = i * args.input_batch_size
            end = start + args.input_batch_size

            batch_image = torch.cat(image_input[start:end])
            batch_text = torch.cat(text_input[start:end])

            image_features = model.encode_image(batch_image)
            text_features = model.encode_text(batch_text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            score_matrix = image_features @ text_features.T
            score_ = torch.sum(score_matrix)
            num_score_ = score_matrix.numel()

            score += score_.item()
            num_score += num_score_

            progress_bar.update(1)
        
    score /= num_score

    return score