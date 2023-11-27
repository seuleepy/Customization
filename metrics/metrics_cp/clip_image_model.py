import os

import clip
import torch
from PIL import Image
import pathlib
from tqdm import tqdm

def clip_image_score(args, model):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def gather_file_paths(directory):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        return file_paths

    input_path = gather_file_paths(args.input_image_dir)
    gen_path = gather_file_paths(args.gen_image_dir)

    input_num_iter = len(input_path) // args.input_batch_size
    gen_num_iter = len(gen_path) // args.gen_batch_size

    input_remainder = len(input_path) % args.input_batch_size
    gen_remainder = len(gen_path) % args.gen_batch_size
    
    if input_remainder > 0:
        raise ValueError("The --input_batch_size encounters a remainder")
    if gen_remainder > 0:
        raise ValueError("The --gen_batch_size encounters a remainder")

    progress_bar = tqdm(
        range(0, int((input_num_iter * gen_num_iter))),
        desc="Steps",
        dynamic_ncols=True,
    )
    
    model, preprocess = clip.load(model, device)

    def make_input_list(img_path_list:list):
        input_list = []
        for path in img_path_list:
            image = Image.open(path)
            input = preprocess(image).unsqueeze(0).to(device)
            input_list.append(input)
        return input_list

    input_img = make_input_list(input_path)
    gen_img = make_input_list(gen_path)

    score = 0
    num_score = 0

    with torch.no_grad():

        for i in range(input_num_iter):
            i_start = i * args.input_batch_size
            i_end = i_start + args.input_batch_size
            batch_input = torch.cat(input_img[i_start:i_end])
            input_features = model.encode_image(batch_input)
            input_features /= input_features.norm(dim=-1, keepdim=True)

            for j in range(gen_num_iter):
                j_start = j * args.gen_batch_size
                j_end = j_start + args.gen_batch_size
                batch_gen = torch.cat(gen_img[j_start:j_end])
                gen_features = model.encode_image(batch_gen)
                gen_features /= gen_features.norm(dim=-1, keepdim=True)

                score_matrix = input_features @ gen_features.T
                score_ = torch.sum(score_matrix)
                num_score_ = score_matrix.numel()
                
                score += score_.item()
                num_score += num_score_

                progress_bar.update(1)

        score /= num_score

        return score
