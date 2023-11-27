import os
import numpy as np
from . import net
import torch
torch.set_num_threads(2)
from face_alignment import align
from deepface import DeepFace
from tqdm import tqdm
import gc

def face_score(args, model):

    gc.collect()
    torch.cuda.empty_cache()

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

    if model == "AdaFace":

        model_path = os.path.expanduser('~') + "/seulgi/Customization/metrics/adaface_ir50_webface4m.ckpt"

        adaface_models = {
            'ir_50':model_path,
        }

        def load_pretrained_model(architecture='ir_50'):
            # load model and pretrained statedict
            assert architecture in adaface_models.keys()
            model = net.build_model(architecture)
            statedict = torch.load(adaface_models[architecture])['state_dict']
            model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
            model.load_state_dict(model_statedict)
            model.eval()
            return model

        def to_input(pil_rgb_image):
            np_img = np.array(pil_rgb_image)
            brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
            tensor = torch.tensor([brg_img.transpose(2,0,1)]).float().to(device)
            return tensor

        model = load_pretrained_model('ir_50').to(device)

        num_fail = 0

        with torch.no_grad():
    
            features_input = []
            for input in input_path:
                aligned_rgb_input = align.get_aligned_face(input)
                bgr_tensor_input = to_input(aligned_rgb_input)
                features_input_, _ = model(bgr_tensor_input)
                features_input.append(features_input_)

            features_gen = []
            for gen in gen_path:
                aligned_rgb_gen = align.get_aligned_face(gen)
                if aligned_rgb_gen is None:
                    num_fail += 1
                    continue
                bgr_tensor_gen = to_input(aligned_rgb_gen)
                features_gen_, _ = model(bgr_tensor_gen)
                features_gen.append(features_gen_)

            score = 0
            num_score = 0

            progress_bar = tqdm(
                range(0, len(features_input) * len(features_gen)),
                desc="Steps",
                dynamic_ncols=True,
            )


            for input in features_input:

                for gen in features_gen:

                    score_matrix = input @ gen.T
                    score_ = torch.sum(score_matrix)
                    num_score_ = score_matrix.numel()

                    score += score_.item()
                    num_score += num_score_
                    progress_bar.update(1)

            score /= (num_score + len(input_path) * num_fail)

    elif model != "AdaFace":

        progress_bar = tqdm(
            range(0, len(input_path) * len(gen_path)),
            desc="Steps",
            dynamic_ncols=True,
        )


        score = 0
        num_score = 0
        num_fail = 0

        with torch.no_grad():

            for input in input_path:
                for gen in gen_path:
                    try:
                        result = DeepFace.verify(img1_path = input, img2_path = gen, model_name=model)
                        distance = result['distance']
                        score_ = 1 - distance
                    except Exception as e:
                        if "Face could not be detected" in str(e):
                            score_ = 0
                            num_fail += 1
                        else:
                            raise
                    score += score_
                    num_score += 1
                    progress_bar.update(1)
            
            score /= num_score

    return score