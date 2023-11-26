import argparse
import warnings
from pathlib import Path
import logging
import copy
import itertools
import gc
import math
import shutil
from tqdm import tqdm

import transformers
import datasets
import torch
import torch.nn.functional as F

from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel
from transformers import AutoTokenizer, PretrainedConfig
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms as T
from huggingface_hub import HfApi, create_repo, model_info, upload_folder
from huggingface_hub.utils import insecure_hashlib


import diffusers
from diffusers.loaders import AttnProcsLayers
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.models.attention_processor import (
    CustomDiffusionAttnProcessor,
    CustomDiffusionAttnProcessor2_0,
    CustomDiffusionXFormersAttnProcessor,
)
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from profusion_diffusers import (
    PromptNetModel,
    DDIMScheduler,
)

from profusion_diffusers.utils import deprecate
from diffusers.utils.import_utils import is_xformers_available

from utils.main_utils import *

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training script.")
    parser.add_argument(
        "--customization_model",
        required=True,
        choices=['DreamBooth', 'TextualInversion', 'CustomDiffusion', 'ProFusion' ]
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "Used in DreamBooth and CustomDiffusion and TextualInversion"
            "ProFusion uses only pretrained model's tokenizer"
            "Pretrained tokenizer name or path if not the same as model_name"

        ),
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help=(
            "Used in DreamBooth and CustomDiffusion"
        "A folder containing the training data of class images."
        ),
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help=(
            "Used in DreamBooth and CustomDiffusion"
            "The prompt with identifier specifying the instance",

        ),
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help=(
            "Used in DreamBooth and CustomDiffusion"
            "The prompt to specify images in the same class as provided instance images."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help=(
            "Used in DreamBooth and CustomDiffusion"
            "Flag to add prior preservation loss."
        ),
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help=(
            "Used in DreamBooth and CustomDiffusion"
            "The weight of prior preservation loss."
        ),
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Used in DreamBooth and CustomDiffusion"
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="finetuning-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help=(
            "Used in DreamBooth"
            "Whether to train the text encoder. If set, the text encoder should be float32 precision."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help=(
            "Used in DreamBooth and CustomDiffusion"
            "Batch size (per device) for sampling images."
        ),
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help=("Number of hard resets of the lr in cosine_with_restarts scheduler."
              "It activates DreamBooth"
        ),
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help=("Power factor of the polynomial scheduler."
              "It activates DreamBooth"
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=("DreamBooth : 0, CustomDiffusion : 2"
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=(
            "A prompt that is used during validation to verify that the model is learning.",
    )
    )
    parser.add_argument(
        "--ref_validation_prompt",
        type=str,
        default=None,
        help=(
            "Used in ProFusion"
            "holder -> someone"
        )
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help=(
            "Used in DreamBooth and CustomDiffusion and TextaulInversion"
            "Number of images that should be generated during validation with `validation_prompt`."
            ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "DreamBooth : 100 CustomDiffusion : 50"
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Used in DreamBooth and CustomDiffusion"
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--offset_noise",
        action="store_true",
        default=False,
        help=(
            "It activates in DreamBooth"
            "Fine-tuning against a modified noise"
            " See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help=(
            "It activates in DreamBooth"
            "SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
            "More details here: https://arxiv.org/abs/2303.09556."
        ),
    )
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help=(
           "Used in DreamBooth"
           "Whether or not to pre-compute text embeddings.If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`."
        ),
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="Used in DreamBooth. The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--skip_save_text_encoder", action="store_true", required=False, help="Used in DreamBooth. Set to not save text encoder"
    )
    parser.add_argument(
        "--validation_images",
        required=False,
        default=None,
        nargs="+",
        help="이미지만 담고 있는 폴더를 줄 것. Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.",
    )
    parser.add_argument(
        "--class_labels_conditioning",
        required=False,
        default=None,
        help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )
    parser.add_argument(
        "--validation_scheduler",
        type=str,
        default="DPMSolverMultistepScheduler",
        choices=["DPMSolverMultistepScheduler", "DDPMScheduler"],
        help="Select which scheduler to use for validation. DDPMScheduler is recommended for DeepFloyd IF.",
    )
    parser.add_argument(
        "--real_prior",
        default=False,
        action="store_true",
        help="real images as prior.",
    )
    parser.add_argument(
        "--freeze_model",
        type=str,
        default="crossattn_kv",
        choices=["crossattn_kv", "crossattn"],
        help="crossattn to enable fine-tuning of all params in the cross attention",
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="A token to use as a modifier for the concept.",
    )
    parser.add_argument( 
        "--initializer_token", type=str, default="ktn+pll+ucd", help="A token to use as initializer word."
    )
    parser.add_argument("--hflip", 
                         action="store_true", help="Apply horizontal flip data augmentation.")
    parser.add_argument(
        "--noaug",
        action="store_true",
        help="Dont apply augmentation during data augmentation when this flag is enabled.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--learnable_property",
        type=str,
        default="object",
        help="Choose between 'object' and 'style'"
    )
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "Used in ProFusion"
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    # added args
    parser.add_argument("--promptnet_l2_reg", type=float, default=0.0, help="Regularization of outputs of PromptNet. Left for ablation study and potential future use")
    parser.add_argument("--residual_l2_reg", type=float, default=0.0, help="Regularization of outputs of residuals. Left for potential future use.")
    parser.add_argument("--region_kernel_size", type=int, default=8,
                        help="Kernel size used in generating prompt embedding, see PromptNet")
    parser.add_argument(
        "--add_prompt",
        type=str,
        default="A photo of ",
        help="add something before embedding, e.g. a face photo of {learned embeddings}"
    )
    parser.add_argument(
        "--finetuning",
        default=False,
        action="store_true",
        help=(
            "Used in ProFusion"
        ),
    )
    parser.add_argument(
        "--train_domain",
        default=False,
        action="store_true",
        help=(
            "Used in ProFusion"
        )
    )
    parser.add_argument(
        "--finetune_unet",
        action="store_true",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.customization_model == "ProFusion":
        if args.finetuning is False and args.train_domain is False:
            raise ValueError("You must specify either --finetuning or --train_domain")
        if args.finetuning and args.train_domain:
            raise ValueError("You must specify either --finetuning or --train_domain ")
        
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    if args.train_text_encoder and args.pre_compute_text_embeddings:
        raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")

    return args



check_min_version("0.24.0.dev0")

logger = get_logger(__name__)

args = parse_args()

if args.non_ema_revision is not None:
    deprecate(
        "non_ema_revision!=None",
        "0.15.0",
        message=(
            "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
            " use `--variant=non_ema` instead."
        ),
    )

logging_dir = Path(args.output_dir, args.logging_dir)

accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir, total_limit=args.checkpoints_total_limit)

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
)

if args.report_to == "wandb":
    if not is_wandb_available():
        raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    
# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
    if args.customization_model == "ProFusion":
        datasets.utils.logging.set_verbosity_warning()
else:
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    if args.customization_model == "ProFusion" and args.train_domain is True:
        datasets.utils.logging.set_verbosity_error()

# We need to initialize the trackers we use, and also store our configuration.
# The trackers initializes automatically on the main process.
if accelerator.is_main_process:
    tracker_config = vars(copy.deepcopy(args))
    tracker_config.pop("validation_images")
    accelerator.init_trackers(args.customization_model, config=tracker_config)

# Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
# This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
# TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
if args.customization_model == "DreamBooth":
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        ) 

# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)

# Used in CustomDiffusion
if args.concepts_list is None:
    args.concepts_list = [
        {
            "instance_prompt": args.instance_prompt,
            "class_prompt": args.class_prompt,
            "instance_data_dir": args.instance_data_dir,
            "class_data_dir": args.class_data_dir,
        }
    ]
else:
    with open(args.concepts_list, "r") as f:
        args.concepts_list = json.load(f)

# Generate class images if prior preservation is enabled.
if args.customization_model == "CustomDiffusion" and args.with_prior_preservation:
    for i, concept in enumerate(args.concepts_list):
        class_images_dir = Path(concept["class_data_dir"])
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True, exist_ok=True)
        if args.real_prior:
            assert (
                class_images_dir / "images"
            ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
            assert (
                len(list((class_images_dir / "images").iterdir())) == args.num_class_images
            ), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
            assert (
                class_images_dir / "caption.txt"
            ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
            assert (
                class_images_dir / "images.txt"
            ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt']}\" --class_data_dir {class_images_dir} --num_class_images {args.num_class_images}"
            concept["class_prompt"] = os.path.join(class_images_dir, "caption.txt")
            concept["class_data_dir"] = os.path.join(class_images_dir, "images.txt")
            args.concepts_list[i] = concept
            accelerator.wait_for_everyone()
        else:
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < args.num_class_images:
                torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                if args.prior_generation_precision == "fp32":
                    torch_dtype = torch.float32
                elif args.prior_generation_precision == "fp16":
                    torch_dtype = torch.float16
                elif args.prior_generation_precision == "bf16":
                    torch_dtype = torch.bfloat16
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    revision=args.revision,
                )
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = args.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(args.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

                sample_dataloader = accelerator.prepare(sample_dataloader)
                pipeline.to(accelerator.device)

                for example in tqdm(
                    sample_dataloader,
                    desc="Generating class images",
                    disable=not accelerator.is_local_main_process,
                ):
                    images = pipeline(example["prompt"]).images

                    for i, image in enumerate(images):
                        hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = (
                            class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        )
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# Generate class images if prior preservation is enabled.
if args.customization_model == "DreamBooth" and args.with_prior_preservation:
    class_images_dir = Path(args.class_data_dir)
    if not class_images_dir.exists():
        class_images_dir.mkdir(parents=True)
    cur_class_images = len(list(class_images_dir.iterdir()))

    if cur_class_images < args.num_class_images:
        torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
        if args.prior_generation_precision == "fp32":
            torch_dtype = torch.float32
        elif args.prior_generation_precision == "fp16":
            torch_dtype = torch.float16
        elif args.prior_generation_precision == "bf16":
            torch_dtype = torch.bfloat16
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            revision=args.revision,
        )
        pipeline.set_progress_bar_config(disable=True)

        num_new_images = args.num_class_images - cur_class_images
        logger.info(f"Number of class images to sample: {num_new_images}.")

        sample_dataset = PromptDataset(args.class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

        sample_dataloader = accelerator.prepare(sample_dataloader)
        pipeline.to(accelerator.device)

        for example in tqdm(
            sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
        ):
            images = pipeline(example["prompt"]).images

            for i, image in enumerate(images):
                hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Handle the repository creation
if accelerator.is_main_process:
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        ).repo_id

# Load the tokenizer
if args.tokenizer_name: # DreamBooth, TextualInversion, CutomDiffusion
    if args.customization_model == "TextualInversion":
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
elif args.pretrained_model_name_or_path:
    if args.customization_model == "TextualInversion":
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    elif args.customization_model == "ProFusion":
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )
        if args.finetuning: pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

# import correct text encoder class
if args.finetuning:
    pass
else:
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

# Load scheduler
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

# Load encoder
if args.customization_model == "ProFusion":
    if args.finetuning: pass
    else:
        openclip = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", revision=args.revision)
        text_encoder = openclip.text_model  # CLIPTextTransformer
        vision_encoder = openclip.vision_model  # CLIPVisionTransformer

else:
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)

# Load vae
if model_has_vae(args):
    if args.finetuning:
        pass
    else:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
else:
    vae = None

# Load unet
if not args.finetuning:
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

# Used on ProFusion
processor = Compose([
    Resize(224, interpolation=PIL.Image.Resampling.BICUBIC),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), ])

# For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
# as these weights are only used for inference, keeping weights in full precision is not required.
weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

# create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        for model in models:
            sub_dir = "unet" if isinstance(model, type(accelerator.unwrap_model(unet))) else "text_encoder"
            model.save_pretrained(os.path.join(output_dir, sub_dir))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

def load_model_hook(models, input_dir):
    while len(models) > 0:
        # pop models so that they are not loaded again
        model = models.pop()

        if isinstance(model, type(accelerator.unwrap_model(text_encoder))):
            # load transformers style into model
            load_model = text_encoder_cls.from_pretrained(input_dir, subfolder="text_encoder")
            model.config = load_model.config
    else:
            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

    model.load_state_dict(load_model.state_dict())
    del load_model

# Used in DreamBooth
if args.customization_model == "DreamBooth":
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

# Used in CustomDiffusion
elif args.customization_model == "CustomDiffusion":
    modifier_token_id = []
    initializer_token_id = []
    if args.modifier_token is not None:
        args.modifier_token = args.modifier_token.split("+")
        args.initializer_token = args.initializer_token.split("+")
        if len(args.modifier_token) > len(args.initializer_token):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(
            args.modifier_token, args.initializer_token[: len(args.modifier_token)]
        ):
            # Add the placeholder token in tokenizer
            num_added_tokens = tokenizer.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer.encode([initializer_token], add_special_tokens=False)
            print(token_ids)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id.append(token_ids[0])
            modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for x, y in zip(modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]

# Used in TextualInversion
# Add the placeholder token in tokenizer
elif args.customization_model == "TextualInversion":
    placeholder_tokens = [args.placeholder_token]
    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    
    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

elif args.customization_model == "ProFusion" and args.train_domain:
    res_channels = (320, 640, 1280, 1280)
    try:
        promptnet = PromptNetModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="promptnet", use_res=True, with_noise=True, with_ebm=False,
        )
    except:
        promptnet = PromptNetModel.from_pretrained(  # Initialize the promptnet with weights from Stable Diffusion
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, low_cpu_mem_usage=False,
            region_kernel_size=args.region_kernel_size, with_noise=True,
            use_clip_vision_embeds=True, use_res=True, with_ebm=False,
            res_block_out_channels=res_channels, prompt_channel=1024,
        )

if args.customization_model == "ProFusion" and args.finetuning:
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = StableDiffusionPromptNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=scheduler,
        torch_dtype=weight_dtype
        )
    pipeline.to("cuda")
    vae = pipeline.vae
    tokenizer = pipeline.tokenizer
    openclip = pipeline.openclip
    text_encoder = openclip.text_model
    vision_encoder = openclip.vision_model
    promptnet = pipeline.promptnet
    unet = pipeline.unet


# Freeze vae
if vae is not None:
    vae.requires_grad_(False)
if args.customization_model == "DreamBooth":
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
elif (args.customization_model == "TextualInversion") or \
    (args.customization_model =="CustomDiffusion" and args.train_domain is True):
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    if args.customization_model == "CustomDiffusion" and \
        args.modifier_token is None:
        text_encoder.requires_grad_(False)
elif args.customization_model == "ProFusion":
    openclip.requires_grad_(False) #어차피 위에 textencoder는 freeze가 안 되지 않나...

# Freeze unet
if args.customization_model == "TextualInversion" or \
    args.customization_model == "CustomDiffusion" or \
    args.customization_model == "ProFusion":
    unet.requires_grad_(False)


if vae is not None:
    vae.to(accelerator.device, dtype=weight_dtype)
if args.customization_model == "ProFusion":
    openclip.to(accelerator.device, dtype=weight_dtype)
    if args.train_domain is True:
        if args.use_ema:
            ema_promptnet.to(accelerator.device)
            if ema_promptnet.model is not None:
                ema_promptnet.model.to(device=accelerator.device, dtype=weight_dtype)
    elif args.finetuning and not args.finetune_unet:
        unet.to(accelerator.device, dtype=weight_dtype)
elif args.customization_model == "CustomDiffusion":
    if accelerator.mixed_precision != "fp16" and args.modifier_token is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
elif args.customization_model == "DreamBooth":
    if not args.train_text_encoder and text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
elif args.customization_model == "TextualInversion":
    unet.to(accelerator.device, dtype=weight_dtype)
# args.used_ema used in only ProFusion
# Create EMA for the unet.
if args.customization_model == "ProFusion" and args.use_ema:
    try:
        ema_promptnet = PromptNetModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="promptnet", use_res=True,
            with_noise=True, with_ebm=False,
        )
    except:
        ema_promptnet = PromptNetModel.from_pretrained(  # Initialize the promptnet with weights from Stable Diffusion
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, low_cpu_mem_usage=False,
            region_kernel_size=args.region_kernel_size, with_noise=True,
            use_clip_vision_embeds=True, use_res=True, with_ebm=False,
            res_block_out_channels=res_channels, prompt_channel=1024,
        )
    ema_promptnet = EMAModel(ema_promptnet.parameters(), decay=0.9999, model_cls=PromptNetModel, model_config=ema_promptnet.config, model=ema_promptnet)

if args.customization_model == "CustomDiffusion":
    attention_class = (
        CustomDiffusionAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention")
        else CustomDiffusionAttnProcessor
        )
        
if args.enable_xformers_memory_efficient_attention:
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        if args.customization_model == "CustomDiffusion":
            attention_class = CustomDiffusionXFormersAttnProcessor
        else:
            unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

if args.customization_model == "CustomDiffusion":
    train_kv = True
    train_q_out = False if args.freeze_model == "crossattn_kv" else True
    custom_diffusion_attn_procs = {}

    st = unet.state_dict()
    for name, _ in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        layer_name = name.split(".processor")[0]
        weights = {
            "to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
            "to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
        }
        if train_q_out:
            weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
            weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
            weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
        if cross_attention_dim is not None:
            custom_diffusion_attn_procs[name] = attention_class(
                train_kv=train_kv,
                train_q_out=train_q_out,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            ).to(unet.device)
            custom_diffusion_attn_procs[name].load_state_dict(weights)
        else:
            custom_diffusion_attn_procs[name] = attention_class(
                train_kv=False,
                train_q_out=False,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
            )
    del st
    unet.set_attn_processor(custom_diffusion_attn_procs)
    custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)

    accelerator.register_for_checkpointing(custom_diffusion_layers)

if args.customization_model == "DreamBooth":
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if args.train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

if args.gradient_checkpointing:
    if args.customization_model == "ProFusion" and args.train_domain:
        promptnet.enable_gradient_checkpointing()
    elif args.customization_model == "TextualInversion":
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()
    elif args.customization_model == "CustomDiffusion":
        unet.enable_gradient_checkpointing()
        if args.modifier_token is not None:
            text_encoder.gradient_checkpointing_enable()
    
# Enable TF32 for faster training on Ampere GPUs,
# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
if args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

if args.scale_lr:
    args.learning_rate = (
        args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    )
    if args.customization_model == "CustomDiffusion" and args.with_prior_preservation:
        args.learning_rate = args.learning_rate * 2.0

# Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
if args.use_8bit_adam:
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
        )

    optimizer_class = bnb.optim.AdamW8bit
else:
    optimizer_class = torch.optim.AdamW

# Optimizer creation
if args.customization_model == "DreamBooth":
    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )

elif args.customization_model == "CustomDiffusion":
    params_to_optimize = itertools.chain(text_encoder.get_input_embeddings().parameters(), custom_diffusion_layers.parameters()) \
        if args.modifier_token is not None \
            else custom_diffusion_layers.parameters()
elif args.customization_model == "TextualInversion":
    params_to_optimize = text_encoder.get_input_embeddings().parameters() # only optimize the embeddings
elif args.customization_model == "ProFusion":
    params_to_optimize = promptnet.parameters()
    if args.finetune_unet:
        params_to_optimize = list(params_to_optimize)
        for (name, param) in unet.named_parameters():
            if 'to_q' in name or 'to_k' in name or 'to_v' in name:
                param.requires_grad = True
                params_to_optimize.append(param)

optimizer = optimizer_class(
    params_to_optimize,
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)

# args.pre_compute_text_embeddings used in only DreamBooth
if args.pre_compute_text_embeddings:

    def compute_text_embeddings(prompt):
        with torch.no_grad():
            text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=args.tokenizer_max_length)
            prompt_embeds = encode_prompt(
                text_encoder,
                text_inputs.input_ids,
                text_inputs.attention_mask,
                text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
            )

        return prompt_embeds

    pre_computed_encoder_hidden_states = compute_text_embeddings(args.instance_prompt)
    validation_prompt_negative_prompt_embeds = compute_text_embeddings("")

    if args.validation_prompt is not None:
        validation_prompt_encoder_hidden_states = compute_text_embeddings(args.validation_prompt)
    else:
        validation_prompt_encoder_hidden_states = None

    if args.class_prompt is not None:
        pre_computed_class_prompt_encoder_hidden_states = compute_text_embeddings(args.class_prompt)
    else:
        pre_computed_class_prompt_encoder_hidden_states = None

    text_encoder = None
    tokenizer = None

    gc.collect()
    torch.cuda.empty_cache()
else:
    pre_computed_encoder_hidden_states = None
    validation_prompt_encoder_hidden_states = None
    validation_prompt_negative_prompt_embeds = None
    pre_computed_class_prompt_encoder_hidden_states = None

# Dataset creation
if args.customization_model == "ProFusion":
    if args.train_domain:
        train_dataset = ProFusionDataset(args)
    elif args.finetuning:
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
        
        mini_fnames = [os.path.join(args.instance_data_dir, file) for file in os.listdir(args.instance_data_dir)]
        #train_dataset = ProFusionFineTuningDataset(mini_fnames)
        latents_, vision_hidden_states_batch_, _ = process_img(mini_fnames, True)

elif args.customization_model == "DreamBooth":
    train_dataset = DreamBoothDataset(
    instance_data_root=args.instance_data_dir,
    instance_prompt=args.instance_prompt,
    class_data_root=args.class_data_dir if args.with_prior_preservation else None,
    class_prompt=args.class_prompt,
    class_num=args.num_class_images,
    tokenizer=tokenizer,
    size=args.resolution,
    center_crop=args.center_crop,
    encoder_hidden_states=pre_computed_encoder_hidden_states,
    class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
    tokenizer_max_length=args.tokenizer_max_length,
)
    
elif args.customization_model == "TextualInversion":
    train_dataset = TextualInversionDataset(
    data_root=args.instance_data_dir,
    tokenizer=tokenizer,
    size=args.resolution,
    placeholder_token=(" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids))),
    repeats=args.repeats,
    learnable_property=args.learnable_property,
    center_crop=args.center_crop,
    set="train",
)
    
elif args.customization_model == "CustomDiffusion":
    train_dataset = CustomDiffusionDataset(
    concepts_list=args.concepts_list,
    tokenizer=tokenizer,
    with_prior_preservation=args.with_prior_preservation,
    size=args.resolution,
    mask_size=vae.encode(
        torch.randn(1, 3, args.resolution, args.resolution).to(dtype=weight_dtype).to(accelerator.device)
    )
    .latent_dist.sample()
    .size()[-1],
    center_crop=args.center_crop,
    num_class_images=args.num_class_images,
    hflip=args.hflip,
    aug=not args.noaug,
)

# DataLoader creation
if not args.finetuning:
    if args.customization_model == "DreamBooth":
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=lambda examples: collate_fn_db(examples, args.with_prior_preservation),
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )
    if args.customization_model == "CustomDiffusion":
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=lambda examples: collate_fn_cd(examples, args.with_prior_preservation),
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )
    elif args.customization_model == "TextualInversion":
        train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
        )
    elif args.customization_model == "ProFusion":
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn_pf,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

else:
    train_dataloader = None

if args.customization_model == "TextualInversion":
    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)

# Scheduler and math around the number of training steps.
overrode_max_train_steps = False

if train_dataloader is not None:
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
else:
    num_update_steps_per_epoch = math.ceil((args.train_batch_size) / args.gradient_accumulation_steps)
if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

if not args.finetuning:
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
elif args.finetuning:
    promptnet.to(dtype=torch.float32)
    if args.finetune_unet:
        unet.to(dtype=torch.float32)

# Prepare everything with our `accelerator`.
if args.customization_model == "DreamBooth":
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
elif args.customization_model == "TextualInversion":
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )
elif args.customization_model == "CustomDiffusion":
    if args.modifier_token is not None:
        custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler
        )
elif args.customization_model == "ProFusion" and args.train_domain:
    promptnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        promptnet, optimizer, train_dataloader, lr_scheduler
    )
elif args.customization_model == "ProFusion" and args.finetuning:
    if args.finetune_unet:
        promptnet, unet, optimizer = accelerator.prepare(promptnet, unet, optimizer)
    else:
        promptnet, optimizer = accelerator.prepare(promptnet, optimizer)


# We need to recalculate our total training steps as the size of the training dataloader may have changed.
if train_dataloader is not None:
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
else:
    num_update_steps_per_epoch = math.ceil((args.train_batch_size) / args.gradient_accumulation_steps)
if overrode_max_train_steps:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
# Afterwards we recalculate our number of training epochs
args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

# Train!
total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

logger.info("***** Running training *****")
if train_dataloader is not None:
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
else:
    logger.info(f"  Num examples = {len(mini_fnames)}")
    logger.info(f"  Num batches each epoch = {args.train_batch_size}")
logger.info(f"  Num Epochs = {args.num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {args.max_train_steps}")
global_step = 0
first_epoch = 0

# Potentially load in the weights and states from a previous save
if args.resume_from_checkpoint:
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
        initial_global_step = 0
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
else:
    initial_global_step = 0

progress_bar = tqdm(
    range(0, args.max_train_steps if not args.finetuning else args.num_train_epochs),
    initial=initial_global_step,
    desc="Steps",
    # Only show the progress bar once on each machine.
    disable=not accelerator.is_local_main_process,
)

# keep original embeddings as reference
if args.customization_model == "TextualInversion":
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

if args.customization_model == "ProFusion" and args.finetuning:
    promptnet.train()
    if args.finetune_unet:
        unet.train()

for epoch in range(first_epoch, args.num_train_epochs):
    if args.customization_model == "DreamBooth" or \
        args.customization_model == "CustomDiffusion":
        unet.train()
        if args.customization_model == "DreamBooth" and args.train_text_encoder:
            text_encoder.train()
        elif args.customization_model == "CustomDiffusion" and \
            args.modifier_token is not None:
            text_encoder.train()
    elif args.customization_model == "TextualInversion":
        text_encoder.train()
    elif args.customization_model =="ProFusion":
        if args.train_domain:
            promptnet.train()
            train_loss = 0.0
            placeholder_pre_prompt_ids = tokenizer(args.add_prompt, padding=True, return_tensors="pt")["input_ids"]
            placeholder_pre_prompt_ids = placeholder_pre_prompt_ids.reshape(-1)
            print(placeholder_pre_prompt_ids)

    if args.customization_model == "ProFusion" and args.finetuning:

        idx = torch.randperm(latents_.shape[0])
        ref_latents = latents_[idx][:args.train_batch_size]
        vision_hidden_states_batch = vision_hidden_states_batch_[idx][:args.train_batch_size]
        idx_2 = torch.randperm(latents_.shape[0])
        latents = latents_[idx_2][:args.train_batch_size]

        placeholder_pre_prompt_ids = tokenizer("a photo of ", padding=True, return_tensors="pt")["input_ids"]
        placeholder_pre_prompt_ids = placeholder_pre_prompt_ids.reshape(-1)

        noise = torch.randn_like(latents)

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        bsz = latents.shape[0]
        
        
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        pseudo_prompt, _, _ = promptnet(sample=ref_latents, timestep=timesteps, encoder_hidden_states=vision_hidden_states_batch, promptnet_cond=noisy_latents, return_dict=False, )
        
        placeholder_prompt_ids = torch.cat([placeholder_pre_prompt_ids[:-1].to(latents.device), torch.tensor([0] * pseudo_prompt.shape[1]).to(latents.device), placeholder_pre_prompt_ids[-1:].to(latents.device)], dim=-1)
        
        pseudo_hidden_states = text_encoder.embeddings(placeholder_prompt_ids)
        pseudo_hidden_states = pseudo_hidden_states.repeat((pseudo_prompt.shape[0], 1, 1))
        pseudo_hidden_states[:, len(placeholder_pre_prompt_ids) - 1: pseudo_prompt.shape[1] + len(placeholder_pre_prompt_ids) - 1, :] = pseudo_prompt 
        causal_attention_mask = text_encoder._build_causal_attention_mask(pseudo_hidden_states.shape[0], pseudo_hidden_states.shape[1], pseudo_hidden_states.dtype).to(pseudo_hidden_states.device)
        encoder_outputs = text_encoder.encoder(pseudo_hidden_states, causal_attention_mask=causal_attention_mask, output_hidden_states=True)
        encoder_hidden_states = text_encoder.final_layer_norm(encoder_outputs.hidden_states[-2]).to(dtype=latents.dtype)

        outputs_ = unet(noisy_latents, timesteps, encoder_hidden_states, down_block_additional_residuals=None, mid_block_additional_residual=None, res_scale=0.0)
        
        
        loss = ((outputs_.sample.float() - target.float()) ** 2).mean((1, 2, 3)).mean()
        
        accelerator.backward(loss)
        # if accelerator.sync_gradients:
        #     accelerator.clip_grad_norm_(promptnet.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        logs = {"train_loss": loss}
        progress_bar.update(1)
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=int(epoch+1))
    
    if train_dataloader is not None:
        for step, batch in enumerate(train_dataloader):
            if args.customization_model == "DreamBooth":
                with accelerator.accumulate(unet):
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                    if vae is not None:
                        # Convert images to latent space
                        model_input = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                        model_input = model_input * vae.config.scaling_factor
                    else:
                        model_input = pixel_values

                    # Sample noise that we'll add to the model input
                    if args.offset_noise:
                        noise = torch.randn_like(model_input) + 0.1 * torch.randn(
                            model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device
                        )
                    else:
                        noise = torch.randn_like(model_input)
                    bsz, channels, height, width = model_input.shape
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                    # Get the text embedding for conditioning
                    if args.pre_compute_text_embeddings:
                        encoder_hidden_states = batch["input_ids"]
                    else:
                        encoder_hidden_states = encode_prompt(
                            text_encoder,
                            batch["input_ids"],
                            batch["attention_mask"],
                            text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                        )

                    if accelerator.unwrap_model(unet).config.in_channels == channels * 2:
                        noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)
                    
                    if args.class_labels_conditioning == "timesteps":
                        class_labels = timesteps
                    else:
                        class_labels = None

                    # Predict the noise residual
                    model_pred = unet(
                        noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels
                    ).sample

                    if model_pred.shape[1] == 6:
                        model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)
                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Compute instance loss
                    if args.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(noise_scheduler, timesteps)
                        base_weight = (
                            torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                        )

                        if noise_scheduler.config.prediction_type == "v_prediction":
                            # Velocity objective needs to be floored to an SNR weight of one.
                            mse_loss_weights = base_weight + 1
                        else:
                            # Epsilon and sample both use the same loss weights.
                            mse_loss_weights = base_weight
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    if args.with_prior_preservation:
                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            elif args.customization_model == "TextualInversion":
                with accelerator.accumulate(text_encoder):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    
                    accelerator.backward(loss)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Let's make sure we don't update any embedding weights besides the newly added token
                    index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                    index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

                    with torch.no_grad():
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]

            elif args.customization_model =="CustomDiffusion":
                    with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):
                        # Convert images to latent space
                        latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                        timesteps = timesteps.long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get the text embedding for conditioning
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                        # Predict the noise residual
                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                        # Get the target for loss depending on the prediction type
                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                        if args.with_prior_preservation:
                            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                            target, target_prior = torch.chunk(target, 2, dim=0)
                            mask = torch.chunk(batch["mask"], 2, dim=0)[0]
                            # Compute instance loss
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                            loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                            # Compute prior loss
                            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                            # Add the prior loss to the instance loss.
                            loss = loss + args.prior_loss_weight * prior_loss
                        else:
                            mask = batch["mask"]
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                            loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()
                        accelerator.backward(loss)
                        # Zero out the gradients for all token embeddings except the newly added
                        # embeddings for the concept, as we only want to optimize the concept embeddings
                        if args.modifier_token is not None:
                            if accelerator.num_processes > 1:
                                grads_text_encoder = text_encoder.module.get_input_embeddings().weight.grad
                            else:
                                grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
                            # Get the index for tokens that we want to zero the grads for
                            index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_id[0]
                            for i in range(len(modifier_token_id[1:])):
                                index_grads_to_zero = index_grads_to_zero & (
                                    torch.arange(len(tokenizer)) != modifier_token_id[i]
                                )
                            grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[
                                index_grads_to_zero, :
                            ].fill_(0)

                        if accelerator.sync_gradients:
                            params_to_clip = (
                                itertools.chain(text_encoder.parameters(), custom_diffusion_layers.parameters())
                                if args.modifier_token is not None
                                else custom_diffusion_layers.parameters()
                            )
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            elif args.customization_model == "ProFusion" and args.train_domain:
                with accelerator.accumulate(promptnet):
                    # Convert images to latent space
                    ref_latents = vae.encode(batch["ref_pixel_values"].to(weight_dtype)).latent_dist.sample()
                    ref_latents = ref_latents * vae.config.scaling_factor
                    img_4_clip = processor((batch["ref_pixel_values"].to(weight_dtype) + 1.) / 2.)
                    vision_embeds = vision_encoder(img_4_clip, output_hidden_states=True)
                    vision_hidden_states = vision_embeds.last_hidden_state

                    latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # ref_mask = (torch.rand((latents.shape[0], 1, 1, 1)) < 0.5).to(device=latents.device, dtype=latents.dtype)
                    # ref_latents = ref_latents*ref_mask + latents*(1-ref_mask)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # we don't use residuals, but the code is kept for potential future use.
                    pseudo_prompt, down_residuals, mid_residuals = promptnet(sample=ref_latents, timestep=timesteps,
                                                                            encoder_hidden_states=vision_hidden_states,
                                                                            promptnet_cond=noisy_latents,
                                                                            return_dict=False, )

                    # Process the pseudo prompt
                    placeholder_prompt_ids = torch.cat([placeholder_pre_prompt_ids[:-1].to(latents.device),
                                                        torch.tensor([0] * pseudo_prompt.shape[1]).to(latents.device),
                                                        placeholder_pre_prompt_ids[-1:].to(latents.device)],
                                                    dim=-1)

                    pseudo_hidden_states = text_encoder.embeddings(placeholder_prompt_ids)

                    pseudo_hidden_states = pseudo_hidden_states.repeat((pseudo_prompt.shape[0], 1, 1))

                    pseudo_hidden_states[:,
                    len(placeholder_pre_prompt_ids) - 1: pseudo_prompt.shape[1] + len(placeholder_pre_prompt_ids) - 1,
                    :] = pseudo_prompt

                    # the causal mask is important, we explicitly write it out because we are doing something inside the model
                    # don't forget about it if you try to customize the code
                    # if you have any question, please refer to https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
                    causal_attention_mask = text_encoder._build_causal_attention_mask(pseudo_hidden_states.shape[0],
                                                                                    pseudo_hidden_states.shape[1],
                                                                                    pseudo_hidden_states.dtype).to(
                        pseudo_hidden_states.device)
                    encoder_outputs = text_encoder.encoder(pseudo_hidden_states,
                                                        causal_attention_mask=causal_attention_mask,
                                                        output_hidden_states=True)

                    encoder_hidden_states = text_encoder.final_layer_norm(encoder_outputs.hidden_states[-2])

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    p_gamma = 0.
                    alpha_prod_t = noise_scheduler.alphas_cumprod.to(accelerator.device)[timesteps].reshape((noisy_latents.shape[0], 1, 1, 1))
                    weight = (1 - alpha_prod_t)**p_gamma

                    outputs_ = unet(noisy_latents, timesteps, encoder_hidden_states,
                                            down_block_additional_residuals=None,
                                            mid_block_additional_residual=None)
                    model_pred_no_res = outputs_.sample

                    unet_loss = (weight * ((model_pred_no_res.float() - target.float()) ** 2).mean((1, 2, 3))).mean()

                    reg = args.promptnet_l2_reg * pseudo_prompt.square().mean()
                    reg = reg + (args.residual_l2_reg * mid_residuals.square()).mean()
                    for down_residual in down_residuals:
                        reg = reg + (args.residual_l2_reg * down_residual.square()).mean()

                    loss = unet_loss +  reg

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(promptnet.parameters(), args.max_grad_norm)


                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        if args.use_ema:
                            ema_promptnet.step(promptnet.parameters())
                        progress_bar.update(1)
                        global_step += 1
                        accelerator.log({"train_loss": train_loss}, step=global_step)
                        train_loss = 0.0

                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                        f" {args.validation_prompt}."
                    )
                    if global_step % 10000 == 0 and global_step > 0 and accelerator.is_main_process:
                        if args.use_ema:
                            log_validation(args, unet, text_encoder, tokenizer, weight_dtype, global_step,
                            vae=vae, accelerator=accelerator, openclip=openclip, promptnet=ema_promptnet.model, processor=processor)
                        else:
                            log_validation(args, unet, text_encoder, tokenizer, weight_dtype, global_step,
                                vae=vae, accelerator=accelerator, openclip=openclip, promptnet=promptnet, processor=processor)

                    logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)

                    if global_step >= args.max_train_steps:
                        break

            # Checks if the accelerator has performed an optimization step behind the scenes
            if args.customization_model == "DreamBooth" or \
                args.customization_model =="CustomDiffusion":
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                            
                        if args.customization_model =="DreamBooth":
                            if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                                logger.info(
                                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                                    f" {args.validation_prompt}."
                                )
                                images = log_validation(
                                    args,
                                    unet,
                                    text_encoder,
                                    tokenizer,
                                    weight_dtype,
                                    global_step,
                                    vae=vae,
                                    accelerator=accelerator,
                                    prompt_embeds=validation_prompt_encoder_hidden_states,
                                    negative_prompt_embeds=validation_prompt_negative_prompt_embeds,
                                )

            elif args.customization_model == "TextualInversion":
                if accelerator.sync_gradients:
                    images = []
                    progress_bar.update(1)
                    global_step += 1
                    if global_step % args.save_steps == 0:
                        weight_name = (
                            f"learned_embeds-steps-{global_step}.bin"
                            if args.no_safe_serialization
                            else f"learned_embeds-steps-{global_step}.safetensors"
                        )
                        save_path = os.path.join(args.output_dir, weight_name)
                        logger.info("Saving embeddings")
                        save_progress(
                            text_encoder,
                            placeholder_token_ids,
                            accelerator,
                            args,
                            save_path,
                            safe_serialization=not args.no_safe_serialization,
                        )

                    if accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                        if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                            logger.info(
                                f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                                f" {args.validation_prompt}."
                            )
                            images = log_validation(
                                args, unet, text_encoder, tokenizer, weight_dtype, global_step,
                                vae=vae, accelerator=accelerator,
                            )
            if args.customization_model == "DreamBooth" or \
                args.customization_model == "TextualInversion" or \
                    args.customization_model =="CustomDiffusion":
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break
                
                if args.customization_model == "CustomDiffusion":
                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                            f" {args.validation_prompt}."
                        )
                        images = log_validation(
                            args,
                            unet,
                            text_encoder,
                            tokenizer,
                            weight_dtype,
                            global_step,
                            accelerator=accelerator,
                        )
# Create the pipeline using the trained modules and save it.
accelerator.wait_for_everyone()
pipeline_args={}
if accelerator.is_main_process:
    if args.customization_model == "ProFusion":
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_promptnet.copy_to(promptnet.parameters())
        pipeline_args["openclip"]=openclip
        pipeline_args["vae"]=vae
        pipeline_args["unet"]=accelerator.unwrap_model(unet)
        pipeline_args["promptnet"]=accelerator.unwrap_model(promptnet)
        pipeline_args["tokenizer"]=tokenizer
    elif args.customization_model == "CustomDiffusion":
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir, safe_serialization=not args.no_safe_serialization)
        logger.info("Saving embeddings")
        save_new_embed(
            text_encoder,
            modifier_token_id,
            accelerator,
            args,
            args.output_dir,
            safe_serialization=not args.no_safe_serialization,
        )
        pipeline_args["torch_dtype"]=weight_dtype
    elif args.customization_model == "DreamBooth":
        if text_encoder is not None:
            pipeline_args["text_encoder"] = accelerator.unwrap_model(text_encoder)
        if args.skip_save_text_encoder:
            pipeline_args["text_encoder"] = None
        pipeline_args["unet"]=accelerator.unwrap_model(unet)

    elif args.customization_model == "TextualInversion":
        pipeline_args["text_encoder"]=accelerator.unwrap_model(text_encoder)    
        pipeline_args["vae"]=vae
        pipeline_args["unet"]=unet
        pipeline_args["tokenizer"]=tokenizer
    
    # create pipeline (note: unet and vae are loaded again in float32)
    if args.customization_model == "ProFusion":
        pipeline = StableDiffusionPromptNetPipeline
    elif args.customization_model == "TextualInversion":
        pipeline = StableDiffusionPipeline
    else:
        pipeline = DiffusionPipeline

    pipeline = pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        **pipeline_args,
    )

    if args.customization_model == "DreamBooth":
        scheduler_args = {}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    elif args.customization_model == "CustomDiffusion":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        weight_name = (
            "pytorch_custom_diffusion_weights.safetensors"
            if not args.no_safe_serialization
            else "pytorch_custom_diffusion_weights.bin"
        )
        pipeline.unet.load_attn_procs(args.output_dir, weight_name=weight_name)
        for token in args.modifier_token:
            token_weight_name = f"{token}.safetensors" if not args.no_safe_serialization else f"{token}.bin"
            pipeline.load_textual_inversion(args.output_dir, weight_name=token_weight_name)
    
    pipeline.save_pretrained(args.output_dir)

    if args.customization_model == "TextualInversion":
        # Save the newly trained embeddings
        weight_name = "learned_embeds.bin" if args.no_safe_serialization else "learned_embeds.safetensors"
        save_path = os.path.join(args.output_dir, weight_name)
        save_progress(
            text_encoder,
            placeholder_token_ids,
            accelerator,
            args,
            save_path,
            safe_serialization=not args.no_safe_serialization,
        )

    card_args={}
    if args.customization_model=="DreamBooth":
        card_args["train_text_encoder"]=args.train_text_encoder
        card_args["pipeline"]=pipeline

    if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
                **card_args,
            )
            api = HfApi(token=args.hub_token)
            api.upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )


accelerator.end_training()

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)