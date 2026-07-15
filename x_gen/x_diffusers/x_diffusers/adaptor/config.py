import argparse


def parse_args(namespace=None, mode="infer"):
    parser = argparse.ArgumentParser(description="AscendX-Video inference script")

    parser = add_network_args(parser)
    parser = add_inference_args(parser)
    parser = add_wan_vace_args(parser)
    parser = add_abilities_args(parser)
    if mode == "benchmark":
        parser = add_benchmark_args(parser)

    args = parser.parse_args(namespace=namespace)

    return args


def add_network_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="AscendX-Video network args")
    group.add_argument("--model", type=str, default="Wan2.2-T2V-A14B", help="Model for inference.")
    group.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Wan-AI/Wan2.2-T2V-14B-Diffusers",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    group.add_argument(
        "--task_type", type=str, default="t2v", choices=["t2v", "i2v", "t2i"], help="Whether to open i2v mode."
    )
    group.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    return parser


def add_inference_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Inference args")
    group.add_argument(
        "--prompt",
        type=str,
        default="A cat and a dog baking a cake together",
        help="The text prompt for video generation.",
    )
    group.add_argument(
        "--negative_prompt",
        type=str,
        default="Briqht tones, overexposed, static",
        help="The negative text prompt for video generation.",
    )
    group.add_argument("--height", type=int, default=480, help="The resolution for the generated video height.")
    group.add_argument("--width", type=int, default=832, help="The resolution for the generated video width.")
    group.add_argument("--frames", type=int, default=81, help="The length of the generated video.")
    group.add_argument("--num_inference_steps", type=int, default=40, help="The number of steps for sampling.")
    group.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="Guidance scale as defined in [Classifier-Free Diffusion Guidance]Classifier-free guidance is enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`",  # noqa: E501
    )
    group.add_argument("--guidance_scale", type=float, default=None, help="The scale factor for the guidance term.")
    group.add_argument(
        "--guidance_scale_2", type=float, default=None, help="The scale factor for the guidance_scale_2 term."
    )
    group.add_argument("--flow_shift", type=float, default=3.0, help="Shift factor for flow matching schedulers.")
    group.add_argument(
        "--seed", type=int, default=42, help="The random seed for generating video, if None, we init a random seed."
    )
    group.add_argument("--save_fps", type=int, default=16, help="The frame of per sec for export videos.")
    group.add_argument("--save_path", type=str, default="./output.mp4", help="Path to save the generated video.")
    group.add_argument("--i2v_image_path", type=str, default="", help="The reference image for video generation")
    group.add_argument(
        "--i2v_image_preprocess",
        type=str,
        default="wan",
        choices=["wan", "raw"],
        help="Whether to resize the original image and change the video resolution accordingly.",
    )
    group.add_argument(
        "--batch_test_input_dir", type=str, default="", help="Directory containing input data for batch testing"
    )
    group.add_argument(
        "--batch_test_mode",
        type=str,
        default="normal",
        choices=["normal", "speed", "accuracy"],
        help="Test mode: normal (regular batch), speed (speed benchmarking), accuracy (accuracy evaluation)",
    )
    group.add_argument(
        "--batch_test_resolutions",
        type=str,
        default="",
        help="List of resolutions for batch testing, format like '832x480,1280x720'",
    )
    group.add_argument(
        "--batch_test_frames",
        type=str,
        default="",
        help="List of frame counts for batch testing, format like '81,49,17'",
    )
    group.add_argument(
        "--batch_test_targets",
        type=str,
        default="",
        help="List of acceleration targets for batch testing, format like 'none,encoder,decoder,encoder_and_decoder'",
    )
    group.add_argument(
        "--batch_test_sample_size",
        type=int,
        default=0,
        help="Number of videos randomly sampled for testing; 0 means testing the full dataset",
    )
    group.add_argument(
        "--batch_test_report_dir", type=str, default="", help="Directory where the test reports will be saved"
    )
    return parser


def add_wan_vace_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Wan2.1 VACE args")
    group.add_argument(
        "--vace_task",
        type=str,
        default="",
        choices=["t2v", "i2v", "v2lf", "flf2v", "random2v", "inpaint", "outpaint", "openpose", "iwri"],
        help="The vace task to run.",
    )
    group.add_argument("--image_path", type=str, default="", help="The image for vace video generation")
    group.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="The path of the videos to be processed, separated by commas if there are multiple.",
    )
    group.add_argument(
        "--first_frame_path", type=str, default="", help="The first reference frame for video generation."
    )
    group.add_argument("--last_frame_path", type=str, default="", help="The last reference frame for video generation.")
    group.add_argument("--image_path_list", nargs="+", default=None, help="The image path list for the random2v task.")
    group.add_argument(
        "--frame_indices",
        type=int,
        nargs="+",
        default=[0, 45, 70],
        help="Random indices of some frames for the random2v task.",
    )
    group.add_argument(
        "--directions", type=str, nargs="+", default=["left", "right"], help="The directions for the outpaint task."
    )
    group.add_argument("--expand_ratio", type=float, default=0.25, help="The expand ratio for the outpaint task.")
    return parser


def add_benchmark_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Benchmarking args")
    group.add_argument("--config", type=str, required=True)
    return parser


def add_abilities_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Inference abilities")

    group.add_argument("--sp", type=int, default=1, help="Sequence parallel number.")
    group.add_argument("--ulysses_degree", type=int, default=None, help="Ulysses sequence parallel number.")
    group.add_argument("--ring_degree", type=int, default=None, help="Ring sequence parallel number.")
    group.add_argument("--phaa_num", type=int, default=0, help="PHAA number.")
    group.add_argument(
        "--turbo_mode",
        type=str,
        default="default",
        choices=["default", "faiz", "next_faiz"],
        help="Use different turbo mode inference.",
    )
    group.add_argument(
        "--vae_lightning",
        type=str,
        default="none",
        choices=["none", "decoder", "encoder", "encoder_and_decoder"],
        help="Light VAE inference. Specify target: none/decoder/encoder/encoder_and_decoder",
    )
    group.add_argument(
        "--vae_output_dir",
        type=str,
        default="",
        help="Codec output result directory name. If not specified, the results will not be saved. Currently this only applies to the encoder.",  # noqa: E501
    )
    group.add_argument(
        "--vae_pad_mode",
        type=str,
        default="all_sides_valid",
        choices=["none", "tail_only", "all_sides", "all_sides_valid"],
        help="Padding strategy when VAE parallel acceleration is enabled.",
    )
    group.add_argument(
        "--vae_pad_content",
        type=str,
        default="constant",
        choices=["constant", "reflect", "replicate"],
        help="The padding value when VAE parallel acceleration is enabled and `vae_pad_mode = all_sides`.",
    )
    group.add_argument(
        "--vae_pad_latent_size",
        type=int,
        default=8,
        help="Latent space padding size when VAE parallel acceleration is enabled.",
    )
    group.add_argument(
        "--vae_use_blend",
        action="store_true",
        default=False,
        help="Whether to perform linear blending on overlapping boundaries when VAE parallel acceleration is enabled.",
    )
    group.add_argument(
        "--enable_vae_time_count",
        action="store_true",
        default=False,
        help="Whether to run repeatedly and calculate the average runtime of the VAE Encoder/Decoder. Enabling this option will make the end-to-end generation time inaccurate.",  # noqa: E501
    )

    group.add_argument(
        "--frame_interpolation", action="store_true", default=False, help="Frame interpolation without using sr model."
    )
    group.add_argument(
        "--frame_interpolation_sr", action="store_true", default=False, help="Frame interpolation using sr model."
    )
    group.add_argument(
        "--frame_model_path", type=str, default="/home/IFRNet_S_Vimeo90K.pth", help="Frame interpolation Model Path."
    )
    group.add_argument("--x", action="store_true", default=False, help="Use x model")
    group.add_argument("--x_model_path", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers_x", help="Path to x model")
    group.add_argument(
        "--x_model_path_2", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers_x_2", help="Path to x model"
    )
    group.add_argument("--joint", action="store_true", default=False, help="Use joint inference")
    group.add_argument("--ten_second", action="store_true", default=False, help="Generate ten second video")
    group.add_argument(
        "--ten_second_model_id_t2v", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers", help="Wan2.2_t2v_path"
    )
    group.add_argument(
        "--ten_second_model_path", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers_x", help="Wan2.2_t2v_distill_path"
    )
    group.add_argument(
        "--ten_second_model_path_2",
        type=str,
        default="Wan-AI/Wan2.1-T2V-14B-Diffusers_x_2",
        help="Wan2.2_t2v_distill_path_2",
    )
    group.add_argument(
        "--pusa_lora", type=str, default="Wan-AI/pytorch_lora_weights_high.safetensors", help="pusa_lora"
    )
    group.add_argument(
        "--pusa_lora2", type=str, default="Wan-AI/pytorch_lora_weights_low.safetensors", help="pusa_lora2"
    )
    group.add_argument(
        "--joint_model_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers_joint", help="Path to joint model"
    )
    parser.add_argument(
        "--fsdp",
        type=str,
        nargs="?",
        const="all",
        choices=["all", "text_encoder", "transformer"],
        help="Enable Fully Sharded Data Parallel (FSDP). "
        "The default is the all model, with options for text_encoder and transformer models",
    )
    group.add_argument("--atten_a8w8", action="store_true", default=False, help="Enable attention quantization (A8W8)")
    group.add_argument("--atten_laser", action="store_true", default=False, help="Enable laser attention")
    group.add_argument("--atten_rainfusion", action="store_true", default=False, help="Enable rainfusion attention")
    group.add_argument(
        "--rainfusion_ratio",
        default=0.375,
        type=float,
        help="The ratio of skip timesteps of rainfusion attention."
        "If num_inference_steps is 40 and rainfusion_ratio is 0.375, the skip_timesteps is 15.",
    )
    group.add_argument("--atten_ada_sparse", action="store_true", default=False, help="Enable ada sparse attention")
    group.add_argument(
        "--ada_sparsity",
        default=0.7,
        type=float,
        help="Proportion of invalid/uncomputed elements in the attention matrix",
    )
    group.add_argument("--matmul_a8w8", action="store_true", default=False, help="Enable matmul quantization (A8W8)")
    group.add_argument("--matmul_a4w4", action="store_true", default=False, help="Enable matmul quantization (A4W4)")
    group.add_argument("--conv3d_w8a8", action="store_true", default=False, help="Enable conv3d quantization (A8W8)")
    group.add_argument("--rope_fused", action="store_true", default=False, help="Enable fused rope operator")
    group.add_argument("--lora_path_list", nargs="+", default=None, help="List of path to load lora weights.")
    group.add_argument("--lora_scale_weight_list", nargs="+", default=None, help="List of lora scale weights.")
    group.add_argument(
        "--lora_transformer_list",
        nargs="+",
        default=None,
        type=int,
        choices=[0, 1, 2],
        help="The list of which transformer need load lora, 0 means all transformer, 1 means transformer_1, 2 means transformer_2.",  # noqa: E501
    )
    group.add_argument("--fuse_lora", action="store_true", default=False, help="Fuse lora weights.")
    group.add_argument("--server", action="store_true", default=False, help="Use API server")
    group.add_argument(
        "--inf_vram_blocks_num",
        type=int,
        default=0,
        help="if > 0, enable Inifinity VRAM and sets the number of blocks to keep in VRAM. 0 disables offloading",
    )
    group.add_argument("--seedvr2_model_dir", type=str, default="../weights/SeedVR2")
    group.add_argument("--seedvr2_model_name", type=str, default="seedvr2_ema_7b_fp16.safetensors")
    group.add_argument("--adopt_sr", action="store_true", default=False, help="if adopt super-resolution")
    group.add_argument(
        "--resolution", type=int, default=1080, choices=[1080, 720], help="generate video with specified resolution."
    )
    group.add_argument(
        "--ada_brighten",
        action="store_true",
        default=False,
        help="Adaptively adjusts and restores input image brightness for video generation",
    )
    group.add_argument("--save_memory", action="store_true", default=False, help="apply save memory for offloadmanager")
    group.add_argument(
        "--cfg_parallel_size",
        default=1,
        type=int,
        choices=[1, 2],
        help="Number of cfg parallel, 1 means not use cfg parallel",
    )
    group.add_argument(
        "--cache_dit",
        action="store_true",
        default=False,
        help="use cache_dit",
    )
    return parser
