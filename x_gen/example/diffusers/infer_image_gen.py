from x_diffusers import ImageInferenceManager, init_cfg_env, parse_args

if __name__ == "__main__":
    args = parse_args()

    if args.cfg_parallel_size == 2:
        init_cfg_env(args.cfg_parallel_size)

    inference_manager = ImageInferenceManager()
    inference_manager.infer(args)
