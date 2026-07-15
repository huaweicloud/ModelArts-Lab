from x_diffusers import InferenceManager, init_env, parse_args

if __name__ == "__main__":
    args = parse_args()
    init_env()
    inference_manager = InferenceManager()
    inference_manager.infer(args)
