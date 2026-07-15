class InferInfo:
    model = ""
    task_type = "t2v"
    width = 832
    height = 480
    save_width = None
    save_height = None
    frames = 121
    fps = 16
    save_path = "./output.mp4"
    vae_lightning = "decoder"
    vae_output_dir = ""
    vae_pad_mode = "all_sides_valid"
    vae_pad_content = "constant"
    vae_pad_latent_size = 8
    vae_use_blend = False
    enable_vae_time_count = False
    batch_test_mode = "normal"
    vae_last_encode_time = "/"
    vae_last_decode_time = "/"
    use_matmul_a4w4 = False
    use_matmul_a8w8 = False
    # 卡数 -> 逻辑分块卡数 的映射表
    # 设置原因：16卡时接缝明显，退化为8卡切分规格
    tile_world_size_map: dict = {16: 8}

    def update_info(self, args):
        self.model = args.model
        self.task_type = args.task_type
        self.width = args.width
        self.height = args.height
        self.frames = args.frames
        self.fps = args.save_fps
        self.save_path = args.save_path
        self.ada_brighten = args.ada_brighten
        self.frame_interpolation = args.frame_interpolation
        self.frame_model_path = args.frame_model_path
        self.ten_second = args.ten_second
        self.i2v_image_path = args.i2v_image_path
        self.vae_lightning = args.vae_lightning
        self.vae_output_dir = args.vae_output_dir
        self.vae_pad_mode = args.vae_pad_mode
        self.vae_pad_content = args.vae_pad_content
        self.vae_pad_latent_size = args.vae_pad_latent_size
        self.vae_use_blend = args.vae_use_blend
        self.enable_vae_time_count = args.enable_vae_time_count
        self.batch_test_mode = args.batch_test_mode
        self.vae_last_encode_time = "/"
        self.vae_last_decode_time = "/"
        self.use_matmul_a4w4 = args.matmul_a4w4
        self.use_matmul_a8w8 = args.matmul_a8w8

        if self.model.startswith("HunyuanVideo") and (self.frames - 1) % 4 != 0:
            self.frames = 4 * ((self.frames - 1) // 4) + 1
            print(f"num_frames must satisfy 4*k+1; adjusted to {self.frames}")

    def update_shape(self, width, height):
        self.width = width
        self.height = height

    def update_adabrighten(self, ada_brighten):
        self.ada_brighten = ada_brighten

    def get_tile_world_size(self, world_size: int) -> int:
        return self.tile_world_size_map.get(world_size, world_size)

    def __str__(self):
        return f"Infer Video Info,model={self.model},task_type:{self.task_type},width={self.width},height={self.height},frames={self.frames},fps={self.fps},save_path={self.save_path}"  # noqa: E501


infer_info = InferInfo()
