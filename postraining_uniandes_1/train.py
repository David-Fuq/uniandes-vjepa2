'''
Handles the training loop for the model.
Pipeline:
1. Load dataset
2. Preprocess data vida droid_dataset_handler
3. Encoder
4. Predictor
5. Loss Calculation
6. Repeat!
'''
import sys
import torch
import numpy as np
from postraining_uniandes.logger_helper import get_logger
from postraining_uniandes.droid_dataset_handler import init_data
from postraining_uniandes.transforms import make_transforms

logger = get_logger(__name__, force=True)
def train(args):
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    r_file = cfgs_meta.get("resume_checkpoint", None)
    p_file = cfgs_meta.get("pretrain_checkpoint", None)
    load_predictor = cfgs_meta.get("load_predictor", False)
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")
    load_encoder = cfgs_meta.get("load_encoder", True)
    #seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
    uniform_power = cfgs_model.get("uniform_power", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    use_extrinsics = cfgs_model.get("use_extrinsics", False)

    # -- DATA
    cfgs_data = args.get("data")
    datasets = cfgs_data.get("datasets", [])
    dataset_path = datasets[0]
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    camera_frame = cfgs_data.get("camera_frame", False)
    camera_views = cfgs_data.get("camera_views", ["left_mp4_path"])
    stereo_view = cfgs_data.get("stereo_view", False)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    horizontal_flip = cfgs_data_aug.get("horizontal_flip", False)
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    normalize_reps = cfgs_loss.get("normalize_reps")
    auto_steps = min(cfgs_loss.get("auto_steps", 1), max_num_frames)
    # --
    tokens_per_frame = int((crop_size // patch_size) ** 2)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    anneal = cfgs_opt.get("anneal")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    enc_lr_scale = cfgs_opt.get("enc_lr_scale", 1.0)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)

    print("Training configuration loaded successfully.")

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[0.75, 1.35],
        random_resize_scale=[1.777, 1.777],
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )
    logger.info("Initializing data loader...")
    data_loader, sampler = init_data(
        data_path=dataset_path,
        batch_size=batch_size,
        frames_per_clip=max_num_frames,
        tubelet_size=1,
        fps=fps,
        camera_views=camera_views,
        camera_frame=camera_frame,
        transform=transform,
        collator=torch.utils.data.default_collate,
        num_workers=num_workers,
        world_size=1,
        rank=0,
        pin_mem=False,
        persistent_workers=False,
    )
    logger.info(f"Data loader initialized. Dataset length: {len(data_loader)}")

    NUM_BATCHES = 3
    loader_iter = iter(data_loader)
    for batch_idx in range(NUM_BATCHES):
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing batch {batch_idx + 1}/{NUM_BATCHES}")
        logger.info('='*80)
        
        try:
            sample = next(loader_iter)
            
            # Unpack sample (no extrinsics!)
            buffer = sample[0]   # [B, C, T, H, W]
            actions = sample[1]  # [B, T-1, 7]
            states = sample[2]   # [B, T, 7]
            indices = sample[3]  # [B, T]
            
            logger.info(f"\n[SHAPES]")
            logger.info(f"  buffer.shape:  {buffer.shape}")
            logger.info(f"  actions.shape: {actions.shape}")
            logger.info(f"  states.shape:  {states.shape}")
            logger.info(f"  indices.shape: {indices.shape}")
            
            logger.info(f"\n[DATA TYPES]")
            logger.info(f"  buffer.dtype:  {buffer.dtype}")
            logger.info(f"  actions.dtype: {actions.dtype}")
            logger.info(f"  states.dtype:  {states.dtype}")
            logger.info(f"  indices.dtype: {indices.dtype}")
            
            logger.info(f"\n[VALUE RANGES]")
            logger.info(f"  buffer:  min={buffer.min():.4f}, max={buffer.max():.4f}, mean={buffer.mean():.4f}")
            logger.info(f"  actions: min={actions.min():.4f}, max={actions.max():.4f}, mean={actions.mean():.4f}")
            logger.info(f"  states:  min={states.min():.4f}, max={states.max():.4f}, mean={states.mean():.4f}")
            
            logger.info(f"\n[FIRST SAMPLE IN BATCH]")
            logger.info(f"  First 3 pixels of first frame (RGB):")
            logger.info(f"    {buffer[0, :, 0, 0, :3]}")
            logger.info(f"  First action vector (xyz, rotation, gripper):")
            logger.info(f"    {actions[0, 0]}")
            logger.info(f"  First state vector:")
            logger.info(f"    {states[0, 0]}")
            logger.info(f"  Frame indices:")
            logger.info(f"    {indices[0]}")
            
            logger.info(f"\n[VALIDATION CHECKS]")
            # Check for NaN/Inf
            has_nan_buffer = torch.isnan(buffer).any().item()
            has_nan_actions = torch.isnan(actions).any().item()
            has_nan_states = torch.isnan(states).any().item()
            
            logger.info(f"  buffer has NaN:  {has_nan_buffer}")
            logger.info(f"  actions has NaN: {has_nan_actions}")
            logger.info(f"  states has NaN:  {has_nan_states}")
            
            # Check expected dimensions
            expected_buffer_shape = (batch_size, 3, max_num_frames, crop_size, crop_size)
            expected_actions_shape = (batch_size, max_num_frames - 1, 7)
            expected_states_shape = (batch_size, max_num_frames, 7)

            assert buffer.shape == expected_buffer_shape, f"Buffer shape mismatch! Expected {expected_buffer_shape}, got {buffer.shape}"
            assert actions.shape == expected_actions_shape, f"Actions shape mismatch! Expected {expected_actions_shape}, got {actions.shape}"
            assert states.shape == expected_states_shape, f"States shape mismatch! Expected {expected_states_shape}, got {states.shape}"
            
            logger.info(f"All shapes correct!")
            
            # Verify action computation (actions should be state differences)
            # For first sample, check if action[0] ≈ state[1] - state[0]
            computed_xyz_diff = states[0, 1, :3] - states[0, 0, :3]
            actual_xyz_diff = actions[0, 0, :3]
            xyz_match = torch.allclose(computed_xyz_diff, actual_xyz_diff, atol=1e-5)
            logger.info(f"  Action XYZ matches state difference: {xyz_match}")
            if not xyz_match:
                logger.warning(f"    Expected: {computed_xyz_diff}")
                logger.warning(f"    Got:      {actual_xyz_diff}")
            
            logger.info("=" * 80)
            logger.info("DEBUG: First batch data shapes and samples")
            logger.info(f"clips.shape: {buffer.shape}")
            logger.info(f"actions.shape: {actions.shape}")
            logger.info(f"states.shape: {states.shape}")
            
            # Log first sample in batch
            logger.info("\nFirst sample in batch:")
            logger.info(f"clips[0, :, 0, 0, :5]: {buffer[0, :, 0, 0, :5]}")  # First 5 pixels of first frame
            logger.info(f"actions[0, :3]: \n{actions[0, :3]}")  # First 3 action vectors
            logger.info(f"states[0, :3]: \n{states[0, :3]}")  # First 3 state vectors
            
            # Check for NaN or Inf
            logger.info(f"\nData validation:")
            logger.info(f"clips has NaN: {torch.isnan(buffer).any().item()}")
            logger.info(f"actions has NaN: {torch.isnan(actions).any().item()}")
            logger.info(f"states has NaN: {torch.isnan(states).any().item()}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error loading batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    logger.info(f"\n{'='*80}")
    logger.info("Data loading test completed!")
    logger.info('='*80)

