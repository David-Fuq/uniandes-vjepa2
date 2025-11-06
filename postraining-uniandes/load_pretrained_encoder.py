from postraining_uniandes.logger_helper import get_logger
import torch

logger = get_logger(__name__, force=True)

def load_pretrained_weights(
    r_path,
    encoder=None,
    target_encoder=None,
    context_encoder_key="encoder", #For our particular case, this is model.
    target_encoder_key="target_encoder", #For our particular case, this is model.
    ):
    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = torch.load(r_path, map_location=torch.device("cpu"))

    epoch = 0

    # Loading encoder
    pretrained_dict = checkpoint[context_encoder_key]
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    # Loading target_encoder
    if target_encoder is not None:
        pretrained_dict = checkpoint[target_encoder_key]
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")
    
    del checkpoint
    
    return encoder, target_encoder