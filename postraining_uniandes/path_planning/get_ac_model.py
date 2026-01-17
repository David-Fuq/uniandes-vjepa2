import torch

encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")


torch.save(encoder, "encoder_full_vjepa2_original.pt")
torch.save(predictor, "predictor_full_vjepa2_original.pt")