import torch 
import torch.nn as nn
from timm.models.vision_transformer import Block
from Data2Seq import Data2Seq


video_tokenier = Data2Seq(modality='video',dim=768)
audio_tokenier = Data2Seq(modality='audio',dim=768)
time_series_tokenier = Data2Seq(modality='time-series',dim=768)

features = torch.concat([video_tokenizer(video),audio_tokenizer(audio), time_series_tokenizer(time_data)],dim=1)
# For base-scale encoder:
ckpt = torch.load("/root/genni/Meta-Transformer_base_patch16_encoder.pth")
encoder = nn.Sequential(*[
            Block(
                dim=768,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(12)])
encoder.load_state_dict(ckpt,strict=True)






encoded_features = encoder(features)





# # For large-scale encoder:
# ckpt = torch.load("Meta-Transformer_large_patch14_encoder.pth")
# encoder = nn.Sequential(*[
#             Block(
#                 dim=1024,
#                 num_heads=16,
#                 mlp_ratio=4.,
#                 qkv_bias=True,
#                 norm_layer=nn.LayerNorm,
#                 act_layer=nn.GELU
#             )
#             for i in range(24)])
# encoder.load_state_dict(ckpt,strict=True)

encoded_features = encoder(features)