import torch

# From https://github.com/zjysteven/StreamDiffusion
class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(self, unet, controlnet) -> None:
        super().__init__()
        self.unet = unet
        self.config = unet.config
        self.controlnet = controlnet

    def forward(self, sample, timestep, encoder_hidden_states, image):
        # hard-coded since it is not clear how to integrate this argument into tensorrt
        conditioning_scale = 0.5

        down_samples, mid_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=image,
            guess_mode=False,
            return_dict=False,
        )

        down_block_res_samples = [
            down_sample * conditioning_scale
            for down_sample in down_samples
        ]
        mid_block_res_sample = conditioning_scale * mid_sample

        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )
        return noise_pred
