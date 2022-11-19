import torch
from torch import nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel_output, duration_predictor_output, mel_target, length_target, **kwargs):
        mel_loss = self.mse_loss(mel_output, mel_target)

        duration_predictor_loss = self.l1_loss(duration_predictor_output,
                                               torch.log(length_target.float())) # log(duration)

        return mel_loss, duration_predictor_loss
