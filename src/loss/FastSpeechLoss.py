from cmath import log

import torch
from torch import nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, mel_output, duration_predictor_output, energy_predictor_output,
                mel_target, length_target, energy_target, **kwargs):
        mel_loss = self.mse_loss(mel_output, mel_target)

        # we add 1 before log to avoid nan from zeros
        duration_predictor_loss = self.mse_loss(duration_predictor_output,
                                                torch.log((length_target + 1).float())) # log(duration)

        energy_predictor_loss = self.mse_loss(energy_predictor_output,
                                              torch.log(energy_target + 1))

        return mel_loss, duration_predictor_loss, energy_predictor_loss
