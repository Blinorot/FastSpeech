import torch
import torch.nn.functional as F
from src.model.blocks.variance_predictor import VariancePredictor
from torch import nn


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, 
        encoder_dim, 
        duration_predictor_filter_size,
        duration_predictor_kernel_size,
        dropout
        ):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = VariancePredictor(encoder_dim, 
            duration_predictor_filter_size,
            duration_predictor_kernel_size,
            dropout
        )

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            # we remove 1 from exp because we estimate (target + 1), also we ensure that min is 0
            duration_predictor_output = (((torch.exp(duration_predictor_output) - 1) * alpha) + 0.5).int()
            duration_predictor_output[duration_predictor_output < 0] = 0

            output = self.LR(x, duration_predictor_output)
            mel_pos = torch.stack(
                [torch.Tensor([i+1  for i in range(output.size(1))])]
            ).long().to(x.device)
            return output, mel_pos
