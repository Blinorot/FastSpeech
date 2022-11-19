import numpy as np
import torch
from src.base import BaseModel
from src.model.blocks import FFTBlock, LengthRegulator, VariancePredictor
from torch import nn


class Encoder(nn.Module):
    def __init__(self, 
        max_seq_len, 
        encoder_n_layer,
        vocab_size,
        encoder_dim,
        encoder_head,
        encoder_conv1d_filter_size,
        fft_conv1d_kernel,
        fft_conv1d_padding,
        PAD,
        dropout
        ):
        super(Encoder, self).__init__()
        
        len_max_seq= max_seq_len
        n_position = len_max_seq + 1
        n_layers = encoder_n_layer

        self.src_word_emb = nn.Embedding(
            vocab_size,
            encoder_dim,
            padding_idx=PAD
        )

        self.position_enc = nn.Embedding(
            n_position,
            encoder_dim,
            padding_idx=PAD
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            encoder_dim,
            encoder_conv1d_filter_size,
            encoder_head,
            encoder_dim // encoder_head,
            encoder_dim // encoder_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout
        ) for _ in range(n_layers)])
        self.PAD = PAD

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, PAD=self.PAD)
        non_pad_mask = get_non_pad_mask(src_seq, self.PAD)
        
        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        

        return enc_output, non_pad_mask

class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, max_seq_len, 
        decoder_n_layer,
        decoder_dim,
        decoder_head,
        decoder_conv1d_filter_size,
        fft_conv1d_kernel,
        fft_conv1d_padding,
        PAD,
        dropout):

        super(Decoder, self).__init__()

        len_max_seq=max_seq_len
        n_position = len_max_seq + 1
        n_layers = decoder_n_layer

        self.position_enc = nn.Embedding(
            n_position,
            decoder_dim,
            padding_idx=PAD,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            decoder_dim,
            decoder_conv1d_filter_size,
            decoder_head,
            decoder_dim // decoder_head,
            decoder_dim // decoder_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout
        ) for _ in range(n_layers)])

        self.PAD = PAD

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, PAD=self.PAD)
        non_pad_mask = get_non_pad_mask(enc_pos, self.PAD)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output


class FastSpeechModel(BaseModel):
    """ FastSpeech """

    def __init__(self, max_seq_len,
        encoder_n_layer,
        decoder_n_layer,
        vocab_size,
        encoder_dim,
        encoder_head,
        encoder_conv1d_filter_size,
        decoder_dim,
        decoder_head,
        decoder_conv1d_filter_size,
        fft_conv1d_kernel,
        fft_conv1d_padding,
        duration_predictor_filter_size,
        duration_predictor_kernel_size,
        energy_predictor_filter_size,
        energy_predictor_kernel_size,
        min_energy,
        max_energy,
        num_bins,
        num_mels,
        PAD,
        dropout
        ):
        super(FastSpeechModel, self).__init__()

        self.encoder = Encoder(max_seq_len, 
            encoder_n_layer,
            vocab_size,
            encoder_dim,
            encoder_head,
            encoder_conv1d_filter_size,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            PAD,
            dropout
        )
        self.length_regulator = LengthRegulator(encoder_dim, 
            duration_predictor_filter_size,
            duration_predictor_kernel_size,
            dropout
        )
        self.decoder = Decoder(max_seq_len, 
            decoder_n_layer,
            decoder_dim,
            decoder_head,
            decoder_conv1d_filter_size,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            PAD,
            dropout
        )

        self.mel_linear = nn.Linear(decoder_dim, num_mels)

        # we estimane energy_target + 1, so we add +1 to bounds
        energy_space = torch.linspace(np.log(min_energy + 1), np.log(max_energy + 2), num_bins)
        self.register_buffer('energy_space', energy_space)

        self.energy_emb = nn.Embedding(num_bins, encoder_dim)
        self.energy_predictor = VariancePredictor(encoder_dim,
            energy_predictor_filter_size,
            energy_predictor_kernel_size,
            dropout
        )

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def get_energy(self, x, energy_target=None, beta=1.0):
        energy_predictor_output = self.energy_predictor(x)
        
        # we estimate energy_target + 1 to avoid nans
        if energy_target is not None:
            buckets = torch.bucketize(torch.log(energy_target + 1), self.energy_space)
        else:
            buckets = torch.bucketize(energy_predictor_output + np.log(beta), self.energy_space)
        emb = self.energy_emb(buckets)
        return emb, energy_predictor_output

    def forward(self, src_seq, src_pos, mel_pos=None,
                mel_max_length=None, length_target=None, 
                energy_target=None, alpha=1.0, beta=1.0,
                **kwargs):
        x, non_pad_mask = self.encoder(src_seq, src_pos)
        if self.training:
            output, duration_predictor_output = self.length_regulator(x, alpha, 
                                                            length_target, mel_max_length)

            energy_emb, energy_predictor_output = self.get_energy(output, energy_target=energy_target, beta=beta)

            output = self.decoder(output + energy_emb, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)

            return {"mel_output": output, 
                    "duration_predictor_output": duration_predictor_output,
                    "energy_predictor_output": energy_predictor_output}
        else:
            output, mel_pos = self.length_regulator(x, alpha)
            energy_emb, _ = self.get_energy(output, beta=beta)
            output = self.decoder(output + energy_emb, mel_pos)
            output = self.mel_linear(output)
            return {"mel_output": output}


def get_non_pad_mask(seq, PAD):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q, PAD):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask
