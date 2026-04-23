import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import clones, is_list_or_tuple
from torchvision.ops import RoIAlign


class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)


class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, c_in, s, h_in, w_in]
            Out x: [n, c_out, s, h_out, w_out]
        """
        n, c, s, h, w = x.size()
        x = self.forward_block(x.transpose(
            1, 2).reshape(-1, c, h, w), *args, **kwargs)
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()


class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **options)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(dim, curr_start, curr_seqL)
            rets.append(self.pooling_func(narrowed_seq, **options))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class TemporalSpectralAdapter(nn.Module):
    def __init__(
        self,
        channels,
        groups=1,
        bottleneck_ratio=0.25,
        fusion='residual',
        spectral_mode='amplitude',
        local_kernel_sizes=(3, 5),
        init_scale=0.1,
        use_spatial_gate=True,
        part_bins=4,
        low_freq_ratio=0.25,
        dynamic_only=True,
        branch_sources=('identity', 'local', 'dynamic', 'dynamic_gap', 'part'),
        **kwargs,
    ):
        super().__init__()
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        if fusion not in ['residual', 'gated_residual']:
            raise ValueError("fusion must be 'residual' or 'gated_residual'")
        if spectral_mode not in ['amplitude', 'amplitude_phase']:
            raise ValueError("spectral_mode must be 'amplitude' or 'amplitude_phase'")
        valid_sources = {'identity', 'local', 'dynamic', 'dynamic_gap', 'part'}
        if any(source not in valid_sources for source in branch_sources):
            raise ValueError("Unsupported branch source in {}".format(branch_sources))
        if 'part' in branch_sources and int(part_bins) <= 0:
            raise ValueError("part source requires part_bins > 0")
        if kwargs:
            # Keep config loading backward compatible while making the active module branch-only.
            ignored = ', '.join(sorted(kwargs.keys()))
            print("TemporalSpectralAdapter ignores unused config keys: {}".format(ignored))

        self.channels = channels
        self.groups = groups
        self.group_channels = channels // groups
        self.fusion = fusion
        self.spectral_mode = spectral_mode
        self.use_spatial_gate = use_spatial_gate
        self.part_bins = int(part_bins)
        self.low_freq_ratio = float(low_freq_ratio)
        self.dynamic_only = dynamic_only
        self.branch_sources = list(branch_sources)

        hidden_dim = max(int(self.group_channels * bottleneck_ratio), 1)
        self.norm = nn.LayerNorm(channels)
        kernel_sizes = local_kernel_sizes if is_list_or_tuple(local_kernel_sizes) else [local_kernel_sizes]
        self.local_branches = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size=int(kernel_size),
                padding=int(kernel_size) // 2,
                groups=groups,
                bias=False,
            )
            for kernel_size in kernel_sizes
        ])
        self.local_fuse = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.amp_mlp = nn.Sequential(
            nn.Linear(self.group_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.group_channels),
        )
        self.low_amp_mlp = nn.Sequential(
            nn.Linear(self.group_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.group_channels),
        )
        if spectral_mode == 'amplitude_phase':
            self.phase_mlp = nn.Sequential(
                nn.Linear(self.group_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.group_channels),
            )
            self.low_phase_mlp = nn.Sequential(
                nn.Linear(self.group_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.group_channels),
            )
        else:
            self.phase_mlp = None
            self.low_phase_mlp = None
        self.spectral_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.identity_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        if self.part_bins > 0:
            self.part_norm = nn.LayerNorm(channels)
            self.part_amp_mlp = nn.Sequential(
                nn.Linear(self.group_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.group_channels),
            )
            self.part_proj = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            )
        else:
            self.part_norm = None
            self.part_amp_mlp = None
            self.part_proj = None
        self.branch_summary = nn.Sequential(
            nn.Linear(channels, max(channels // 4, 32)),
            nn.GELU(),
            nn.Linear(max(channels // 4, 32), len(self.branch_sources)),
        )
        self.branch_source_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.branch_out_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.res_scale = nn.Parameter(torch.full((1, channels, 1), float(init_scale)))
        self.fusion_gate = nn.Conv1d(channels, channels, kernel_size=1, bias=True) \
            if fusion == 'gated_residual' else None
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        ) if use_spatial_gate else None
        self.part_spatial_gate = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.Sigmoid(),
        ) if self.part_bins > 0 else None

    def _modulate_spectrum(self, amp, phase, amp_mlp, phase_mlp=None, gate_scale=0.25):
        n, freq_bins, c = amp.shape
        amp_grouped = amp.view(n, freq_bins, self.groups, self.group_channels)
        amp_gate = gate_scale * torch.tanh(amp_mlp(amp_grouped)).view(n, freq_bins, c)
        amp = amp * (1.0 + amp_gate)
        if phase_mlp is not None:
            phase_grouped = phase.view(n, freq_bins, self.groups, self.group_channels)
            phase = phase + gate_scale * torch.tanh(phase_mlp(phase_grouped)).view(n, freq_bins, c)
        return amp, phase

    def _split_reconstruct(self, desc_tokens, amp_mlp_high, amp_mlp_low, phase_mlp_high=None, phase_mlp_low=None):
        n, s, c = desc_tokens.shape
        freq = torch.fft.rfft(desc_tokens, dim=1, norm='ortho')
        amp = torch.abs(freq)
        phase = torch.angle(freq)
        freq_bins = amp.size(1)
        low_bins = min(max(int(round(freq_bins * self.low_freq_ratio)), 1), freq_bins)

        low_amp = amp[:, :low_bins]
        low_phase = phase[:, :low_bins]
        low_amp, low_phase = self._modulate_spectrum(
            low_amp, low_phase, amp_mlp_low, phase_mlp_low, gate_scale=0.15)

        high_amp = amp[:, low_bins:]
        high_phase = phase[:, low_bins:]
        if high_amp.numel() > 0:
            high_amp, high_phase = self._modulate_spectrum(
                high_amp, high_phase, amp_mlp_high, phase_mlp_high, gate_scale=0.30)

        identity_freq = torch.zeros_like(freq)
        identity_freq[:, :low_bins] = torch.polar(low_amp, low_phase)
        dynamic_freq = torch.zeros_like(freq)
        if high_amp.numel() > 0:
            dynamic_freq[:, low_bins:] = torch.polar(high_amp, high_phase)
        elif not self.dynamic_only:
            dynamic_freq[:, :low_bins] = identity_freq[:, :low_bins]

        identity = torch.fft.irfft(identity_freq, n=s, dim=1, norm='ortho')
        dynamic = torch.fft.irfft(dynamic_freq, n=s, dim=1, norm='ortho')
        return dynamic, identity

    def _compute_local(self, desc_norm):
        local = 0.0
        for branch in self.local_branches:
            local = local + branch(desc_norm)
        return self.local_fuse(local / len(self.local_branches))

    def _compute_part_context(self, x):
        n, c, s = x.size()[:3]
        part_context = torch.zeros_like(x.mean(dim=(-1, -2)))
        part_spatial_gate = None
        if self.part_bins <= 0:
            return part_context, part_spatial_gate

        part_desc = x.mean(dim=-1)
        part_desc = part_desc.permute(0, 2, 1, 3).reshape(n * s, c, x.size(-2))
        part_desc = F.adaptive_avg_pool1d(part_desc, self.part_bins)
        part_desc = part_desc.view(n, s, c, self.part_bins).permute(0, 3, 1, 2).contiguous()
        part_tokens = self.part_norm(part_desc).view(n * self.part_bins, s, c)
        part_dynamic, _ = self._split_reconstruct(
            part_tokens,
            self.part_amp_mlp,
            self.low_amp_mlp,
            None,
            None,
        )
        part_dynamic = part_dynamic.view(n, self.part_bins, s, c).permute(0, 3, 2, 1).contiguous()
        part_context = self.part_proj(part_dynamic.mean(dim=-1))
        if self.part_spatial_gate is not None:
            part_gate = self.part_spatial_gate(part_dynamic.mean(dim=2))
            part_gate = F.interpolate(part_gate, size=x.size(-2), mode='linear', align_corners=False)
            part_spatial_gate = part_gate.unsqueeze(2).unsqueeze(-1)
        return part_context, part_spatial_gate

    def _compute_delta(self, x, desc_norm, delta_1d, part_spatial_gate=None):
        delta = delta_1d
        if self.fusion_gate is not None:
            delta = delta * torch.sigmoid(self.fusion_gate(desc_norm))
        delta = self.res_scale * delta
        delta = delta.unsqueeze(-1).unsqueeze(-1)
        if self.spatial_gate is not None:
            spatial_gate = self.spatial_gate(x.mean(dim=2)).unsqueeze(2)
            delta = delta * spatial_gate
        if part_spatial_gate is not None:
            delta = delta * part_spatial_gate
        return delta

    def _forward_branch_attention(self, x, desc_norm, global_tokens, local):
        dynamic, identity = self._split_reconstruct(
            global_tokens,
            self.amp_mlp,
            self.low_amp_mlp,
            self.phase_mlp,
            self.low_phase_mlp,
        )
        dynamic = self.spectral_proj(dynamic.transpose(1, 2).contiguous())
        identity = self.identity_proj(identity.transpose(1, 2).contiguous())
        part_context, part_spatial_gate = self._compute_part_context(x)
        source_map = {
            'identity': identity,
            'local': local,
            'dynamic': dynamic,
            'dynamic_gap': self.branch_source_proj(dynamic - identity),
            'part': part_context,
        }
        branch_stack = torch.stack([source_map[name] for name in self.branch_sources], dim=1)
        branch_query = desc_norm.mean(dim=-1)
        branch_weights = torch.softmax(self.branch_summary(branch_query), dim=-1)
        delta = (branch_stack * branch_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        delta = self.branch_out_proj(delta)
        return self._compute_delta(x, desc_norm, delta, part_spatial_gate)

    def forward(self, x):
        """
            x: [n, c, s, h, w]
        """
        n, c, s = x.size()[:3]
        desc = x.mean(dim=(-1, -2))  # [n, c, s]
        desc_norm = self.norm(desc.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        local = self._compute_local(desc_norm)

        global_tokens = desc_norm.transpose(1, 2).contiguous()
        delta = self._forward_branch_attention(x, desc_norm, global_tokens, local)
        return x + delta


class AdaptiveHarmonicResonanceAdapter(nn.Module):
    def __init__(
        self,
        channels,
        groups=1,
        bottleneck_ratio=0.25,
        fusion='gated_residual',
        spectral_mode='amplitude_phase',
        local_kernel_sizes=(3, 5),
        init_scale=0.05,
        use_spatial_gate=True,
        low_freq_ratio=0.20,
        dynamic_only=True,
        harmonic_orders=(1, 2, 3),
        harmonic_sigma=1.5,
    ):
        super().__init__()
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        if fusion not in ['residual', 'gated_residual']:
            raise ValueError("fusion must be 'residual' or 'gated_residual'")
        if spectral_mode not in ['amplitude', 'amplitude_phase']:
            raise ValueError("spectral_mode must be 'amplitude' or 'amplitude_phase'")
        if len(harmonic_orders) == 0:
            raise ValueError("harmonic_orders must not be empty")

        self.channels = channels
        self.groups = groups
        self.group_channels = channels // groups
        self.fusion = fusion
        self.spectral_mode = spectral_mode
        self.use_spatial_gate = use_spatial_gate
        self.low_freq_ratio = float(low_freq_ratio)
        self.dynamic_only = dynamic_only
        self.harmonic_sigma = float(harmonic_sigma)

        hidden_dim = max(int(self.group_channels * bottleneck_ratio), 1)
        self.norm = nn.LayerNorm(channels)
        kernel_sizes = local_kernel_sizes if is_list_or_tuple(local_kernel_sizes) else [local_kernel_sizes]
        self.local_branches = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size=int(kernel_size),
                padding=int(kernel_size) // 2,
                groups=groups,
                bias=False,
            )
            for kernel_size in kernel_sizes
        ])
        self.local_fuse = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.harmonic_amp_mlp = nn.Sequential(
            nn.Linear(self.group_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.group_channels),
        )
        self.low_amp_mlp = nn.Sequential(
            nn.Linear(self.group_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.group_channels),
        )
        if spectral_mode == 'amplitude_phase':
            self.harmonic_phase_mlp = nn.Sequential(
                nn.Linear(self.group_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.group_channels),
            )
            self.low_phase_mlp = nn.Sequential(
                nn.Linear(self.group_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.group_channels),
            )
        else:
            self.harmonic_phase_mlp = None
            self.low_phase_mlp = None
        self.register_buffer(
            'harmonic_orders',
            torch.tensor(list(harmonic_orders), dtype=torch.float32),
            persistent=False,
        )
        self.order_weight_mlp = nn.Sequential(
            nn.Linear(channels, max(channels // 4, 32)),
            nn.GELU(),
            nn.Linear(max(channels // 4, 32), len(harmonic_orders)),
        )
        self.identity_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.harmonic_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.harmonic_gap_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Conv1d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
        )
        self.res_scale = nn.Parameter(torch.full((1, channels, 1), float(init_scale)))
        self.fusion_gate = nn.Conv1d(channels, channels, kernel_size=1, bias=True) \
            if fusion == 'gated_residual' else None
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        ) if use_spatial_gate else None

    def _modulate_spectrum(self, amp, phase, amp_mlp, phase_mlp=None, gate_scale=0.25):
        n, freq_bins, c = amp.shape
        amp_grouped = amp.view(n, freq_bins, self.groups, self.group_channels)
        amp_gate = gate_scale * torch.tanh(amp_mlp(amp_grouped)).view(n, freq_bins, c)
        amp = amp * (1.0 + amp_gate)
        if phase_mlp is not None:
            phase_grouped = phase.view(n, freq_bins, self.groups, self.group_channels)
            phase = phase + gate_scale * torch.tanh(phase_mlp(phase_grouped)).view(n, freq_bins, c)
        return amp, phase

    def _compute_local(self, desc_norm):
        local = 0.0
        for branch in self.local_branches:
            local = local + branch(desc_norm)
        return self.local_fuse(local / len(self.local_branches))

    def _build_harmonic_mask(self, amp, desc_norm, low_bins):
        n, freq_bins = amp.size(0), amp.size(1)
        energy = amp.mean(dim=-1)
        high_energy = energy[:, low_bins:]
        if high_energy.numel() == 0:
            return energy.new_zeros(n, freq_bins)

        peak_idx = high_energy.argmax(dim=-1) + low_bins
        order_weights = torch.softmax(self.order_weight_mlp(desc_norm.mean(dim=-1)), dim=-1)
        centers = peak_idx.float().unsqueeze(1) * self.harmonic_orders.unsqueeze(0)
        centers = centers.clamp(max=freq_bins - 1)
        freq_axis = torch.arange(freq_bins, device=amp.device, dtype=amp.dtype).view(1, 1, -1)
        sigma = max(self.harmonic_sigma, 1e-3)
        masks = torch.exp(-0.5 * ((freq_axis - centers.unsqueeze(-1)) / sigma) ** 2)
        harmonic_mask = (masks * order_weights.unsqueeze(-1)).sum(dim=1)
        harmonic_mask[:, :low_bins] = 0.0
        return harmonic_mask

    def _compute_delta(self, x, desc_norm, delta_1d):
        delta = delta_1d
        if self.fusion_gate is not None:
            delta = delta * torch.sigmoid(self.fusion_gate(desc_norm))
        delta = self.res_scale * delta
        delta = delta.unsqueeze(-1).unsqueeze(-1)
        if self.spatial_gate is not None:
            spatial_gate = self.spatial_gate(x.mean(dim=2)).unsqueeze(2)
            delta = delta * spatial_gate
        return delta

    def forward(self, x):
        n, c, s = x.size()[:3]
        desc = x.mean(dim=(-1, -2))
        desc_norm = self.norm(desc.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        local = self._compute_local(desc_norm)

        tokens = desc_norm.transpose(1, 2).contiguous()
        freq = torch.fft.rfft(tokens, dim=1, norm='ortho')
        amp = torch.abs(freq)
        phase = torch.angle(freq)
        freq_bins = amp.size(1)
        low_bins = min(max(int(round(freq_bins * self.low_freq_ratio)), 1), freq_bins)

        low_amp = amp[:, :low_bins]
        low_phase = phase[:, :low_bins]
        low_amp, low_phase = self._modulate_spectrum(
            low_amp, low_phase, self.low_amp_mlp, self.low_phase_mlp, gate_scale=0.12)

        harmonic_amp = amp.clone()
        harmonic_phase = phase.clone()
        harmonic_mask = self._build_harmonic_mask(amp, desc_norm, low_bins).unsqueeze(-1)
        harmonic_amp, harmonic_phase = self._modulate_spectrum(
            harmonic_amp, harmonic_phase, self.harmonic_amp_mlp, self.harmonic_phase_mlp, gate_scale=0.25)
        harmonic_amp = harmonic_amp * harmonic_mask
        harmonic_phase = phase + (harmonic_phase - phase) * harmonic_mask

        identity_freq = torch.zeros_like(freq)
        identity_freq[:, :low_bins] = torch.polar(low_amp, low_phase)
        harmonic_freq = torch.zeros_like(freq)
        if harmonic_amp[:, low_bins:].numel() > 0:
            harmonic_freq[:, low_bins:] = torch.polar(harmonic_amp[:, low_bins:], harmonic_phase[:, low_bins:])
        elif not self.dynamic_only:
            harmonic_freq[:, :low_bins] = identity_freq[:, :low_bins]

        identity = torch.fft.irfft(identity_freq, n=s, dim=1, norm='ortho').transpose(1, 2).contiguous()
        harmonic = torch.fft.irfft(harmonic_freq, n=s, dim=1, norm='ortho').transpose(1, 2).contiguous()
        identity = self.identity_proj(identity)
        harmonic = self.harmonic_proj(harmonic)
        harmonic_gap = self.harmonic_gap_proj(harmonic - identity)
        delta = self.mix(torch.cat([desc_norm, local, harmonic, harmonic_gap], dim=1))
        delta = self._compute_delta(x, desc_norm, delta)
        return x + delta


class ComplexHarmonicFilterBankAdapter(nn.Module):
    def __init__(
        self,
        channels,
        groups=1,
        bottleneck_ratio=0.25,
        fusion='gated_residual',
        local_kernel_sizes=(3, 5),
        init_scale=0.05,
        use_spatial_gate=True,
        low_freq_ratio=0.20,
        harmonic_orders=(1, 2, 3),
        harmonic_sigma=1.5,
        bank_temperature=1.0,
        **kwargs,
    ):
        super().__init__()
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        if fusion not in ['residual', 'gated_residual']:
            raise ValueError("fusion must be 'residual' or 'gated_residual'")
        if len(harmonic_orders) == 0:
            raise ValueError("harmonic_orders must not be empty")
        if kwargs:
            ignored = ', '.join(sorted(kwargs.keys()))
            print("ComplexHarmonicFilterBankAdapter ignores unused config keys: {}".format(ignored))

        self.channels = channels
        self.groups = groups
        self.group_channels = channels // groups
        self.fusion = fusion
        self.use_spatial_gate = use_spatial_gate
        self.low_freq_ratio = float(low_freq_ratio)
        self.harmonic_sigma = float(harmonic_sigma)
        self.bank_temperature = float(bank_temperature)

        hidden_dim = max(int(self.group_channels * bottleneck_ratio), 1)
        self.norm = nn.LayerNorm(channels)
        kernel_sizes = local_kernel_sizes if is_list_or_tuple(local_kernel_sizes) else [local_kernel_sizes]
        self.local_branches = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size=int(kernel_size),
                padding=int(kernel_size) // 2,
                groups=groups,
                bias=False,
            )
            for kernel_size in kernel_sizes
        ])
        self.local_fuse = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.register_buffer(
            'harmonic_orders',
            torch.tensor(list(harmonic_orders), dtype=torch.float32),
            persistent=False,
        )
        self.order_weight_mlp = nn.Sequential(
            nn.Linear(channels, max(channels // 4, 32)),
            nn.GELU(),
            nn.Linear(max(channels // 4, 32), len(harmonic_orders)),
        )

        # Three spectral banks: low-frequency identity-like, harmonic band, residual band.
        self.bank_names = ['low', 'harmonic', 'residual']
        self.bank_scales = (0.10, 0.20, 0.15)
        self.bank_gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.group_channels * 5, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.group_channels * 2),
            )
            for _ in self.bank_names
        ])
        self.bank_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            )
            for _ in self.bank_names
        ])
        self.router = nn.Sequential(
            nn.Linear(channels + 4, max(channels // 4, 32)),
            nn.GELU(),
            nn.Linear(max(channels // 4, 32), len(self.bank_names)),
        )
        self.bank_fuse = nn.Sequential(
            nn.Conv1d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
        )
        self.res_scale = nn.Parameter(torch.full((1, channels, 1), float(init_scale)))
        self.fusion_gate = nn.Conv1d(channels, channels, kernel_size=1, bias=True) \
            if fusion == 'gated_residual' else None
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        ) if use_spatial_gate else None
        self.delta_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def _compute_local(self, desc_norm):
        local = 0.0
        for branch in self.local_branches:
            local = local + branch(desc_norm)
        return self.local_fuse(local / len(self.local_branches))

    def _spectral_statistics(self, amp, low_bins):
        energy = amp.mean(dim=-1)  # [n, freq]
        denom = energy.sum(dim=1, keepdim=True).clamp_min(1e-6)
        prob = energy / denom
        entropy = -(prob * (prob.clamp_min(1e-6)).log()).sum(dim=1, keepdim=True)
        entropy = entropy / math.log(float(max(energy.size(1), 2)))

        if energy.size(1) > low_bins:
            high_energy = energy[:, low_bins:]
            peak_idx = high_energy.argmax(dim=-1) + low_bins
        else:
            peak_idx = energy.argmax(dim=-1)
        peak_val = energy.gather(1, peak_idx.unsqueeze(-1)).squeeze(-1)

        mean_energy = energy.mean(dim=1, keepdim=True)
        std_energy = energy.std(dim=1, keepdim=True, unbiased=False)
        peak_contrast = (peak_val.unsqueeze(-1) - mean_energy) / (std_energy + 1e-6)
        return torch.cat([mean_energy, std_energy, entropy, peak_contrast], dim=-1), peak_idx

    def _build_bank_masks(self, amp, desc_norm, low_bins):
        n, freq_bins = amp.size(0), amp.size(1)
        bank_device = amp.device
        bank_dtype = amp.dtype
        low_mask = torch.zeros(n, freq_bins, device=bank_device, dtype=bank_dtype)
        low_mask[:, :low_bins] = 1.0

        energy_stats, peak_idx = self._spectral_statistics(amp, low_bins)
        order_weights = torch.softmax(self.order_weight_mlp(desc_norm.mean(dim=-1)), dim=-1)
        centers = peak_idx.float().unsqueeze(1) * self.harmonic_orders.unsqueeze(0)
        centers = centers.clamp(max=freq_bins - 1)
        freq_axis = torch.arange(freq_bins, device=bank_device, dtype=bank_dtype).view(1, 1, -1)
        sigma = max(self.harmonic_sigma, 1e-3)
        harmonic_masks = torch.exp(-0.5 * ((freq_axis - centers.unsqueeze(-1)) / sigma) ** 2)
        harmonic_mask = (harmonic_masks * order_weights.unsqueeze(-1)).sum(dim=1)
        harmonic_mask[:, :low_bins] = 0.0
        harmonic_norm = harmonic_mask.amax(dim=1, keepdim=True).clamp_min(1e-6)
        harmonic_mask = harmonic_mask / harmonic_norm

        residual_mask = torch.clamp(1.0 - torch.clamp(low_mask + harmonic_mask, max=1.0), min=0.0)
        return [low_mask, harmonic_mask, residual_mask], energy_stats

    def _complex_gate(self, freq, gate_mlp, gate_scale=0.2):
        n, freq_bins, c = freq.shape
        real = freq.real.view(n, freq_bins, self.groups, self.group_channels)
        imag = freq.imag.view(n, freq_bins, self.groups, self.group_channels)
        amp = torch.sqrt(real * real + imag * imag + 1e-6)
        amp_norm = amp / amp.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        phase_denom = amp.clamp_min(1e-6)
        cos_phase = real / phase_denom
        sin_phase = imag / phase_denom
        gate_input = torch.cat([real, imag, amp_norm, cos_phase, sin_phase], dim=-1)
        gate = torch.tanh(gate_mlp(gate_input))
        gate = gate.view(n, freq_bins, self.groups, 2, self.group_channels)
        gate_r = gate[:, :, :, 0, :] * gate_scale
        gate_i = gate[:, :, :, 1, :] * gate_scale
        out_real = real * (1.0 + gate_r) - imag * gate_i
        out_imag = real * gate_i + imag * (1.0 + gate_r)
        out_real = out_real.reshape(n, freq_bins, c)
        out_imag = out_imag.reshape(n, freq_bins, c)
        return torch.complex(out_real, out_imag)

    def _route_banks(self, desc_norm, energy_stats):
        router_input = torch.cat([desc_norm.mean(dim=-1), energy_stats], dim=-1)
        weights = torch.softmax(self.router(router_input) / max(self.bank_temperature, 1e-6), dim=-1)
        return weights

    def _compute_delta(self, x, desc_norm, delta_1d):
        delta = delta_1d
        if self.fusion_gate is not None:
            delta = delta * torch.sigmoid(self.fusion_gate(desc_norm))
        delta = self.res_scale * delta
        delta = self.delta_proj(delta)
        delta = delta.unsqueeze(-1).unsqueeze(-1)
        if self.spatial_gate is not None:
            spatial_gate = self.spatial_gate(x.mean(dim=2)).unsqueeze(2)
            delta = delta * spatial_gate
        return delta

    def forward(self, x):
        """
            x: [n, c, s, h, w]
        """
        n, c, s = x.size()[:3]
        x_fp32 = x.float()
        desc = x_fp32.mean(dim=(-1, -2))  # [n, c, s]
        desc_norm = self.norm(desc.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        local = self._compute_local(desc_norm)

        tokens = desc_norm.transpose(1, 2).contiguous()
        freq = torch.fft.rfft(tokens, dim=1, norm='ortho')
        amp = torch.abs(freq)
        freq_bins = amp.size(1)
        low_bins = min(max(int(round(freq_bins * self.low_freq_ratio)), 1), freq_bins)

        bank_masks, energy_stats = self._build_bank_masks(amp, desc_norm, low_bins)
        bank_outputs = []
        for idx, (mask, gate_mlp, proj, gate_scale) in enumerate(zip(
                bank_masks, self.bank_gate_mlps, self.bank_proj, self.bank_scales)):
            bank_freq = freq * mask.unsqueeze(-1)
            bank_freq = self._complex_gate(bank_freq, gate_mlp, gate_scale=gate_scale)
            bank_time = torch.fft.irfft(bank_freq, n=s, dim=1, norm='ortho').transpose(1, 2).contiguous()
            bank_outputs.append(proj(bank_time))

        bank_stack = torch.stack(bank_outputs, dim=1)  # [n, bank, c, s]
        bank_weights = self._route_banks(desc_norm, energy_stats)
        fused_bank = (bank_stack * bank_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        bank_gap = self.bank_proj[-1](fused_bank - bank_outputs[0])
        delta = self.bank_fuse(torch.cat([desc_norm, local, fused_bank, bank_gap], dim=1))
        delta = self._compute_delta(x_fp32, desc_norm, delta)
        return x + delta.to(dtype=x.dtype)


class PeriodicTemporalStateAdapter(nn.Module):
    def __init__(
        self,
        channels,
        groups=1,
        bottleneck_ratio=0.25,
        fusion='gated_residual',
        local_kernel_sizes=(3, 5, 7),
        init_scale=0.05,
        use_spatial_gate=True,
        low_freq_ratio=0.20,
        router_temperature=1.0,
        **kwargs,
    ):
        super().__init__()
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        if fusion not in ['residual', 'gated_residual']:
            raise ValueError("fusion must be 'residual' or 'gated_residual'")
        if kwargs:
            ignored = ', '.join(sorted(kwargs.keys()))
            print("PeriodicTemporalStateAdapter ignores unused config keys: {}".format(ignored))

        self.channels = channels
        self.groups = groups
        self.fusion = fusion
        self.low_freq_ratio = float(low_freq_ratio)
        self.router_temperature = float(router_temperature)

        hidden_dim = max(int(channels * bottleneck_ratio), 16)
        router_hidden = max(channels // 4, 32)
        kernel_sizes = local_kernel_sizes if is_list_or_tuple(local_kernel_sizes) else [local_kernel_sizes]

        self.norm = nn.LayerNorm(channels)
        self.motion_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    channels * 3,
                    channels,
                    kernel_size=int(kernel_size),
                    padding=int(kernel_size) // 2,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            )
            for kernel_size in kernel_sizes
        ])
        self.motion_fuse = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.phase_gate = nn.Sequential(
            nn.Conv1d(3, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, channels, kernel_size=1, bias=True),
        )
        self.phase_out_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.memory_gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1, groups=groups, bias=True),
            nn.Sigmoid(),
        )
        self.memory_out_proj = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
        )
        self.router = nn.Sequential(
            nn.Linear(channels + 5, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, 3),
        )
        self.branch_out_proj = nn.Sequential(
            nn.Conv1d(channels * 4, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
        )
        self.res_scale = nn.Parameter(torch.full((1, channels, 1), float(init_scale)))
        self.fusion_gate = nn.Conv1d(channels, channels, kernel_size=1, bias=True) \
            if fusion == 'gated_residual' else None
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        ) if use_spatial_gate else None

    def _temporal_diff(self, desc_norm, order):
        if desc_norm.size(-1) <= order:
            return torch.zeros_like(desc_norm)
        diff = desc_norm[:, :, order:] - desc_norm[:, :, :-order]
        return F.pad(diff, (order, 0))

    def _compute_motion(self, desc_norm):
        diff1 = self._temporal_diff(desc_norm, 1)
        diff2 = self._temporal_diff(desc_norm, 2)
        motion_input = torch.cat([desc_norm, diff1, diff2], dim=1)
        motion = 0.0
        for branch in self.motion_branches:
            motion = motion + branch(motion_input)
        motion = self.motion_fuse(motion / len(self.motion_branches))
        return motion, diff1, diff2

    def _estimate_cycle(self, desc_norm):
        n, c, s = desc_norm.shape
        phase_cos = desc_norm.new_ones(n, 1, s)
        phase_sin = desc_norm.new_zeros(n, 1, s)
        cycle_conf = desc_norm.new_zeros(n, 1)
        peak_norm = desc_norm.new_zeros(n, 1)

        tokens = desc_norm.transpose(1, 2).contiguous()
        freq = torch.fft.rfft(tokens, dim=1, norm='ortho')
        freq_energy = torch.abs(freq).mean(dim=-1)
        freq_bins = freq_energy.size(1)
        if freq_bins <= 1:
            return phase_cos, phase_sin, cycle_conf, peak_norm

        low_bins = min(max(int(round(freq_bins * self.low_freq_ratio)), 1), freq_bins - 1)
        high_energy = freq_energy[:, low_bins:]
        peak_rel = high_energy.argmax(dim=-1)
        peak_idx = peak_rel + low_bins
        peak_val = high_energy.gather(1, peak_rel.unsqueeze(-1))
        cycle_conf = peak_val / high_energy.sum(dim=1, keepdim=True).clamp_min(1e-6)
        peak_norm = peak_idx.to(dtype=desc_norm.dtype).unsqueeze(-1) / float(freq_bins - 1)

        mean_freq = freq.mean(dim=-1)
        dom_coeff = mean_freq.gather(1, peak_idx.unsqueeze(-1)).squeeze(-1)
        dom_phase = torch.angle(dom_coeff)
        time_axis = torch.arange(s, device=desc_norm.device, dtype=desc_norm.dtype).view(1, s)
        angular = (2.0 * math.pi / max(s, 1)) * peak_idx.to(dtype=desc_norm.dtype).unsqueeze(-1)
        phase_cos = torch.cos(angular * time_axis + dom_phase.unsqueeze(-1)).unsqueeze(1)
        phase_sin = torch.sin(angular * time_axis + dom_phase.unsqueeze(-1)).unsqueeze(1)
        return phase_cos, phase_sin, cycle_conf, peak_norm

    def _compute_phase_branch(self, desc_norm):
        n, _, s = desc_norm.shape
        phase_cos, phase_sin, cycle_conf, peak_norm = self._estimate_cycle(desc_norm)
        cycle_token = cycle_conf.unsqueeze(-1).expand(n, 1, s)
        phase_features = torch.cat([phase_cos, phase_sin, cycle_token], dim=1)
        phase_gate = torch.sigmoid(self.phase_gate(phase_features))
        phase = self.phase_out_proj(desc_norm * phase_gate)
        return phase, cycle_conf, peak_norm

    def _run_bidirectional_memory(self, desc_norm, gate):
        n, c, s = desc_norm.shape
        forward = torch.zeros_like(desc_norm)
        backward = torch.zeros_like(desc_norm)

        state = desc_norm[:, :, 0]
        forward[:, :, 0] = state
        for idx in range(1, s):
            alpha = gate[:, :, idx]
            state = alpha * state + (1.0 - alpha) * desc_norm[:, :, idx]
            forward[:, :, idx] = state

        state = desc_norm[:, :, -1]
        backward[:, :, -1] = state
        for idx in range(s - 2, -1, -1):
            alpha = gate[:, :, idx]
            state = alpha * state + (1.0 - alpha) * desc_norm[:, :, idx]
            backward[:, :, idx] = state
        return forward, backward

    def _compute_memory(self, desc_norm, diff1):
        gate = self.memory_gate(torch.cat([desc_norm, diff1.abs()], dim=1))
        forward, backward = self._run_bidirectional_memory(desc_norm, gate)
        memory = self.memory_out_proj(torch.cat([desc_norm, forward, backward], dim=1))
        return memory

    def _compute_delta(self, x, desc_norm, delta_1d):
        delta = delta_1d
        if self.fusion_gate is not None:
            delta = delta * torch.sigmoid(self.fusion_gate(desc_norm))
        delta = self.res_scale * delta
        delta = delta.unsqueeze(-1).unsqueeze(-1)
        if self.spatial_gate is not None:
            spatial_gate = self.spatial_gate(x.mean(dim=2)).unsqueeze(2)
            delta = delta * spatial_gate
        return delta

    def forward(self, x):
        n, c, s = x.size()[:3]
        x_fp32 = x.float()
        desc = x_fp32.mean(dim=(-1, -2))
        desc_norm = self.norm(desc.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        motion, diff1, _ = self._compute_motion(desc_norm)
        phase, cycle_conf, peak_norm = self._compute_phase_branch(desc_norm)
        memory = self._compute_memory(desc_norm, diff1)

        seq_std = desc_norm.std(dim=2, unbiased=False).mean(dim=1, keepdim=True)
        diff_energy = diff1.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        motion_energy = motion.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        memory_energy = (memory - desc_norm).abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        router_stats = torch.cat(
            [seq_std, diff_energy, cycle_conf, peak_norm, memory_energy + motion_energy],
            dim=-1,
        )
        router_input = torch.cat([desc_norm.mean(dim=-1), router_stats], dim=-1)
        branch_weights = torch.softmax(
            self.router(router_input) / max(self.router_temperature, 1e-6), dim=-1)

        branch_stack = torch.stack([motion, phase, memory], dim=1)
        fused = (branch_stack * branch_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        delta = self.branch_out_proj(
            torch.cat([desc_norm, fused, motion - phase, memory - desc_norm], dim=1))
        delta = self._compute_delta(x_fp32, desc_norm, delta)
        return x + delta.to(dtype=x.dtype)


class TemporalQualityGateAdapter(nn.Module):
    def __init__(
        self,
        channels,
        groups=1,
        bottleneck_ratio=0.125,
        fusion='gated_residual',
        local_kernel_size=3,
        init_scale=0.05,
        init_gate_strength=0.5,
        use_spatial_gate=True,
        part_bins=4,
        **kwargs,
    ):
        super().__init__()
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        if fusion not in ['residual', 'gated_residual']:
            raise ValueError("fusion must be 'residual' or 'gated_residual'")

        if 'local_kernel_sizes' in kwargs:
            kernel_sizes = kwargs.pop('local_kernel_sizes')
            if is_list_or_tuple(kernel_sizes):
                local_kernel_size = int(kernel_sizes[len(kernel_sizes) // 2])
            else:
                local_kernel_size = int(kernel_sizes)
        if kwargs:
            ignored = ', '.join(sorted(kwargs.keys()))
            print("TemporalQualityGateAdapter ignores unused config keys: {}".format(ignored))

        self.channels = channels
        self.groups = groups
        self.fusion = fusion
        self.part_bins = int(part_bins)
        hidden_dim = max(int(channels * bottleneck_ratio), 16)

        self.norm = nn.LayerNorm(channels)
        self.temporal_context = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=int(local_kernel_size),
                padding=int(local_kernel_size) // 2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.frame_gate_mlp = nn.Sequential(
            nn.Conv1d(channels * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=True),
        )
        self.part_gate_mlp = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=True),
        ) if self.part_bins > 0 else None

        self.frame_gate_strength = nn.Parameter(torch.tensor(float(init_gate_strength)))
        self.part_gate_strength = nn.Parameter(torch.tensor(float(init_gate_strength)))
        self.res_scale = nn.Parameter(torch.full((1, channels, 1), float(init_scale)))
        self.fusion_gate = nn.Conv1d(channels, channels, kernel_size=1, bias=True) \
            if fusion == 'gated_residual' else None
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        ) if use_spatial_gate else None

        nn.init.zeros_(self.temporal_context[-1].weight)
        nn.init.zeros_(self.temporal_context[-1].bias)
        nn.init.zeros_(self.frame_gate_mlp[-1].weight)
        nn.init.zeros_(self.frame_gate_mlp[-1].bias)
        if self.part_gate_mlp is not None:
            nn.init.zeros_(self.part_gate_mlp[-1].weight)
            nn.init.zeros_(self.part_gate_mlp[-1].bias)

    def _temporal_diff(self, desc_norm):
        diff = desc_norm[:, :, 1:] - desc_norm[:, :, :-1]
        return F.pad(diff, (1, 0))

    def _build_gate_weight(self, gate_logits, strength, norm_dim):
        gate = torch.sigmoid(gate_logits)
        weight = 1.0 + torch.tanh(strength) * (gate - 0.5) * 2.0
        return weight / weight.mean(dim=norm_dim, keepdim=True).clamp_min(1e-6)

    def _compute_part_weight(self, x_fp32):
        if self.part_gate_mlp is None:
            return 1.0

        n, _, s, h, _ = x_fp32.shape
        part_desc = x_fp32.mean(dim=-1).permute(0, 2, 1, 3).reshape(n * s, self.channels, h)
        part_desc = F.adaptive_avg_pool1d(part_desc, self.part_bins)
        part_logits = self.part_gate_mlp(part_desc)
        part_weight = self._build_gate_weight(part_logits, self.part_gate_strength, norm_dim=-1)
        part_weight = F.interpolate(part_weight, size=h, mode='linear', align_corners=False)
        part_weight = part_weight.view(n, s, 1, h).permute(0, 2, 1, 3).unsqueeze(-1)
        return part_weight

    def _compute_delta(self, x_fp32, desc_norm, temporal_delta):
        delta = temporal_delta
        if self.fusion_gate is not None:
            delta = delta * torch.sigmoid(self.fusion_gate(desc_norm))
        delta = self.res_scale * delta
        delta = delta.unsqueeze(-1).unsqueeze(-1)
        if self.spatial_gate is not None:
            spatial_gate = self.spatial_gate(x_fp32.mean(dim=2)).unsqueeze(2)
            delta = delta * spatial_gate
        return delta

    def forward(self, x):
        x_fp32 = x.float()
        desc = x_fp32.mean(dim=(-1, -2))
        desc_norm = self.norm(desc.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        diff = self._temporal_diff(desc_norm).abs()

        frame_logits = self.frame_gate_mlp(torch.cat([desc_norm, diff], dim=1))
        frame_weight = self._build_gate_weight(frame_logits, self.frame_gate_strength, norm_dim=2)
        frame_weight = frame_weight.unsqueeze(-1).unsqueeze(-1)
        part_weight = self._compute_part_weight(x_fp32)

        temporal_input = desc_norm + 0.5 * diff
        temporal_delta = self.temporal_context(temporal_input)
        temporal_delta = self._compute_delta(x_fp32, desc_norm, temporal_delta)
        gated = x_fp32 * frame_weight
        if not isinstance(part_weight, float):
            gated = gated * part_weight
        return (gated + temporal_delta).to(dtype=x.dtype)


class LaStGaitAdapter(nn.Module):
    def __init__(
        self,
        channels,
        part_bins=4,
        topk_ratio=0.35,
        min_topk=1,
        gaussian_sigma=0.25,
        gate_strength=0.25,
        init_scale=0.02,
        min_gate=0.75,
        max_gate=1.25,
        use_delta=True,
        **kwargs,
    ):
        super().__init__()
        if int(part_bins) <= 0:
            raise ValueError("part_bins must be positive")
        if kwargs:
            ignored = ', '.join(sorted(kwargs.keys()))
            print("LaStGaitAdapter ignores unused config keys: {}".format(ignored))

        self.channels = channels
        self.part_bins = int(part_bins)
        self.topk_ratio = float(topk_ratio)
        self.min_topk = int(min_topk)
        self.gaussian_sigma = float(gaussian_sigma)
        self.min_gate = float(min_gate)
        self.max_gate = float(max_gate)
        self.use_delta = use_delta
        self.eps = 1e-6

        self.gate_strength = nn.Parameter(torch.tensor(float(gate_strength)))
        self.res_scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), float(init_scale)))

    def _channel_low_pass(self, tokens):
        n, token_num, channels = tokens.shape
        freq = torch.fft.rfft(tokens.float(), dim=-1, norm='ortho')
        freq_bins = freq.size(-1)
        freq_axis = torch.linspace(0, 1, freq_bins, device=tokens.device, dtype=tokens.float().dtype)
        sigma = max(self.gaussian_sigma, 1e-4)
        mask = torch.exp(-0.5 * (freq_axis / sigma) ** 2)
        low = torch.fft.irfft(freq * mask.view(1, 1, -1), n=channels, dim=-1, norm='ortho')
        return low

    def _part_tokens(self, x_fp32):
        n, c, s, h, _ = x_fp32.shape
        part_feat = x_fp32.mean(dim=-1).permute(0, 2, 1, 3).reshape(n * s, c, h)
        part_feat = F.adaptive_avg_pool1d(part_feat, self.part_bins)
        tokens = part_feat.view(n, s, c, self.part_bins).permute(0, 1, 3, 2).contiguous()
        return tokens.view(n, s * self.part_bins, c), s, h

    def _stability_vote(self, tokens):
        low = self._channel_low_pass(tokens)
        stability = low.abs() / (low.sub(tokens.float()).abs() + self.eps)
        token_num = tokens.size(1)
        topk = min(max(int(round(token_num * self.topk_ratio)), self.min_topk), token_num)
        topk_idx = stability.topk(topk, dim=1).indices
        selected = stability.new_zeros(stability.shape)
        selected.scatter_(1, topk_idx, 1.0)
        vote = selected.mean(dim=-1)
        vote = vote / vote.mean(dim=1, keepdim=True).clamp_min(self.eps)
        gate = 1.0 + torch.tanh(self.gate_strength) * (vote - 1.0)
        return gate.clamp(min=self.min_gate, max=self.max_gate), low

    def _expand_part_map(self, part_map, seq_len, height):
        n = part_map.size(0)
        part_map = part_map.view(n, seq_len, self.part_bins).permute(0, 2, 1).unsqueeze(1)
        part_map = F.interpolate(part_map, size=(height, seq_len), mode='bilinear', align_corners=False)
        return part_map.permute(0, 1, 3, 2).unsqueeze(-1).contiguous()

    def forward(self, x):
        x_fp32 = x.float()
        tokens, seq_len, height = self._part_tokens(x_fp32)
        token_gate, low_tokens = self._stability_vote(tokens)
        gate = self._expand_part_map(token_gate, seq_len, height)
        out = x_fp32 * gate

        if self.use_delta:
            delta = low_tokens.sub(tokens.float())
            delta = delta.view(x.size(0), seq_len, self.part_bins, self.channels)
            delta = delta.permute(0, 3, 2, 1).contiguous()
            delta = F.interpolate(delta, size=(height, seq_len), mode='bilinear', align_corners=False)
            delta = delta.permute(0, 1, 3, 2).unsqueeze(-1).contiguous()
            out = out + self.res_scale * delta
        return out.to(dtype=x.dtype)


class LaStTemporalPooling(nn.Module):
    def __init__(
        self,
        topk_ratio=0.35,
        min_topk=1,
        gaussian_sigma=0.25,
        stable_fusion_weight=0.35,
        learnable_fusion=True,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            ignored = ', '.join(sorted(kwargs.keys()))
            print("LaStTemporalPooling ignores unused config keys: {}".format(ignored))
        self.topk_ratio = float(topk_ratio)
        self.min_topk = int(min_topk)
        self.gaussian_sigma = float(gaussian_sigma)
        self.learnable_fusion = learnable_fusion
        self.eps = 1e-6

        stable_fusion_weight = min(max(float(stable_fusion_weight), 1e-4), 1.0 - 1e-4)
        if learnable_fusion:
            init_logit = math.log(stable_fusion_weight / (1.0 - stable_fusion_weight))
            self.fusion_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        else:
            self.register_buffer('fusion_weight', torch.tensor(stable_fusion_weight, dtype=torch.float32))

    def _channel_low_pass(self, tokens):
        channels = tokens.size(-1)
        freq = torch.fft.rfft(tokens.float(), dim=-1, norm='ortho')
        freq_bins = freq.size(-1)
        freq_axis = torch.linspace(0, 1, freq_bins, device=tokens.device, dtype=tokens.float().dtype)
        sigma = max(self.gaussian_sigma, 1e-4)
        mask = torch.exp(-0.5 * (freq_axis / sigma) ** 2)
        return torch.fft.irfft(freq * mask.view(*([1] * (freq.ndim - 1)), -1), n=channels, dim=-1, norm='ortho')

    def _stable_weight(self):
        if self.learnable_fusion:
            return torch.sigmoid(self.fusion_logit)
        return self.fusion_weight

    def _pool_dim2(self, x):
        n, c, s, h, w = x.shape
        tokens = x.permute(0, 2, 3, 4, 1).contiguous()
        low = self._channel_low_pass(tokens)
        stability = low.abs() / (low.sub(tokens.float()).abs() + self.eps)
        score = stability.permute(0, 4, 1, 2, 3).contiguous()
        topk = min(max(int(round(s * self.topk_ratio)), self.min_topk), s)
        indices = score.topk(topk, dim=2).indices
        stable_pool = x.gather(2, indices).mean(dim=2)
        max_pool = x.max(dim=2)[0]
        alpha = self._stable_weight().to(device=x.device, dtype=x.dtype)
        return alpha * stable_pool + (1.0 - alpha) * max_pool

    def forward(self, seqs, seqL=None, dim=2, options={}):
        if options and 'dim' in options:
            dim = options['dim']
        if dim != 2:
            raise ValueError("LaStTemporalPooling only supports temporal dim=2 for [n, c, s, h, w] tensors.")

        if seqL is None:
            return [self._pool_dim2(seqs)]

        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]
        pooled = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(dim, curr_start, curr_seqL)
            pooled.append(self._pool_dim2(narrowed_seq))
        return [torch.cat(pooled)]


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.permute(2, 0, 1).contiguous()
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0).contiguous()


class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(
                x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()


class FocalConv2d(nn.Module):
    """
        GaitPart: Temporal Part-based Model for Gait Recognition
        CVPR2020: https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_GaitPart_Temporal_Part-Based_Model_for_Gait_Recognition_CVPR_2020_paper.pdf
        Github: https://github.com/ChaoFan96/GaitPart
    """
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.size(2)
            split_size = int(h // 2**self.halving)
            z = x.split(split_size, 2)
            z = torch.cat([self.conv(_) for _ in z], 2)
        return z


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs


class GaitAlign(nn.Module):
    """
        GaitEdge: Beyond Plain End-to-end Gait Recognition for Better Practicality
        ECCV2022: https://arxiv.org/pdf/2203.03972v2.pdf
        Github: https://github.com/ShiqiYu/OpenGait/tree/master/configs/gaitedge
    """
    def __init__(self, H=64, W=44, eps=1, **kwargs):
        super(GaitAlign, self).__init__()
        self.H, self.W, self.eps = H, W, eps
        self.Pad = nn.ZeroPad2d((int(self.W / 2), int(self.W / 2), 0, 0))
        self.RoiPool = RoIAlign((self.H, self.W), 1, sampling_ratio=-1)

    def forward(self, feature_map, binary_mask, w_h_ratio):
        """
           In  sils:         [n, c, h, w]
               w_h_ratio:    [n, 1]
           Out aligned_sils: [n, c, H, W]
        """
        n, c, h, w = feature_map.size()
        # w_h_ratio = w_h_ratio.repeat(1, 1) # [n, 1]
        w_h_ratio = w_h_ratio.view(-1, 1)  # [n, 1]

        h_sum = binary_mask.sum(-1)  # [n, c, h]
        _ = (h_sum >= self.eps).float().cumsum(axis=-1)  # [n, c, h]
        h_top = (_ == 0).float().sum(-1)  # [n, c]
        h_bot = (_ != torch.max(_, dim=-1, keepdim=True)
                 [0]).float().sum(-1) + 1.  # [n, c]

        w_sum = binary_mask.sum(-2)  # [n, c, w]
        w_cumsum = w_sum.cumsum(axis=-1)  # [n, c, w]
        w_h_sum = w_sum.sum(-1).unsqueeze(-1)  # [n, c, 1]
        w_center = (w_cumsum < w_h_sum / 2.).float().sum(-1)  # [n, c]

        p1 = self.W - self.H * w_h_ratio
        p1 = p1 / 2.
        p1 = torch.clamp(p1, min=0)  # [n, c]
        t_w = w_h_ratio * self.H / w
        p2 = p1 / t_w  # [n, c]

        height = h_bot - h_top  # [n, c]
        width = height * w / h  # [n, c]
        width_p = int(self.W / 2)

        feature_map = self.Pad(feature_map)
        w_center = w_center + width_p  # [n, c]

        w_left = w_center - width / 2 - p2  # [n, c]
        w_right = w_center + width / 2 + p2  # [n, c]

        w_left = torch.clamp(w_left, min=0., max=w+2*width_p)
        w_right = torch.clamp(w_right, min=0., max=w+2*width_p)

        boxes = torch.cat([w_left, h_top, w_right, h_bot], dim=-1)
        # index of bbox in batch
        box_index = torch.arange(n, device=feature_map.device)
        rois = torch.cat([box_index.view(-1, 1), boxes], -1)
        crops = self.RoiPool(feature_map, rois)  # [n, c, H, W]
        return crops


def RmBN2dAffine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False


'''
Modifed from https://github.com/BNU-IVC/FastPoseGait/blob/main/fastposegait/modeling/components/units
'''

class Graph():
    """
    # Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
    """
    def __init__(self, joint_format='coco', max_hop=2, dilation=1):
        self.joint_format = joint_format
        self.max_hop = max_hop
        self.dilation = dilation

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.joint_format == 'coco':
            # keypoints = {
            #     0: "nose",
            #     1: "left_eye",
            #     2: "right_eye",
            #     3: "left_ear",
            #     4: "right_ear",
            #     5: "left_shoulder",
            #     6: "right_shoulder",
            #     7: "left_elbow",
            #     8: "right_elbow",
            #     9: "left_wrist",
            #     10: "right_wrist",
            #     11: "left_hip",
            #     12: "right_hip",
            #     13: "left_knee",
            #     14: "right_knee",
            #     15: "left_ankle",
            #     16: "right_ankle"
            # }
            num_node = 17
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
                             (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                             (11, 13), (13, 15), (12, 14), (14, 16)]
            self.edge = self_link + neighbor_link
            self.center = 0
            self.flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
            parts = [
                np.array([5, 7, 9]),                      # left_arm
                np.array([6, 8, 10]),                     # right_arm
                np.array([11, 13, 15]),                   # left_leg
                np.array([12, 14, 16]),                   # right_leg
                np.array([0, 1, 2, 3, 4]),                # head
            ]

        elif self.joint_format == 'coco-no-head':
            num_node = 12
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1),
                             (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7),
                             (6, 8), (8, 10), (7, 9), (9, 11)]
            self.edge = self_link + neighbor_link
            self.center = 0
            connect_joint = np.array([3,1,0,2,4,0,6,8,10,7,9,11])
            parts =[
                np.array([0, 2, 4]),       # left_arm
                np.array([1, 3, 5]),       # right_arm
                np.array([6, 8, 10]),      # left_leg
                np.array([7, 9, 11])       # right_leg
            ]

        elif self.joint_format =='alphapose' or self.joint_format =='openpose':
            num_node = 18
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (0, 14), (0, 15), (14, 16), (15, 17),
                             (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                             (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]
            self.edge = self_link + neighbor_link
            self.center = 1
            self.flip_idx = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
            connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            parts = [
                np.array([5, 6, 7]),               # left_arm
                np.array([2, 3, 4]),               # right_arm
                np.array([11, 12, 13]),            # left_leg
                np.array([8, 9, 10]),              # right_leg
                np.array([0, 1, 14, 15, 16, 17]),  # head
            ]

        else:
            num_node, neighbor_link, connect_joint, parts = 0, [], [], []
            raise ValueError('Error: Do NOT exist this dataset: {}!'.format(self.dataset))
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD


class TemporalBasicBlock(nn.Module):
    """
        TemporalConv_Res_Block
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, channels, temporal_window_size, stride=1, residual=False,reduction=0,get_res=False,tcn_stride=False):
        super(TemporalBasicBlock, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)

        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride,1)),
                nn.BatchNorm2d(channels),
            )

        self.conv = nn.Conv2d(channels, channels, (temporal_window_size,1), (stride,1), padding)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + res_block + res_module)

        return x


class TemporalBottleneckBlock(nn.Module):
    """
        TemporalConv_Res_Bottleneck
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, channels, temporal_window_size, stride=1, residual=False, reduction=4,get_res=False, tcn_stride=False):
        super(TemporalBottleneckBlock, self).__init__()
        tcn_stride =False
        padding = ((temporal_window_size - 1) // 2, 0)
        inter_channels = channels // reduction
        if get_res:
            if tcn_stride:
                stride =2
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (2,1)),
                nn.BatchNorm2d(channels),
            )
            tcn_stride= True
        else:
            if not residual:
                self.residual = lambda x: 0
            elif stride == 1:
                self.residual = lambda x: x
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(channels, channels, 1, (2,1)),
                    nn.BatchNorm2d(channels),
                )
                tcn_stride= True

        self.conv_down = nn.Conv2d(channels, inter_channels, 1)
        self.bn_down = nn.BatchNorm2d(inter_channels)
        if tcn_stride:
            stride=2
        self.conv = nn.Conv2d(inter_channels, inter_channels, (temporal_window_size,1), (stride,1), padding)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.conv_up = nn.Conv2d(inter_channels, channels, 1)
        self.bn_up = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block + res_module)
        return x



class SpatialGraphConv(nn.Module):
    """
        SpatialGraphConv_Basic_Block
        Arxiv: https://arxiv.org/abs/1801.07455
        Github: https://github.com/yysijie/st-gcn
    """
    def __init__(self, in_channels, out_channels, max_graph_distance):
        super(SpatialGraphConv, self).__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = max_graph_distance + 1

        # weights of different spatial classes
        self.gcn = nn.Conv2d(in_channels, out_channels*self.s_kernel_size, 1)

    def forward(self, x, A):

        # numbers in same class have same weight
        x = self.gcn(x)

        # divide nodes into different classes
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v).contiguous()

        # spatial graph convolution
        x = torch.einsum('nkctv,kvw->nctw', (x, A[:self.s_kernel_size])).contiguous()

        return x

class SpatialBasicBlock(nn.Module):
    """
        SpatialGraphConv_Res_Block
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, in_channels, out_channels, max_graph_distance, residual=False,reduction=0):
        super(SpatialBasicBlock, self).__init__()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv = SpatialGraphConv(in_channels, out_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res_block = self.residual(x)

        x = self.conv(x, A)
        x = self.bn(x)
        x = self.relu(x + res_block)

        return x

class SpatialBottleneckBlock(nn.Module):
    """
        SpatialGraphConv_Res_Bottleneck
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """

    def __init__(self, in_channels, out_channels, max_graph_distance, residual=False, reduction=4):
        super(SpatialBottleneckBlock, self).__init__()

        inter_channels = out_channels // reduction

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv_down = nn.Conv2d(in_channels, inter_channels, 1)
        self.bn_down = nn.BatchNorm2d(inter_channels)
        self.conv = SpatialGraphConv(inter_channels, inter_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.conv_up = nn.Conv2d(inter_channels, out_channels, 1)
        self.bn_up = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res_block = self.residual(x)

        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)

        x = self.conv(x, A)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block)

        return x

class SpatialAttention(nn.Module):
    """
    This class implements Spatial Transformer. 
    Function adapted from: https://github.com/leaderj1001/Attention-Augmented-Conv2d
    """
    def __init__(self, in_channels, out_channel, A, num_point, dk_factor=0.25, kernel_size=1, Nh=8, num=4, stride=1):
        super(SpatialAttention, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dk = int(dk_factor * out_channel)
        self.dv = int(out_channel)
        self.num = num
        self.Nh = Nh
        self.num_point=num_point
        self.A = A[0] + A[1] + A[2]
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size,
                                    stride=stride,
                                    padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

    def forward(self, x):
        # Input x
        # (batch_size, channels, 1, joints)
        B, _, T, V = x.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, dvh or dkh, joints)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v obtained by doing 2D convolution on the input (q=XWq, k=XWk, v=XWv)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)

        # Calculate the scores, obtained by doing q*k
        # (batch_size, Nh, joints, dkh)*(batch_size, Nh, dkh, joints) =  (batch_size, Nh, joints,joints)
        # The multiplication can also be divided (multi_matmul) in case of space problems

        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)

        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, joints, dvh)
        # weights*V
        # (batch, Nh, joints, joints)*(batch, Nh, joints, dvh)=(batch, Nh, joints, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))

        attn_out = torch.reshape(attn_out, (B, self.Nh, T, V, self.dv // self.Nh))

        attn_out = attn_out.permute(0, 1, 4, 2, 3)

        # combine_heads_2d, combine heads only after having calculated each Z separately
        # (batch, Nh*dv, 1, joints)
        attn_out = self.combine_heads_2d(attn_out)

        # Multiply for W0 (batch, out_channels, 1, joints) with out_channels=dv
        attn_out = self.attn_out(attn_out)
        return attn_out

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        # T=1 in this case, because we are considering each frame separately
        N, _, T, V = qkv.size()

        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q*(dkh ** -0.5)
        flat_q = torch.reshape(q, (N, Nh, dkh, T * V))
        flat_k = torch.reshape(k, (N, Nh, dkh, T * V))
        flat_v = torch.reshape(v, (N, Nh, dv // self.Nh, T * V))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        B, channels, T, V = x.size()
        ret_shape = (B, Nh, channels // Nh, T, V)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, T, V = x.size()
        ret_shape = (batch, Nh * dv, T, V)
        return torch.reshape(x, ret_shape)

from einops import rearrange
class ParallelBN1d(nn.Module):
    def __init__(self, parts_num, in_channels, **kwargs):
        super(ParallelBN1d, self).__init__()
        self.parts_num = parts_num
        self.bn1d = nn.BatchNorm1d(in_channels * parts_num, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, p]
        '''
        x = rearrange(x, 'n c p -> n (c p)')
        x = self.bn1d(x)
        x = rearrange(x, 'n (c p) -> n c p', p=self.parts_num)
        return x
    

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlockP3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,  downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockP3D, self).__init__()
        if norm_layer is None:
            norm_layer2d = nn.BatchNorm2d
            norm_layer3d = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.relu  = nn.ReLU(inplace=True)
        
        self.conv1 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(inplanes, planes, stride), 
                norm_layer2d(planes), 
                nn.ReLU(inplace=True)
            )
        )

        self.conv2 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(planes, planes), 
                norm_layer2d(planes), 
            )
        )

        self.shortcut3d = nn.Conv3d(planes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False)
        self.sbn        = norm_layer3d(planes)

        self.downsample = downsample

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        identity = x

        out = self.conv1(x)
        out = self.relu(out + self.sbn(self.shortcut3d(out)))
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=[1, 1, 1],  downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        assert stride[0] in [1, 2, 3]
        if stride[0] in [1, 2]: 
            tp = 1
        else:
            tp = 0
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=[tp, 1, 1], bias=False)
        self.bn1   = norm_layer(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=[1, 1, 1], padding=[1, 1, 1], bias=False)
        self.bn2   = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



# Modified from https://github.com/autonomousvision/unimatch
class FlowFunc(nn.Module):
    def __init__(self, radius=3, padding_mode='zeros'): 
        super(FlowFunc, self).__init__()
        self.radius = radius
        self.padding_mode = padding_mode

    def coords_grid(self, n, h, w, device=None):
        assert device is not None
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [h, w]
        stacks = [x, y]
        grid = torch.stack(stacks, dim=0).float()  # [2, h, w]
        grid = grid[None].repeat(n, 1, 1, 1)  # [n, 2, h, w]
        return grid.to(device)
    
    def generate_window_grid(self, h_min, h_max, w_min, w_max, len_h, len_w, device=None):
        assert device is not None
        x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w),
                    torch.linspace(h_min, h_max, len_h)],
                )
        grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [h, w, 2]
        return grid.to(device)
    
    def normalize_coords(self, coords, h, w):
        # coords: [n*s, h, w, 2]
        c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
        return (coords - c) / c  # [-1, 1]
    
    def forward(self, feature0, feature1):
        '''
            features: [n, c, s, h, w]
        '''
        n = feature0.size(0)
        s = feature1.size(2)
        feature0 = rearrange(feature0, 'n c s h w -> (n s) c h w')
        feature1 = rearrange(feature1, 'n c s h w -> (n s) c h w')
        
        n_s, c, h, w = feature1.size()
        coords_init = self.coords_grid(n_s, h, w, feature1.device) # [n*s, 2, h, w]
        coords = coords_init.view(n_s, 2, -1).permute(0, 2, 1)  # [n*s, h*w, 2]

        local_h = 2 * self.radius + 1
        local_w = 2 * self.radius + 1
        
        window_grid = self.generate_window_grid(-self.radius, self.radius, -self.radius, self.radius,
                            local_h, local_w, device=feature0.device)  # [2r+1, 2r+1, 2]
        window_grid = window_grid.reshape(-1, 2).repeat(n_s, 1, 1, 1) # [n*s, 1, (2r+1)**2, 2]
        sample_coords = coords.unsqueeze(-2) + window_grid  # [n*s, h*w, (2r+1)**2, 2]
        
        sample_coords_softmax = sample_coords
        # exclude coords that are out of image space
        valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)  # [n*s, h*w, (2r+1)**2]
        valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)  # [n*s, h*w, (2r+1)**2]
        valid   = valid_x & valid_y  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax
        
        # normalize coordinates to [-1, 1]
        sample_coords_norm = self.normalize_coords(sample_coords, h, w)  # [-1, 1]
        window_feature = F.grid_sample(feature1.contiguous(), sample_coords_norm.contiguous(),
                            padding_mode=self.padding_mode, align_corners=True
                        ).permute(0, 2, 1, 3).contiguous()  # [n*s, h*w, c, (2r+1)**2]
        feature0_view = feature0.permute(0, 2, 3, 1).contiguous().view(n_s, h * w, 1, c)  # [n*s, h*w, 1, c]

        corr = torch.matmul(feature0_view, window_feature).view(n_s, h * w, -1) / (c ** 0.5)  # [n*s, h*w, (2r+1)**2]

        # mask invalid locations
        corr[~valid] = float("-inf")
        # corr[~valid] = -1e9

        prob = F.softmax(corr, -1)  # [n*s, h*w, (2r+1)**2]

        correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax).squeeze(-2).view(
            n_s, h, w, 2).permute(0, 3, 1, 2)  # [n*s, 2, h, w]

        flow = correspondence - coords_init # [n*s, 2, h, w]
        flow = rearrange(flow, '(n s) c h w -> n c s h w', n=n)
        correspondence = rearrange(correspondence, '(n s) c h w -> n c s h w', n=n)

        return flow
