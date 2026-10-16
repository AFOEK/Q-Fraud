import math

import torch
from torch import nn


class SixQubitReuploadingVQC(nn.Module):
    def __init__(self, layers=3):
        super().__init__()

        self.layers = layers

        self.ry_bias = nn.Parameter(
            torch.randn(layers, 6) * 0.08
        )
        self.rz_bias = nn.Parameter(
            torch.randn(layers, 6) * 0.08
        )

        self.ry_scale = nn.Parameter(
            torch.ones(layers, 6)
        )
        self.rz_scale = nn.Parameter(
            torch.ones(layers, 6)
        )

        idx = torch.arange(64)
        permutations = []

        for control in range(6):
            target = (control + 1) % 6

            mask = (
                idx & (1 << (5 - control))
            ) != 0

            permutations.append(
                torch.where(
                    mask,
                    idx ^ (1 << (5 - target)),
                    idx
                )
            )

        self.register_buffer(
            "cnot_permutations",
            torch.stack(permutations),
            persistent=False
        )

        bits = torch.stack([
            (idx >> (5 - q)) & 1
            for q in range(6)
        ])

        z = 1.0 - 2.0 * bits.float()
        zz = z * z.roll(-1, dims=0)

        self.register_buffer(
            "observable_signs",
            torch.cat((z, zz), dim=0),
            persistent=False
        )

    @staticmethod
    def rotation_y(state, angles, qubit):
        batch = state.shape[0]

        state = state.reshape(
            batch,
            1 << qubit,
            2,
            1 << (5 - qubit)
        )

        a = state[:, :, 0, :]
        b = state[:, :, 1, :]

        c = torch.cos(angles / 2).view(
            batch, 1, 1
        )
        s = torch.sin(angles / 2).view(
            batch, 1, 1
        )

        return torch.stack(
            (
                c * a - s * b,
                s * a + c * b
            ),
            dim=2
        ).reshape(batch, 64)

    @staticmethod
    def rotation_z(state, angles, qubit):
        batch = state.shape[0]

        state = state.reshape(
            batch,
            1 << qubit,
            2,
            1 << (5 - qubit)
        )

        a = state[:, :, 0, :] * torch.exp(
            -0.5j * angles
        ).view(batch, 1, 1)

        b = state[:, :, 1, :] * torch.exp(
            0.5j * angles
        ).view(batch, 1, 1)

        return torch.stack(
            (a, b),
            dim=2
        ).reshape(batch, 64)

    def forward(self, angles):
        batch = angles.shape[0]

        state = torch.zeros(
            (batch, 64),
            dtype=torch.complex64,
            device=angles.device
        )
        state[:, 0] = 1

        for layer in range(self.layers):
            for q in range(6):
                ry = (
                    angles[:, q]
                    * self.ry_scale[layer, q]
                    + self.ry_bias[layer, q]
                )

                rz = (
                    angles[:, q]
                    * self.rz_scale[layer, q]
                    + self.rz_bias[layer, q]
                )

                state = self.rotation_y(
                    state, ry, q
                )
                state = self.rotation_z(
                    state, rz, q
                )

            for q in range(6):
                state = state.index_select(
                    1,
                    self.cnot_permutations[q]
                )

        probabilities = (
            state.real.square()
            + state.imag.square()
        )

        return (
            probabilities
            @ self.observable_signs.T
        )


class QGuardV5(nn.Module):
    def __init__(self, tab_dim, kind="quantum"):
        super().__init__()

        if kind not in (
            "quantum",
            "classical",
            "classical_strong"
        ):
            raise ValueError(f"Unknown kind: {kind}")

        self.kind = kind

        # Preserve the original V4 feature encoders.
        self.tab = nn.Sequential(
            nn.Linear(tab_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU()
        )

        self.gru = nn.GRU(
            input_size=3,
            hidden_size=24,
            batch_first=True
        )

        self.graph_proj = nn.Linear(8, 32)

        self.sage = nn.Sequential(
            nn.Linear(66, 32),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(120, 64),
            nn.ReLU()
        )

        # Shared six-dimensional bottleneck.
        self.angle_proj = nn.Linear(64, 6)

        if kind == "quantum":
            self.head = SixQubitReuploadingVQC(
                layers=3
            )

        elif kind == "classical":
            # Similar head parameter count to VQC.
            self.head = nn.Sequential(
                nn.Linear(6, 4),
                nn.Tanh(),
                nn.Linear(4, 12),
                nn.Tanh()
            )

        else:
            # Stronger classical capacity control.
            self.head = nn.Sequential(
                nn.Linear(6, 32),
                nn.Tanh(),
                nn.Linear(32, 12),
                nn.Tanh()
            )

        # No 64-dimensional bypass.
        self.out = nn.Linear(12, 3)

    def forward(self, batch):
        tab = self.tab(batch["tab"])

        _, hidden = self.gru(batch["seq"])
        temporal = hidden[-1]

        self_emb = torch.relu(
            self.graph_proj(
                batch["graph_self"]
            )
        )

        device_emb = torch.relu(
            self.graph_proj(
                batch["graph_device"]
            )
        )

        token_emb = torch.relu(
            self.graph_proj(
                batch["graph_token"]
            )
        )

        links = batch["graph_links"]

        dm = (
            links[:, 0:1] > 0
        ).to(links.dtype)

        tm = (
            links[:, 1:2] > 0
        ).to(links.dtype)

        neighbours = (
            device_emb * dm
            + token_emb * tm
        ) / (dm + tm).clamp_min(1)

        graph = self.sage(
            torch.cat(
                (self_emb, neighbours, links),
                dim=-1
            )
        )

        fused = self.fusion(
            torch.cat(
                (tab, temporal, graph),
                dim=-1
            )
        )

        angles = math.pi * torch.tanh(
            self.angle_proj(fused)
        )

        transformed = self.head(angles)

        return self.out(transformed)


if __name__ == "__main__":
    for kind in (
        "classical",
        "classical_strong",
        "quantum"
    ):
        model = QGuardV5(51, kind)

        batch = {
            "tab": torch.randn(8, 51),
            "seq": torch.randn(8, 3, 3),
            "graph_self": torch.randn(8, 8),
            "graph_device": torch.randn(8, 8),
            "graph_token": torch.randn(8, 8),
            "graph_links": torch.randint(
                0, 2, (8, 2)
            ).float()
        }

        output = model(batch)

        assert output.shape == (8, 3)
        assert torch.isfinite(output).all()

        output.sum().backward()

        assert all(
            p.grad is not None
            and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )

        print(
            f"[PASS] {kind}: "
            f"output={tuple(output.shape)}, "
            f"head_parameters="
            f"{sum(p.numel() for p in model.head.parameters())}"
        )
