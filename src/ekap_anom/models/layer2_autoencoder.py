"""Layer 2 – Phase 2: Denoising Autoencoder → latent → Isolation Forest."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ekap_anom.config import AutoencoderConfig, IForestConfig
from ekap_anom.logging import get_logger

logger = get_logger(__name__)


class DenoisingAutoencoder(nn.Module):
    """PyTorch denoising autoencoder with variable hidden layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror)
        decoder_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, latent)."""
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation only."""
        return self.encoder(x)


class AELatentIForest:
    """Denoising AE → latent space → Isolation Forest scoring.

    Phase 2 model that combines reconstruction error with
    latent-space anomaly detection.
    """

    def __init__(
        self,
        ae_config: AutoencoderConfig | None = None,
        if_config: IForestConfig | None = None,
    ) -> None:
        self.ae_config = ae_config or AutoencoderConfig()
        self.if_config = if_config or IForestConfig()

        self.scaler: StandardScaler | None = None
        self.ae_model: DenoisingAutoencoder | None = None
        self.latent_iforest: IsolationForest | None = None
        self.feature_columns: list[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
    ) -> dict[str, float]:
        """Train the AE + latent IF pipeline.

        Returns training metrics (final loss, etc.).
        """
        self.feature_columns = feature_columns
        X = df[feature_columns].fillna(0).values.astype(np.float32)

        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Build AE
        input_dim = X_scaled.shape[1]
        self.ae_model = DenoisingAutoencoder(
            input_dim=input_dim,
            hidden_dims=self.ae_config.hidden_dims,
            latent_dim=self.ae_config.latent_dim,
            dropout=self.ae_config.dropout,
        ).to(self.device)

        # Train AE
        optimizer = torch.optim.Adam(
            self.ae_model.parameters(), lr=self.ae_config.learning_rate
        )
        criterion = nn.MSELoss()

        dataset = torch.tensor(X_scaled, dtype=torch.float32)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.ae_config.batch_size, shuffle=True
        )

        self.ae_model.train()
        final_loss = 0.0
        for epoch in range(self.ae_config.epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                batch = batch.to(self.device)
                # Add noise
                noise = torch.randn_like(batch) * self.ae_config.noise_factor
                noisy = batch + noise

                recon, _ = self.ae_model(noisy)
                loss = criterion(recon, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            final_loss = epoch_loss / max(len(dataloader), 1)
            if (epoch + 1) % 10 == 0:
                logger.info("ae_training", epoch=epoch + 1, loss=round(final_loss, 6))

        # Extract latent representations
        self.ae_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            latent = self.ae_model.encode(X_tensor).cpu().numpy()

        # Train IF on latent space
        self.latent_iforest = IsolationForest(
            n_estimators=self.if_config.n_estimators,
            max_samples=self.if_config.max_samples,
            contamination=self.if_config.contamination,
            random_state=self.if_config.random_state,
            n_jobs=-1,
        )
        self.latent_iforest.fit(latent)

        logger.info(
            "ae_latent_if_trained",
            n_samples=len(X),
            latent_dim=self.ae_config.latent_dim,
            final_loss=round(final_loss, 6),
        )
        return {"final_loss": final_loss}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Score requests. Returns anomaly scores in [0, 1]."""
        if self.ae_model is None or self.scaler is None or self.latent_iforest is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = df[self.feature_columns].fillna(0).values.astype(np.float32)
        X_scaled = self.scaler.transform(X)

        self.ae_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            recon, latent = self.ae_model(X_tensor)

            # Reconstruction error
            recon_error = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()

            # IF score on latent
            latent_np = latent.cpu().numpy()
            if_raw = self.latent_iforest.decision_function(latent_np)
            if_score = 1.0 / (1.0 + np.exp(if_raw))

        # Normalize recon error to [0, 1]
        recon_norm = 1.0 / (1.0 + np.exp(-recon_error + np.median(recon_error)))

        # Combine (configurable, default: use IF latent score)
        scores = (0.3 * recon_norm + 0.7 * if_score).astype(np.float32)
        return scores

    def save(self, path: str | Path) -> None:
        """Save model state."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "ae_config": self.ae_config,
            "if_config": self.if_config,
            "feature_columns": self.feature_columns,
            "scaler": self.scaler,
            "ae_state_dict": self.ae_model.state_dict() if self.ae_model else None,
            "ae_input_dim": self.ae_model.input_dim if self.ae_model else None,
            "latent_iforest": self.latent_iforest,
        }
        with open(p, "wb") as fh:
            pickle.dump(state, fh)

    @classmethod
    def load(cls, path: str | Path) -> "AELatentIForest":
        """Load model state."""
        with open(Path(path), "rb") as fh:
            state = pickle.load(fh)
        obj = cls(ae_config=state["ae_config"], if_config=state["if_config"])
        obj.feature_columns = state["feature_columns"]
        obj.scaler = state["scaler"]
        obj.latent_iforest = state["latent_iforest"]
        if state["ae_state_dict"] is not None:
            obj.ae_model = DenoisingAutoencoder(
                input_dim=state["ae_input_dim"],
                hidden_dims=obj.ae_config.hidden_dims,
                latent_dim=obj.ae_config.latent_dim,
                dropout=obj.ae_config.dropout,
            )
            obj.ae_model.load_state_dict(state["ae_state_dict"])
            obj.ae_model.to(obj.device)
        return obj
