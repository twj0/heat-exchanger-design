"""
Transformer-based Feature Extractor for SAC Policy.

Innovation C: Using attention mechanism to capture temporal dependencies
in price and carbon intensity forecasts for better decision making.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Type
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor for building control.
    
    This extractor uses self-attention to process:
    1. Current state features (temperature, power, etc.)
    2. Forecast sequences (price, carbon intensity)
    
    The attention mechanism allows the agent to learn which future
    timesteps are most relevant for current decisions.
    
    Args:
        observation_space: Gymnasium observation space
        features_dim: Output feature dimension
        n_forecast_steps: Number of forecast timesteps in observation
        d_model: Transformer model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer encoder layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        n_forecast_steps: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)
        
        self.n_forecast_steps = n_forecast_steps
        self.d_model = d_model
        
        # Calculate input dimensions
        obs_dim = observation_space.shape[0]
        # Forecast: price (N) + carbon (N) = 2N
        self.forecast_dim = 2 * n_forecast_steps
        self.state_dim = obs_dim - self.forecast_dim
        
        # State encoder (MLP for current state features)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Forecast embedding (project each timestep to d_model)
        # Each timestep has 2 features: price and carbon
        self.forecast_embedding = nn.Linear(2, d_model)
        
        # Positional encoding for forecast sequence
        self.pos_encoding = self._create_positional_encoding(
            n_forecast_steps, d_model
        )
        
        # Transformer encoder for forecast processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        
        # Attention pooling to aggregate forecast features
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=1,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )
        
    def _create_positional_encoding(
        self, 
        seq_len: int, 
        d_model: int
    ) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, seq_len, d_model)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer extractor.
        
        Args:
            observations: Batch of observations (batch_size, obs_dim)
            
        Returns:
            Extracted features (batch_size, features_dim)
        """
        batch_size = observations.shape[0]
        
        # Split observation into state and forecast
        state = observations[:, :self.state_dim]
        forecast_flat = observations[:, self.state_dim:]
        
        # Encode current state
        state_features = self.state_encoder(state)  # (batch, d_model)
        
        # Reshape forecast: (batch, 2*N) -> (batch, N, 2)
        # First N values are prices, next N are carbon factors
        prices = forecast_flat[:, :self.n_forecast_steps]
        carbons = forecast_flat[:, self.n_forecast_steps:]
        forecast = torch.stack([prices, carbons], dim=-1)  # (batch, N, 2)
        
        # Embed forecast sequence
        forecast_embedded = self.forecast_embedding(forecast)  # (batch, N, d_model)
        
        # Add positional encoding
        pos_enc = self.pos_encoding.to(observations.device)
        forecast_embedded = forecast_embedded + pos_enc
        
        # Process through transformer
        forecast_encoded = self.transformer(forecast_embedded)  # (batch, N, d_model)
        
        # Attention pooling to get single forecast representation
        query = self.query.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        forecast_pooled, _ = self.attention_pool(
            query, forecast_encoded, forecast_encoded
        )
        forecast_features = forecast_pooled.squeeze(1)  # (batch, d_model)
        
        # Fuse state and forecast features
        combined = torch.cat([state_features, forecast_features], dim=-1)
        output = self.fusion(combined)
        
        return output


class SimpleMLP(BaseFeaturesExtractor):
    """
    Simple MLP feature extractor for baseline comparison.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


class TemporalTransformerExtractor(BaseFeaturesExtractor):
    """
    Enhanced Transformer with temporal attention for time-series building data.
    
    This extractor is specifically designed for building control with:
    1. Separate encoders for different feature types
    2. Cross-attention between state and forecast
    3. Learnable time embeddings
    4. Layer normalization for training stability
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        n_forecast_steps: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        super().__init__(observation_space, features_dim)
        
        self.n_forecast_steps = n_forecast_steps
        self.d_model = d_model
        self.use_cross_attention = use_cross_attention
        
        obs_dim = observation_space.shape[0]
        self.forecast_dim = 2 * n_forecast_steps
        self.state_dim = obs_dim - self.forecast_dim
        
        # State feature groups
        self.n_zone_temps = 5
        self.n_outdoor = 2  # temp, solar
        self.n_power = 1
        self.n_time = 2
        self.n_grid = 2  # price, carbon
        
        # Zone temperature encoder
        self.zone_encoder = nn.Sequential(
            nn.Linear(self.n_zone_temps, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        
        # Context encoder (outdoor, power, time, grid)
        context_dim = self.state_dim - self.n_zone_temps
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        
        # Forecast embedding with temporal encoding
        self.forecast_embedding = nn.Linear(2, d_model)
        self.temporal_embedding = nn.Embedding(n_forecast_steps, d_model)
        
        # Self-attention for forecasts
        self.forecast_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.forecast_norm = nn.LayerNorm(d_model)
        
        # Cross-attention: state attends to forecasts
        if use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(d_model)
        
        # Final fusion
        fusion_input_dim = d_model * (3 if use_cross_attention else 3)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, features_dim),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Split observation
        state = observations[:, :self.state_dim]
        forecast_flat = observations[:, self.state_dim:]
        
        # Extract state components
        zone_temps = state[:, :self.n_zone_temps]
        context = state[:, self.n_zone_temps:]
        
        # Encode state
        zone_features = self.zone_encoder(zone_temps)  # (batch, d_model)
        context_features = self.context_encoder(context)  # (batch, d_model)
        
        # Process forecasts
        prices = forecast_flat[:, :self.n_forecast_steps]
        carbons = forecast_flat[:, self.n_forecast_steps:]
        forecast = torch.stack([prices, carbons], dim=-1)  # (batch, N, 2)
        
        # Embed forecasts with temporal position
        forecast_embedded = self.forecast_embedding(forecast)  # (batch, N, d_model)
        positions = torch.arange(self.n_forecast_steps, device=observations.device)
        temporal_emb = self.temporal_embedding(positions)  # (N, d_model)
        forecast_embedded = forecast_embedded + temporal_emb.unsqueeze(0)
        
        # Self-attention on forecasts
        forecast_attn, _ = self.forecast_attention(
            forecast_embedded, forecast_embedded, forecast_embedded
        )
        forecast_attn = self.forecast_norm(forecast_attn + forecast_embedded)
        
        # Pool forecasts (mean pooling)
        forecast_pooled = forecast_attn.mean(dim=1)  # (batch, d_model)
        
        # Cross-attention if enabled
        if self.use_cross_attention:
            # State as query, forecasts as key/value
            state_combined = (zone_features + context_features).unsqueeze(1)  # (batch, 1, d_model)
            cross_attn, _ = self.cross_attention(
                state_combined, forecast_attn, forecast_attn
            )
            cross_attn = self.cross_norm(cross_attn + state_combined)
            cross_features = cross_attn.squeeze(1)  # (batch, d_model)
            
            # Fuse all features
            combined = torch.cat([zone_features, context_features, cross_features], dim=-1)
        else:
            combined = torch.cat([zone_features, context_features, forecast_pooled], dim=-1)
        
        output = self.fusion(combined)
        return output


def create_policy_kwargs(
    extractor_type: str = "transformer",
    **kwargs
) -> Dict:
    """
    Create policy_kwargs for Stable-Baselines3 SAC.
    
    Args:
        extractor_type: "transformer", "temporal_transformer", or "mlp"
        **kwargs: Additional arguments for extractor
        
    Returns:
        policy_kwargs dict for SAC initialization
    """
    extractors = {
        "transformer": TransformerExtractor,
        "temporal_transformer": TemporalTransformerExtractor,
        "mlp": SimpleMLP,
    }
    
    extractor_class = extractors.get(extractor_type, TransformerExtractor)
    
    return {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": kwargs,
    }
