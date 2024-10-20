import torch
import torch.nn as nn

from models.mtad_gat.modules import (
    ConvLayer,
    FeatureAttentionLayer,
    Forecasting_Model,
    GRULayer,
    ReconstructionModel,
    TemporalAttentionLayer,
)


class MTAD_GAT(nn.Module):
    """
    MTAD-GAT model class.
        Args:
            n_features: Number of input features
            window_size: Length of the input sequence
            out_dim: Number of features to output
            kernel_size: size of kernel to use in the 1-D convolution
            feat_gat_embed_dim: embedding dimension (output dimension of linear
                transformation)
                in feat-oriented GAT layer
            time_gat_embed_dim: embedding dimension (output dimension of linear
                transformation)
                in time-oriented GAT layer
            use_gatv2: whether to use the modified attention mechanism of GATv2 instead
                of standard GAT
            gru_n_layers: number of layers in the GRU layer
            gru_hid_dim: hidden dimension in the GRU layer
            forecast_n_layers: number of layers in the FC-based Forecasting Model
            forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
            recon_n_layers: number of layers in the GRU-based Reconstruction Model
            recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
            dropout: dropout rate
            alpha: negative slope used in the leaky rely activation function
    """

    def __init__(
        self,
        n_features,
        out_dim,
        window_size,
        kernel_size=7,
        use_gatv2=True,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        gru_n_layers=1,
        gru_hid_dim=300,
        forecast_n_layers=3,
        forecast_hid_dim=300,
        recon_n_layers=1,
        recon_hid_dim=300,
        dropout=0.3,
        alpha=0.2,
    ):
        super(MTAD_GAT, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(
            n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2
        )
        self.temporal_gat = TemporalAttentionLayer(
            n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2
        )
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(
            gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout
        )
        self.recon_model = ReconstructionModel(
            window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout
        )

    def forward(self, x):
        """
        Model forward pass.

        Args:
            x: input tensor of shape (b, n, k):
                b - batch size, n - window size, k - number of features
        returns:
            - Predictions tensor of shape (b, out_dim)
            - Reconstruction tensor of shape (b, n, out_dim)
        """
        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)  # Hidden state for last timestamp

        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end)

        return predictions, recons
