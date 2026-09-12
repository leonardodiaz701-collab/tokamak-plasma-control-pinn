"""
Módulo 2: Motor Predictivo PINN (Physics-Informed Neural Network)
Calcula las ecuaciones magnetohidrodinámicas (MHD) y el límite de Troyon a 15ms.
Autor: Leo Ruiz Diaz
"""

import torch
import torch.nn as nn


class PINNPredictor(nn.Module):

  def __init__(
      self, input_dim=6, hidden_dim=64, troyon_limit=0.035, forecast_ms=15
  ):
    super(PINNPredictor, self).__init__()
    self.troyon_limit = troyon_limit
    self.forecast_ms = forecast_ms

    # Red neuronal para calcular la evolución del plasma
    self.network = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 3),  # Salidas: [Beta_futuro, Modo_MHD, Incertidumbre]
    )

  def forward(self, state_vector):
    """Evalúa el estado futuro del plasma a 15ms."""
    output = self.network(state_vector)
    beta_pred = output[..., 0]
    mhd_mode_amplitude = output[..., 1]
    uncertainty = torch.sigmoid(output[..., 2])

    return beta_pred, mhd_mode_amplitude, uncertainty

  def compute_physics_loss(self, state_vector, beta_pred, mhd_amplitude):
    """Aplica la pérdida física basada en los límites de Troyon e inestabilidades MHD."""
    # Penalización por superación del límite de Beta de Troyon
    troyon_violation = torch.relu(beta_pred - self.troyon_limit)
    loss_troyon = torch.mean(troyon_violation**2)

    # Penalización por crecimiento de ondas MHD
    loss_mhd = torch.mean(torch.relu(mhd_amplitude - 0.05) ** 2)

    return loss_troyon + loss_mhd

  def evaluate_disruption_risk(self, state_vector):
    """Diagnóstico en tiempo real para determinar el riesgo de disrupción."""
    self.eval()
    with torch.no_grad():
      beta_pred, mhd_amp, uncertainty = self.forward(state_vector)

      is_risk = (beta_pred > self.troyon_limit) or (mhd_amp > 0.05)

      return {
          "disruption_risk": bool(is_risk),
          "forecast_window_ms": self.forecast_ms,
          "predicted_beta": float(beta_pred.item()),
          "mhd_mode_amp": float(mhd_amp.item()),
          "confidence_score": float(1.0 - uncertainty.item()),
      }

