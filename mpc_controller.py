"""
Módulo 3: Controlador Predictivo por Modelos (MPC)
Calcula la actuación en las bobinas magnéticas para estabilizar el plasma.
Autor: Leo Ruiz Diaz
"""

import numpy as np


class MagneticMPCController:

  def __init__(
      self,
      num_coils=8,
      max_current_ramp_rate=5000.0,
      safety_factor_q_min=2.05,
  ):
    """Inicializa los límites de operación de las bobinas de control magnético.

    max_current_ramp_rate: Límite dI/dt (Amperios/segundo) para evitar el quench.
    safety_factor_q_min: Factor de seguridad mínimo en la frontera del plasma
    (q >= 2.05).
    """
    self.num_coils = num_coils
    self.max_dI_dt = max_current_ramp_rate
    self.q_min = safety_factor_q_min

  def calculate_lorentz_inversion(
      self, target_force_vector: np.ndarray
  ) -> np.ndarray:
    """Calcula las corrientes requeridas en las bobinas invirtiendo la fuerza de Lorentz.

    F_target = I x B -> I_target = F_target / B_toroidal
    """
    # Matriz sintética de acoplamiento inductivo entre bobinas y superficie del plasma
    coupling_matrix = np.eye(self.num_coils) * 1.5 + 0.1

    # Inversión de matriz para resolver corrientes objetivo
    coil_currents = np.linalg.solve(coupling_matrix, target_force_vector)
    return coil_currents

  def compute_control_action(
      self, predicted_beta: float, mhd_amplitude: float, current_state: dict
  ) -> dict:
    """Calcula la acción de control en tiempo real reduciendo el riesgo detectado por la PINN."""
    current_coil_state = current_state.get(
        "coil_currents", np.zeros(self.num_coils)
    )

    # Vector de fuerza correctiva proporcional a la desviación de Beta e inestabilidad MHD
    force_correction = np.full(
        self.num_coils, (predicted_beta * 10.0) + (mhd_amplitude * 20.0)
    )

    # Inversión magnética
    target_currents = self.calculate_lorentz_inversion(force_correction)

    # Aplicar restricción de rampa máxima dI/dt (Prevención de Quench)
    delta_current = target_currents - current_coil_state
    clamped_delta = np.clip(delta_current, -self.max_dI_dt, self.max_dI_dt)

    final_coil_commands = current_coil_state + clamped_delta

    # Estimación del factor de seguridad q ajustado
    estimated_q = max(
        1.5, self.q_min + (0.5 - mhd_amplitude) - (predicted_beta * 2.0)
    )

    return {
        "coil_current_commands": final_coil_commands.tolist(),
        "applied_ramp_rates": clamped_delta.tolist(),
        "estimated_q_factor": float(estimated_q),
        "control_status": (
            "OPTIMAL" if estimated_q >= self.q_min else "WARN_LOW_Q"
        ),
    }
