"""
Módulo 4: Sistema de Mitigación de Emergencia (Failsafe / SPI)
Capa de seguridad de última instancia para disrupciones e inyección de gas/pellets.
Autor: Leo Ruiz Diaz
"""

import time


class EmergencyFailsafeSystem:

  def __init__(
      self,
      uncertainty_threshold=0.35,
      troyon_hard_limit=0.045,
      mhd_hard_limit=0.12,
  ):
    """Límites críticos de seguridad operacional para disparo de emergencia."""
    self.max_uncertainty = uncertainty_threshold
    self.hard_troyon_limit = troyon_hard_limit
    self.hard_mhd_limit = mhd_hard_limit

  def evaluate_emergency_triggers(
      self,
      predicted_beta: float,
      mhd_amplitude: float,
      confidence_score: float,
  ) -> dict:
    """Evalúa en tiempo real si el plasma entra en zona irrecuperable."""
    uncertainty = 1.0 - confidence_score

    # Condición 1: Superación de límites físicos críticos
    physical_hard_violation = (predicted_beta >= self.hard_troyon_limit) or (
        mhd_amplitude >= self.hard_mhd_limit
    )

    # Condición 2: Alta incertidumbre en el modelo predictivo
    uncertainty_trigger = uncertainty >= self.max_uncertainty

    # Determinación de activación de mitigación activa (SPI / MGI)
    trigger_spi = physical_hard_violation or uncertainty_trigger

    reason = "NORMAL"
    if physical_hard_violation:
      reason = "CRITICAL_PHYSICAL_LIMIT_EXCEEDED"
    elif uncertainty_trigger:
      reason = "HIGH_MODEL_UNCERTAINTY"

    return {
      "spi_triggered": bool(trigger_spi),
      "trigger_reason": reason,
      "override_mpc_control": bool(trigger_spi),
      "timestamp": time.time(),
    }

  def execute_spi_sequence(self, trigger_info: dict) -> dict:
    """Simula la secuencia de disparo de Pellets Inyectados (Neón/Argón) para enfriamiento homogéneo por radiación."""
    if not trigger_info.get("spi_triggered", False):
      return {
          "action": "STANDBY",
          "valve_status": "CLOSED",
          "gas_mix": "NONE",
      }

    # Secuencia de mitigación activada
    return {
        "action": "SHATTERED_PELLET_INJECTION_ACTIVE",
        "valve_status": "OPEN",
        "pellet_composition": "Neon (80%) + Argon (20%)",
        "expected_thermal_quench_ms": 2.5,
        "radiation_cooling_status": "HOMOGENEOUS_DISSIPATION_ENABLED",
    }
