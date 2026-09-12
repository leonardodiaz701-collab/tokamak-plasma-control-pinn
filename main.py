"""
Orquestador Principal del Sistema de Control de Plasma para Tokamak (Closed-Loop)
Integra Ingesta, Predicción PINN, Control MPC y Failsafe SPI.
Autor: Leo Ruiz Diaz
"""

import numpy as np
import torch
from gauss_projection import GaussProjectionLayer
from pinn_model import PINNPredictor
from mpc_controller import MagneticMPCController
from failsafe_mitigation import EmergencyFailsafeSystem


def run_control_cycle(sensor_data_tensor, system_state):
  # 1. Módulo 1: Reconciliación magnética div(B) = 0
  gauss_layer = GaussProjectionLayer()
  clean_b_field = gauss_layer.project_valid_field(sensor_data_tensor)

  # 2. Módulo 2: Predicción PINN a 15ms
  pinn = PINNPredictor()
  # Formato de vector plano para evaluación de la PINN
  input_features = clean_b_field.view(clean_b_field.size(0), -1)[:, :6]
  risk_assessment = pinn.evaluate_disruption_risk(input_features)

  # 3. Módulo 4: Evaluación Failsafe de Emergencia
  failsafe = EmergencyFailsafeSystem()
  emergency_status = failsafe.evaluate_emergency_triggers(
      predicted_beta=risk_assessment["predicted_beta"],
      mhd_amplitude=risk_assessment["mhd_mode_amp"],
      confidence_score=risk_assessment["confidence_score"],
  )

  # 4. Decisión de Actuación: Control MPC o Interrupción por Emergencia
  if emergency_status["override_mpc_control"]:
    spi_action = failsafe.execute_spi_sequence(emergency_status)
    return {
        "status": "EMERGENCY_SHUTDOWN",
        "action": spi_action,
        "telemetry": risk_assessment,
    }

  # Módulo 3: Actuación MPC Magnética Nominal
  mpc = MagneticMPCController()
  control_action = mpc.compute_control_action(
      predicted_beta=risk_assessment["predicted_beta"],
      mhd_amplitude=risk_assessment["mhd_mode_amp"],
      current_state=system_state,
  )

  return {
      "status": "NOMINAL_CONTROL",
      "action": control_action,
      "telemetry": risk_assessment,
  }


if __name__ == "__main__":
  # Simulación de un paso de control con datos sintéticos
  print(
      "Iniciando ciclo de prueba del Sistema de Control de Plasma (Closed-Loop)..."
  )

  dummy_b_field = torch.randn(1, 3, 10, 10, 10, requires_grad=True)
  dummy_system_state = {"coil_currents": np.zeros(8)}

  result = run_control_cycle(dummy_b_field, dummy_system_state)

  print(f"\nEstado del Sistema: {result['status']}")
  print(f"Riesgo de Disrupción: {result['telemetry']['disruption_risk']}")
  print(f"Acción Ejecutada: {result['action']}")
