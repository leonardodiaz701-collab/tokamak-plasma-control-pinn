# tokamak-plasma-control-pinn

## Declaración de Autoría y Propiedad Intelectual

- **Autor e Inventor:** Leo Ruiz Diaz
- **Registro de Obra / Proyecto:** Inscrito en Safe Creative bajo la categoría de Proyecto Técnico.
- **Licencia de Código Fuente:** MIT License (Permite el uso, modificación y distribución pública manteniendo la atribución legal obligatoria del autor).
- **Licencia de Arquitectura Técnica:** Creative Commons Attribution 4.0 International (CC BY 4.0).

Este repositorio constituye la implementación en software de la arquitectura matemática y física desarrollada por Leo Ruiz Diaz para sistemas de control de plasma en reactores Tokamak aplicados a energía sostenible y desalinización.

## Módulos del Sistema

- `gauss_projection.py`: Ingesta y limpieza de datos magnéticos (div B = 0).
- `pinn_model.py`: Motor predictivo mediante Red Neuronal Informada por la Física (15ms).
- `mpc_controller.py`: Control Predictivo por Modelos sobre las bobinas magnéticas.
- `failsafe_mitigation.py`: Sistema de mitigación de emergencia (SPI/MGI).
- `main.py`: Orquestador principal en bucle cerrado (closed-loop).
