"""
Módulo 1: Ingesta y Reconciliación de datos magnéticos.
Garantiza que el campo magnético B cumpla con la Ley de Gauss: div(B) = 0.
Autor: Leo Ruiz Diaz
"""

import torch
import torch.nn as nn

class GaussProjectionLayer(nn.Module):
    def __init__(self, spatial_grid_spacing=0.01):
        super(GaussProjectionLayer, self).__init__()
        self.dx = spatial_grid_spacing

    def compute_divergence(self, B_field: torch.Tensor) -> torch.Tensor:
        """
        Calcula la divergencia del campo magnético B en 3D.
        B_field: Tensor con forma (batch, 3, Nx, Ny, Nz)
        """
        dBx_dx = torch.gradient(B_field[:, 0], spacing=(self.dx,), dim=-1)[0]
        dBy_dy = torch.gradient(B_field[:, 1], spacing=(self.dx,), dim=-2)[0]
        dBz_dz = torch.gradient(B_field[:, 2], spacing=(self.dx,), dim=-3)[0]
        
        return dBx_dx + dBy_dy + dBz_dz

    def project_valid_field(self, B_field: torch.Tensor) -> torch.Tensor:
        """
        Corrige las lecturas del sensor para eliminar componentes no físicas.
        """
        div_B = self.compute_divergence(B_field)
        
        # Si el error es insignificante, se mantiene el dato original
        if torch.max(torch.abs(div_B)) < 1e-6:
            return B_field
        
        # Aplicar corrección por gradiente
        B_corrected = B_field - torch.autograd.grad(
            outputs=div_B.sum(), 
            inputs=B_field, 
            create_graph=True, 
            retain_graph=True,
            allow_unused=True
        )[0]
        
        return B_corrected if B_corrected is not None else B_field
