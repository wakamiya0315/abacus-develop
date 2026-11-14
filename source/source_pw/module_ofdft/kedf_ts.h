#ifndef KEDF_TS_H
#define KEDF_TS_H

#ifdef __MLALGO

#include "source_base/matrix.h"
#include "source_basis/module_pw/pw_basis.h"

#include <string>
#include <torch/script.h>

/**
 * @brief Pauli KEDF given by a TorchScript module (of_kinetic = "ml-ts").
 *
 * The module file "kedf_ts.pt" in the working directory must provide
 *     forward(rho: Tensor[nx, ny, nz] (float64, e/bohr^3), cell: Tensor[3, 3] (float64, bohr, rows = lattice vectors))
 *         -> Tensor[] (T_P in Ry)
 * T_P is the Pauli kinetic energy, i.e. everything except the von Weizsaecker term, which ABACUS adds with of_vw_weight.
 * The Pauli potential V_P = dT_P/drho / dV is obtained by automatic differentiation inside libtorch, so any model
 * (descriptors, convolutions, message passing) written in PyTorch can be used without writing its chain rule in C++.
 *
 * The density must be on one process (full real-space grid in C order x, y, z), so only one MPI process is supported.
 * The model can run on CPU or GPU (of_ml_device).
 */
class KEDF_TS
{
  public:
    void set_para(const std::string& device_inpt, std::ostream& ofs_running);

    void ts_potential(const double* const* prho, ModulePW::PW_Basis* pw_rho, ModuleBase::matrix& rpotential);

    double ts_energy = 0.;

  private:
    torch::jit::script::Module module_;
    torch::Device device_ = torch::Device(torch::kCPU);
};

#endif
#endif
