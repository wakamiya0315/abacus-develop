#ifdef __MLALGO

#include "write_mlkedf_desc.h"

#include "npy.hpp"
#include "source_base/constants.h"
#include "source_estate/module_charge/chg_symm.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>

namespace ModuleIO
{
namespace
{
// Fork addition (opt-in): ABACUS_MLKEDF_LABELS_ONLY=1 writes rho and the labels (F, V_P, veff) but not
// the ~35 descriptor arrays. The descriptors are deterministic functions of rho and can be recomputed
// outside ABACUS; writing them all in float64 takes ~20 GB for a 12 A box at 800 Ry.
bool labels_only_requested()
{
    const char* value = std::getenv("ABACUS_MLKEDF_LABELS_ONLY");
    return value != nullptr && std::string(value) == "1";
}

// The constant ABACUS adds to V_P in getF_KS: the largest eigenvalue among bands with
// wg >= threshold_wg. For smeared metals this is not the Fermi level, so it is written next to
// the labels together with E_F, which lets V_P be re-referenced to E_F afterwards.
double top_occupied_eigenvalue(const elecstate::ElecState* pelec)
{
    double eps = pelec->ekb(0, 0);
    for (int ik = 0; ik < pelec->wg.nr; ++ik)
    {
        for (int ib = 0; ib < pelec->wg.nc; ++ib)
        {
            if (pelec->wg(ik, ib) >= ModuleBase::threshold_wg && pelec->ekb(ik, ib) > eps)
            {
                eps = pelec->ekb(ik, ib);
            }
        }
    }
    return eps;
}

// F and V_P are symmetrized the same way as in the original generateTrainData_KS.
void symmetrize_field(std::vector<double>& field, ModulePW::PW_Basis* pw_rho, UnitCell& ucell)
{
    std::vector<std::complex<double>> fieldg(pw_rho->npw);
    double* ptr = field.data();
    std::complex<double>* gptr = fieldg.data();
    module_charge::cal_rhog_symm(0, &ptr, &gptr, pw_rho->npw, nullptr, pw_rho, ucell.symm);
}
// Final outputs shared by both label routes: enhancement.npy, pauli.npy, veff.npy, pauli_reference.txt.
void write_final_labels(const std::string& out_dir, const elecstate::ElecState* pelec, ModulePW::PW_Basis* pw_rho,
                        UnitCell& ucell, const double* veff, const int nrxx,
                        std::vector<double>& enhancement, std::vector<double>& pauli)
{
    symmetrize_field(enhancement, pw_rho, ucell);
    symmetrize_field(pauli, pw_rho, ucell);
    const long unsigned cshape[] = {(long unsigned) nrxx};
    npy::SaveArrayAsNumpy(out_dir + "/enhancement.npy", false, 1, cshape, enhancement);
    npy::SaveArrayAsNumpy(out_dir + "/pauli.npy", false, 1, cshape, pauli);
    std::vector<double> veff_vec(veff, veff + nrxx);
    npy::SaveArrayAsNumpy(out_dir + "/veff.npy", false, 1, cshape, veff_vec);
    std::ofstream ref(out_dir + "/pauli_reference.txt");
    ref << std::setprecision(15) << "# V_P in pauli.npy = (tau_P - sum_i w_i eps_i |psi_i|^2) / rho + epsilon_M (Ry)\n"
        << "epsilon_M_Ry " << top_occupied_eigenvalue(pelec) << "\n"
        << "efermi_Ry " << pelec->eferm.ef << "\n";
}

// SCF-path outputs (float32): path/iter_<iter>_{rho,enhancement,pauli}.npy and a line in path/meta.txt.
void write_path_labels(const std::string& out_dir, const int iter, const elecstate::ElecState* pelec,
                       ModulePW::PW_Basis* pw_rho, UnitCell& ucell, const int nrxx,
                       std::vector<double>& enhancement, std::vector<double>& pauli)
{
    const std::string dir = out_dir + "/path";
    std::filesystem::create_directories(dir);
    symmetrize_field(enhancement, pw_rho, ucell);
    symmetrize_field(pauli, pw_rho, ucell);
    // float32 halves the storage; the path data are for later training, not for the reference energies.
    char prefix[32];
    std::snprintf(prefix, sizeof(prefix), "/iter_%04d_", iter);
    const long unsigned cshape[] = {(long unsigned) nrxx};
    std::vector<float> buffer(nrxx);
    std::copy(pelec->charge->rho[0], pelec->charge->rho[0] + nrxx, buffer.begin());
    npy::SaveArrayAsNumpy(dir + prefix + "rho.npy", false, 1, cshape, buffer);
    std::copy(enhancement.begin(), enhancement.end(), buffer.begin());
    npy::SaveArrayAsNumpy(dir + prefix + "enhancement.npy", false, 1, cshape, buffer);
    std::copy(pauli.begin(), pauli.end(), buffer.begin());
    npy::SaveArrayAsNumpy(dir + prefix + "pauli.npy", false, 1, cshape, buffer);
    const bool new_file = !std::filesystem::exists(dir + "/meta.txt");
    std::ofstream meta(dir + "/meta.txt", std::ios::app);
    if (new_file)
    {
        meta << "# iter epsilon_M_Ry efermi_Ry (rho is the output density of the iteration, before mixing)\n";
    }
    meta << std::setprecision(15) << iter << " " << top_occupied_eigenvalue(pelec) << " " << pelec->eferm.ef << "\n";
}

// KS labels from tau_plus = sum w |grad psi|^2 and eps_rho = sum w eps |psi|^2 (both per volume, Ry units),
// with the same formulas as Cal_MLKEDF_Descriptors::getF_KS:
//   tau_P = tau_plus - |grad rho|^2 / (8 rho) * 2,  F = tau_P / (c_TF rho^{5/3}),  V_P = (tau_P - eps_rho) / rho + epsilon_M.
void labels_from_densities(const elecstate::ElecState* pelec, const double c_tf, const int nrxx,
                           const std::vector<double>& tau_plus, const std::vector<double>& eps_rho,
                           const std::vector<std::vector<double>>& nabla_rho,
                           std::vector<double>& enhancement, std::vector<double>& pauli)
{
    const double* rho = pelec->charge->rho[0];
    const double eps_m = top_occupied_eigenvalue(pelec);
    enhancement.resize(nrxx);
    pauli.resize(nrxx);
    for (int ir = 0; ir < nrxx; ++ir)
    {
        const double grad2 = nabla_rho[0][ir] * nabla_rho[0][ir] + nabla_rho[1][ir] * nabla_rho[1][ir]
                             + nabla_rho[2][ir] * nabla_rho[2][ir];
        const double tau_p = tau_plus[ir] - grad2 / (8. * rho[ir]) * 2.;
        enhancement[ir] = tau_p / (c_tf * std::pow(rho[ir], 5. / 3.));
        pauli[ir] = (tau_p - eps_rho[ir]) / rho[ir] + eps_m;
    }
}
} // namespace

void Write_MLKEDF_Descriptors::generateTrainData_KS(
    const std::string& out_dir,
    psi::Psi<std::complex<double>> *psi,
    elecstate::ElecState *pelec,
    ModulePW::PW_Basis_K *pw_psi,
    ModulePW::PW_Basis *pw_rho,
    UnitCell& ucell,
    const double* veff,
    const int nrxx
)
{
    if (nrxx <= 0)
    {
        ModuleBase::WARNING_QUIT("Write_MLKEDF_Descriptors::generateTrainData_KS", "nrxx must be greater than 0");
    }

    std::vector<std::vector<double>> drho(3, std::vector<double>(nrxx, 0.));
    this->prepare_rho_outputs(out_dir, pelec, pw_rho, nrxx, drho);

    std::vector<double> enhancement(nrxx);
    std::vector<double> pauli(nrxx);

    this->cal_tool->getF_KS(psi, pelec, pw_psi, pw_rho, ucell, drho, enhancement, pauli);
    write_final_labels(out_dir, pelec, pw_rho, ucell, veff, nrxx, enhancement, pauli);
}

void Write_MLKEDF_Descriptors::prepare_rho_outputs(const std::string& out_dir,
                                                   elecstate::ElecState* pelec,
                                                   ModulePW::PW_Basis* pw_rho,
                                                   const int nrxx,
                                                   std::vector<std::vector<double>>& drho)
{
    const long unsigned cshape[] = {(long unsigned) nrxx};
    if (labels_only_requested())
    {
        std::vector<double> rho(pelec->charge->rho[0], pelec->charge->rho[0] + nrxx);
        npy::SaveArrayAsNumpy(out_dir + "/rho.npy", false, 1, cshape, rho);
        this->cal_tool->getNablaRho(pelec->charge->rho, pw_rho, drho); // still needed for F
    }
    else
    {
        this->generate_descriptor(out_dir, pelec->charge->rho, pw_rho, drho, nrxx);
    }
}

void Write_MLKEDF_Descriptors::generateTrainData_KS(const std::string& out_dir,
                                                    elecstate::ElecState* pelec,
                                                    ModulePW::PW_Basis* pw_rho,
                                                    UnitCell& ucell,
                                                    const double* veff,
                                                    const int nrxx,
                                                    const std::vector<double>& tau_plus,
                                                    const std::vector<double>& eps_rho)
{
    std::vector<std::vector<double>> drho(3, std::vector<double>(nrxx, 0.));
    this->prepare_rho_outputs(out_dir, pelec, pw_rho, nrxx, drho);
    std::vector<double> enhancement, pauli;
    labels_from_densities(pelec, this->cal_tool->cTF, nrxx, tau_plus, eps_rho, drho, enhancement, pauli);
    write_final_labels(out_dir, pelec, pw_rho, ucell, veff, nrxx, enhancement, pauli);
}

void Write_MLKEDF_Descriptors::generatePathData_KS(const std::string& out_dir,
                                                   const int iter,
                                                   elecstate::ElecState* pelec,
                                                   ModulePW::PW_Basis* pw_rho,
                                                   UnitCell& ucell,
                                                   const int nrxx,
                                                   const std::vector<double>& tau_plus,
                                                   const std::vector<double>& eps_rho)
{
    std::vector<std::vector<double>> drho(3, std::vector<double>(nrxx, 0.));
    this->cal_tool->getNablaRho(pelec->charge->rho, pw_rho, drho);
    std::vector<double> enhancement, pauli;
    labels_from_densities(pelec, this->cal_tool->cTF, nrxx, tau_plus, eps_rho, drho, enhancement, pauli);
    write_path_labels(out_dir, iter, pelec, pw_rho, ucell, nrxx, enhancement, pauli);
}

void Write_MLKEDF_Descriptors::generatePathData_KS(
    const std::string& out_dir,
    const int iter,
    psi::Psi<std::complex<double>> *psi,
    elecstate::ElecState *pelec,
    ModulePW::PW_Basis_K *pw_psi,
    ModulePW::PW_Basis *pw_rho,
    UnitCell& ucell,
    const int nrxx
)
{
    std::vector<std::vector<double>> drho(3, std::vector<double>(nrxx, 0.));
    this->cal_tool->getNablaRho(pelec->charge->rho, pw_rho, drho);
    std::vector<double> enhancement(nrxx);
    std::vector<double> pauli(nrxx);
    this->cal_tool->getF_KS(psi, pelec, pw_psi, pw_rho, ucell, drho, enhancement, pauli);
    write_path_labels(out_dir, iter, pelec, pw_rho, ucell, nrxx, enhancement, pauli);
}

void Write_MLKEDF_Descriptors::generatePathData_KS(
    const std::string& out_dir,
    const int iter,
    psi::Psi<std::complex<float>> *psi,
    elecstate::ElecState *pelec,
    ModulePW::PW_Basis_K *pw_psi,
    ModulePW::PW_Basis *pw_rho,
    UnitCell& ucell,
    const int nrxx
)
{
    psi::Psi<std::complex<double>, base_device::DEVICE_CPU> psi_double(*psi);
    this->generatePathData_KS(out_dir, iter, &psi_double, pelec, pw_psi, pw_rho, ucell, nrxx);
}

#if ((defined __CUDA) || (defined __ROCM))
void Write_MLKEDF_Descriptors::generatePathData_KS(
    const std::string& out_dir,
    const int iter,
    psi::Psi<std::complex<double>, base_device::DEVICE_GPU>* psi,
    elecstate::ElecState *pelec,
    ModulePW::PW_Basis_K *pw_psi,
    ModulePW::PW_Basis *pw_rho,
    UnitCell& ucell,
    const int nrxx
)
{
    psi::Psi<std::complex<double>, base_device::DEVICE_CPU> psi_cpu(*psi);
    this->generatePathData_KS(out_dir, iter, &psi_cpu, pelec, pw_psi, pw_rho, ucell, nrxx);
}

void Write_MLKEDF_Descriptors::generatePathData_KS(
    const std::string& out_dir,
    const int iter,
    psi::Psi<std::complex<float>, base_device::DEVICE_GPU>* psi,
    elecstate::ElecState *pelec,
    ModulePW::PW_Basis_K *pw_psi,
    ModulePW::PW_Basis *pw_rho,
    UnitCell& ucell,
    const int nrxx
)
{
    psi::Psi<std::complex<double>, base_device::DEVICE_CPU> psi_cpu_double(*psi);
    this->generatePathData_KS(out_dir, iter, &psi_cpu_double, pelec, pw_psi, pw_rho, ucell, nrxx);
}
#endif

void Write_MLKEDF_Descriptors::generateTrainData_KS(
    const std::string& out_dir,
    psi::Psi<std::complex<float>> *psi,
    elecstate::ElecState *pelec,
    ModulePW::PW_Basis_K *pw_psi,
    ModulePW::PW_Basis *pw_rho,
    UnitCell& ucell,
    const double* veff,
    const int nrxx
)
{
    psi::Psi<std::complex<double>, base_device::DEVICE_CPU> psi_double(*psi);

    this->generateTrainData_KS(out_dir, &psi_double, pelec, pw_psi, pw_rho, ucell, veff, nrxx);
}

#if ((defined __CUDA) || (defined __ROCM))
void Write_MLKEDF_Descriptors::generateTrainData_KS(
    const std::string& out_dir,
    psi::Psi<std::complex<double>, base_device::DEVICE_GPU>* psi,
    elecstate::ElecState *pelec,
    ModulePW::PW_Basis_K *pw_psi,
    ModulePW::PW_Basis *pw_rho,
    UnitCell& ucell,
    const double* veff,
    const int nrxx
)
{
    psi::Psi<std::complex<double>, base_device::DEVICE_CPU> psi_cpu(*psi);

    this->generateTrainData_KS(out_dir, &psi_cpu, pelec, pw_psi, pw_rho, ucell, veff, nrxx);
}

void Write_MLKEDF_Descriptors::generateTrainData_KS(
    const std::string& dir,
    psi::Psi<std::complex<float>, base_device::DEVICE_GPU>* psi,
    elecstate::ElecState *pelec,
    ModulePW::PW_Basis_K *pw_psi,
    ModulePW::PW_Basis *pw_rho,
    UnitCell& ucell,
    const double *veff,
    const int nrxx
)
{
    psi::Psi<std::complex<double>, base_device::DEVICE_CPU> psi_cpu_double(*psi);

    this->generateTrainData_KS(dir, &psi_cpu_double, pelec, pw_psi, pw_rho, ucell, veff, nrxx);
}
#endif

void Write_MLKEDF_Descriptors::generate_descriptor(
    const std::string& out_dir,
    const double * const *prho, 
    ModulePW::PW_Basis *pw_rho,
    std::vector<std::vector<double>> &nablaRho,
    const int nrxx
)
{
    // container which will contain gamma, p, q in turn
    std::vector<double> container(nrxx);
    std::vector<double> new_container(nrxx);
    // container contains gammanl, pnl, qnl in turn
    std::vector<double> containernl(nrxx);
    std::vector<double> new_containernl(nrxx);

    const long unsigned cshape[] = {(long unsigned) nrxx};

    // rho
    std::vector<double> rho(nrxx);
    for (int ir = 0; ir < nrxx; ++ir){
        rho[ir] = prho[0][ir];
    }
    npy::SaveArrayAsNumpy(out_dir + "/rho.npy", false, 1, cshape, rho);

    // gamma
    this->cal_tool->getGamma(prho, container);
    npy::SaveArrayAsNumpy(out_dir + "/gamma.npy", false, 1, cshape, container);

    for (int ik = 0; ik < this->cal_tool->nkernel; ++ik)
    {
        int ktype = this->cal_tool->kernel_type[ik];
        double kscaling = this->cal_tool->kernel_scaling[ik];

        // gamma_nl
        this->cal_tool->getGammanl(ik, container, pw_rho, containernl);
        npy::SaveArrayAsNumpy(this->file_name(out_dir, "gammanl", ktype, kscaling), false, 1, cshape, containernl);

        // xi = gamma_nl/gamma
        this->cal_tool->getXi(container, containernl, new_container);
        npy::SaveArrayAsNumpy(this->file_name(out_dir, "xi", ktype, kscaling), false, 1, cshape, new_container);

        // tanhxi = tanh(xi)
        this->cal_tool->getTanhXi(ik, container, containernl, new_container);
        npy::SaveArrayAsNumpy(this->file_name(out_dir, "tanhxi", ktype, kscaling), false, 1, cshape, new_container);

        // (tanhxi)_nl
        this->cal_tool->getTanhXi_nl(ik, new_container, pw_rho, new_containernl);
        npy::SaveArrayAsNumpy(this->file_name(out_dir, "tanhxi_nl", ktype, kscaling), false, 1, cshape, new_containernl);
    }

    // nabla rho
    this->cal_tool->getNablaRho(prho, pw_rho, nablaRho);
    npy::SaveArrayAsNumpy(out_dir + "/nablaRhox.npy", false, 1, cshape, nablaRho[0]);
    npy::SaveArrayAsNumpy(out_dir + "/nablaRhoy.npy", false, 1, cshape, nablaRho[1]);
    npy::SaveArrayAsNumpy(out_dir + "/nablaRhoz.npy", false, 1, cshape, nablaRho[2]);

    // p
    this->cal_tool->getP(prho, pw_rho, nablaRho, container);
    npy::SaveArrayAsNumpy(out_dir + "/p.npy", false, 1, cshape, container);

    for (int ik = 0; ik < this->cal_tool->nkernel; ++ik)
    {
        int ktype = this->cal_tool->kernel_type[ik];
        double kscaling = this->cal_tool->kernel_scaling[ik];

        // p_nl
        this->cal_tool->getPnl(ik, container, pw_rho, containernl);
        npy::SaveArrayAsNumpy(this->file_name(out_dir, "pnl", ktype, kscaling), false, 1, cshape, containernl);

        // tanh(p_nl)
        this->cal_tool->getTanh_Pnl(ik, containernl, new_containernl);
        npy::SaveArrayAsNumpy(this->file_name(out_dir, "tanh_pnl", ktype, kscaling), false, 1, cshape, new_containernl);

        // tanh(p)
        this->cal_tool->getTanhP(container, new_container);
        npy::SaveArrayAsNumpy(out_dir + "/tanhp.npy", false, 1, cshape, new_container);

        // tanh(p)_nl
        this->cal_tool->getTanhP_nl(ik, new_container, pw_rho, new_containernl);
        npy::SaveArrayAsNumpy(this->file_name(out_dir, "tanhp_nl", ktype, kscaling), false, 1, cshape, new_containernl);
    }

    // q
    this->cal_tool->getQ(prho, pw_rho, container);
    npy::SaveArrayAsNumpy(out_dir + "/q.npy", false, 1, cshape, container);

    for (int ik = 0; ik < this->cal_tool->nkernel; ++ik)
    {
        int ktype = this->cal_tool->kernel_type[ik];
        double kscaling = this->cal_tool->kernel_scaling[ik];

        // q_nl
        this->cal_tool->getQnl(ik, container, pw_rho, containernl);
        npy::SaveArrayAsNumpy(this->file_name(out_dir, "qnl", ktype, kscaling), false, 1, cshape, containernl);

        // tanh(q_nl)
        this->cal_tool->getTanh_Qnl(ik, containernl, new_containernl);
        npy::SaveArrayAsNumpy(this->file_name(out_dir, "tanh_qnl", ktype, kscaling), false, 1, cshape, new_containernl);

        // tanh(q)
        this->cal_tool->getTanhQ(container, new_container);
        npy::SaveArrayAsNumpy(out_dir + "/tanhq.npy", false, 1, cshape, new_container);

        // tanh(q)_nl
        this->cal_tool->getTanhQ_nl(ik, new_container, pw_rho, new_containernl);
        npy::SaveArrayAsNumpy(this->file_name(out_dir, "tanhq_nl", ktype, kscaling), false, 1, cshape, new_containernl);
    }
}

std::string Write_MLKEDF_Descriptors::file_name(
    const std::string& out_dir,
    std::string parameter,
    const int kernel_type,
    const double kernel_scaling
)
{
    std::stringstream ss;
    ss << out_dir << "/" << parameter << "_" << kernel_type << "_" << kernel_scaling << ".npy";
    return ss.str();
}

}

#endif
