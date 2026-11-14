#ifndef WRITE_MLKEDF_DESC_H
#define WRITE_MLKEDF_DESC_H

#ifdef __MLALGO

#include "cal_mlkedf_desc.h"
#include "source_estate/elecstate_pw.h"

namespace ModuleIO
{

class Write_MLKEDF_Descriptors
{
public:
    Write_MLKEDF_Descriptors() {
        this->cal_tool = new Cal_MLKEDF_Descriptors();
    }
    
    ~Write_MLKEDF_Descriptors() {
        if (this->cal_tool != nullptr) {
            delete this->cal_tool;
            this->cal_tool = nullptr;
        }
    }

    void generateTrainData_KS(
        const std::string& out_dir,
        psi::Psi<std::complex<double>> *psi,
        elecstate::ElecState *pelec,
        ModulePW::PW_Basis_K *pw_psi,
        ModulePW::PW_Basis *pw_rho,
        UnitCell& ucell,
        const double *veff,
        const int nrxx
    );
    void generateTrainData_KS(
        const std::string& out_dir,
        psi::Psi<std::complex<float>> *psi,
        elecstate::ElecState *pelec,
        ModulePW::PW_Basis_K *pw_psi,
        ModulePW::PW_Basis *pw_rho,
        UnitCell& ucell,
        const double *veff,
        const int nrxx
    );

#if ((defined __CUDA) || (defined __ROCM))
    void generateTrainData_KS(
        const std::string& dir,
        psi::Psi<std::complex<double>, base_device::DEVICE_GPU>* psi,
        elecstate::ElecState *pelec,
        ModulePW::PW_Basis_K *pw_psi,
        ModulePW::PW_Basis *pw_rho,
        UnitCell& ucell,
        const double *veff,
        const int nrxx
    );
    void generateTrainData_KS(
        const std::string& dir,
        psi::Psi<std::complex<float>, base_device::DEVICE_GPU>* psi,
        elecstate::ElecState *pelec,
        ModulePW::PW_Basis_K *pw_psi,
        ModulePW::PW_Basis *pw_rho,
        UnitCell& ucell,
        const double *veff,
        const int nrxx
    );
#endif

    // Fork addition: training labels along the SCF path. Writes, as float32, the output density of
    // SCF iteration `iter` and the KS labels F and V_P computed from the same orbitals and occupations
    // to <out_dir>/path/iter_<iter>_{rho,enhancement,pauli}.npy, and appends the V_P reference
    // energies to <out_dir>/path/meta.txt. Called from ESolver_KS_PW when ABACUS_MLKEDF_PATH_EVERY > 0.
    void generatePathData_KS(
        const std::string& out_dir,
        const int iter,
        psi::Psi<std::complex<double>> *psi,
        elecstate::ElecState *pelec,
        ModulePW::PW_Basis_K *pw_psi,
        ModulePW::PW_Basis *pw_rho,
        UnitCell& ucell,
        const int nrxx
    );
    void generatePathData_KS(
        const std::string& out_dir,
        const int iter,
        psi::Psi<std::complex<float>> *psi,
        elecstate::ElecState *pelec,
        ModulePW::PW_Basis_K *pw_psi,
        ModulePW::PW_Basis *pw_rho,
        UnitCell& ucell,
        const int nrxx
    );
#if ((defined __CUDA) || (defined __ROCM))
    void generatePathData_KS(
        const std::string& out_dir,
        const int iter,
        psi::Psi<std::complex<double>, base_device::DEVICE_GPU>* psi,
        elecstate::ElecState *pelec,
        ModulePW::PW_Basis_K *pw_psi,
        ModulePW::PW_Basis *pw_rho,
        UnitCell& ucell,
        const int nrxx
    );
    void generatePathData_KS(
        const std::string& out_dir,
        const int iter,
        psi::Psi<std::complex<float>, base_device::DEVICE_GPU>* psi,
        elecstate::ElecState *pelec,
        ModulePW::PW_Basis_K *pw_psi,
        ModulePW::PW_Basis *pw_rho,
        UnitCell& ucell,
        const int nrxx
    );
#endif

    // Fork addition: the same outputs, with the KS labels built from tau_plus = sum w |grad psi|^2 and
    // eps_rho = sum w eps |psi|^2 accumulated on the device (ElecStatePW::cal_kedf_label_densities)
    // instead of the band-by-band CPU FFTs of Cal_MLKEDF_Descriptors::getF_KS.
    void generateTrainData_KS(const std::string& out_dir,
                              elecstate::ElecState* pelec,
                              ModulePW::PW_Basis* pw_rho,
                              UnitCell& ucell,
                              const double* veff,
                              const int nrxx,
                              const std::vector<double>& tau_plus,
                              const std::vector<double>& eps_rho);
    void generatePathData_KS(const std::string& out_dir,
                             const int iter,
                             elecstate::ElecState* pelec,
                             ModulePW::PW_Basis* pw_rho,
                             UnitCell& ucell,
                             const int nrxx,
                             const std::vector<double>& tau_plus,
                             const std::vector<double>& eps_rho);

    // rho.npy only (ABACUS_MLKEDF_LABELS_ONLY=1) or all descriptors; fills nablaRho in both cases.
    void prepare_rho_outputs(const std::string& out_dir,
                             elecstate::ElecState* pelec,
                             ModulePW::PW_Basis* pw_rho,
                             const int nrxx,
                             std::vector<std::vector<double>>& nablaRho);

    void generate_descriptor(
        const std::string& out_dir,
        const double * const *prho,
        ModulePW::PW_Basis *pw_rho,
        std::vector<std::vector<double>> &nablaRho,
        const int nrxx
    );

    std::string file_name(
        const std::string& out_dir,
        std::string parameter,
        const int kernel_type,
        const double kernel_scaling
    );

    Cal_MLKEDF_Descriptors* cal_tool = nullptr; // pointer to the calculation tool
};

} // namespace ModuleIO
#endif
#endif