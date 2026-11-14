#ifdef __MLALGO

#include "kedf_ts.h"

#include "source_base/global_variable.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"

#include <torch/cuda.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include "npy.hpp"

#include <cstdlib>
#include <string>

void KEDF_TS::set_para(const std::string& device_inpt, std::ostream& ofs_running)
{
    if (GlobalV::NPROC != 1)
    {
        ModuleBase::WARNING_QUIT("KEDF_TS", "of_kinetic ml-ts supports only one MPI process (the model needs the full grid).");
    }
    if (device_inpt == "gpu" && torch::cuda::is_available())
    {
        this->device_ = torch::Device(torch::kCUDA);
    }
    try
    {
        this->module_ = torch::jit::load("kedf_ts.pt", this->device_);
    }
    catch (const c10::Error& e)
    {
        ModuleBase::WARNING_QUIT("KEDF_TS", "Failed to load kedf_ts.pt (TorchScript module).");
    }
    this->module_.eval();
    // TorchScriptのグラフ最適化は止める。最適化された2回目以降の呼び出しで、FFT（複素数）を含むモデルの
    // 自動微分の V_P が誤った値になった（Ptバルク、ネイティブの ml と照合して確認）。
    // ABACUS_MLKEDF_TS_OPT=1 のときだけ最適化を使う（速度の比較用）。
    if (std::getenv("ABACUS_MLKEDF_TS_OPT") == nullptr)
    {
        torch::jit::setGraphExecutorOptimize(false);
        ofs_running << " ML-TS KEDF: TorchScript graph optimization disabled" << std::endl;
    }
    for (auto p: this->module_.parameters())
    {
        p.set_requires_grad(false);
    }
    ofs_running << " ML-TS KEDF: loaded kedf_ts.pt on " << (this->device_.is_cuda() ? "GPU" : "CPU") << std::endl;
}

/**
 * @brief Add the Pauli potential V_P (Ry) to rpotential and store T_P (Ry) in ts_energy.
 */
void KEDF_TS::ts_potential(const double* const* prho, ModulePW::PW_Basis* pw_rho, ModuleBase::matrix& rpotential)
{
    ModuleBase::timer::start("KEDF_TS", "ts_potential");
    const int nx = pw_rho->nx, ny = pw_rho->ny, nz = pw_rho->nz;
    if (pw_rho->nrxx != nx * ny * nz)
    {
        ModuleBase::WARNING_QUIT("KEDF_TS", "the real-space grid is distributed; ml-ts needs the full grid on one process.");
    }
    const double dV = pw_rho->omega / pw_rho->nxyz;
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor rho = torch::from_blob(const_cast<double*>(prho[0]), {nx, ny, nz}, opts).clone().to(this->device_);
    rho.set_requires_grad(true);
    torch::Tensor cell = torch::empty({3, 3}, opts);
    auto c = cell.accessor<double, 2>();
    const ModuleBase::Matrix3& a = pw_rho->latvec;
    const double l0 = pw_rho->lat0;
    c[0][0] = a.e11 * l0; c[0][1] = a.e12 * l0; c[0][2] = a.e13 * l0;
    c[1][0] = a.e21 * l0; c[1][1] = a.e22 * l0; c[1][2] = a.e23 * l0;
    c[2][0] = a.e31 * l0; c[2][1] = a.e32 * l0; c[2][2] = a.e33 * l0;
    cell = cell.to(this->device_);

    std::vector<torch::jit::IValue> inputs{rho, cell};
    torch::Tensor tp = this->module_.forward(inputs).toTensor();
    tp.backward();
    torch::Tensor grad = rho.grad().to(torch::kCPU).contiguous();
    const double* g = grad.data_ptr<double>();
    for (int ir = 0; ir < pw_rho->nrxx; ++ir)
    {
        rpotential(0, ir) += g[ir] / dV;
    }
    this->ts_energy = tp.item<double>();
    // ABACUS_MLKEDF_TS_DUMP=N: 最初のN回の呼び出しで、受け取った密度と返したV_P（Ry）を npy に書く（照合用）
    static int ncall = 0;
    const char* dump = std::getenv("ABACUS_MLKEDF_TS_DUMP");
    if (dump != nullptr && ncall < std::atoi(dump))
    {
        const long unsigned shape[] = {(long unsigned)pw_rho->nrxx};
        std::vector<double> buf(prho[0], prho[0] + pw_rho->nrxx);
        npy::SaveArrayAsNumpy("ts_dump_rho_" + std::to_string(ncall) + ".npy", false, 1, shape, buf);
        for (int ir = 0; ir < pw_rho->nrxx; ++ir)
        {
            buf[ir] = g[ir] / dV;
        }
        npy::SaveArrayAsNumpy("ts_dump_vp_" + std::to_string(ncall) + ".npy", false, 1, shape, buf);
    }
    ++ncall;
    ModuleBase::timer::end("KEDF_TS", "ts_potential");
}

#endif
