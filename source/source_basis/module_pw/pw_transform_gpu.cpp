#include "source_base/tool_quit.h"
#include <cstdlib>
#include <string>
#include "source_base/timer.h"
#include "source_basis/module_pw/kernels/pw_op.h"
#include "pw_basis.h"
namespace ModulePW
{
#if (defined(__CUDA) || defined(__ROCM))
template <typename FPTYPE>
void PW_Basis::real2recip_gpu(const FPTYPE* in, std::complex<FPTYPE>* out, const bool add, const FPTYPE factor) const
{
    ModuleBase::timer::start(this->classname, "real_to_recip gpu");
    assert(this->poolnproc == 1);
    const size_t size = this->nrxx;
    base_device::memory::cast_memory_op<std::complex<FPTYPE>, FPTYPE,base_device::DEVICE_GPU, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
        in,
        size);

    this->fft_bundle.fft3D_forward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                                   this->nxyz,
                                                                   add,
                                                                   factor,
                                                                   this->ig2ixyz_gpu,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);
    ModuleBase::timer::end(this->classname, "real_to_recip gpu");
}
template <typename FPTYPE>
void PW_Basis::real2recip_gpu(const std::complex<FPTYPE>* in,
                              std::complex<FPTYPE>* out,
                              const bool add,
                              const FPTYPE factor) const
{
    ModuleBase::timer::start(this->classname, "real_to_recip gpu");
    assert(this->poolnproc == 1);
    base_device::memory::synchronize_memory_op<std::complex<FPTYPE>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_GPU>()(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                          in,
                                                                          this->nrxx);
    this->fft_bundle.fft3D_forward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                                   this->nxyz,
                                                                   add,
                                                                   factor,
                                                                   this->ig2ixyz_gpu,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);
    ModuleBase::timer::end(this->classname, "real_to_recip gpu");
}

template <typename FPTYPE>
void PW_Basis::recip2real_gpu(const std::complex<FPTYPE>* in, FPTYPE* out, const bool add, const FPTYPE factor) const
{
    ModuleBase::timer::start(this->classname, "recip_to_real gpu");
    assert(this->poolnproc == 1);
    // ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxr_3d_data<FPTYPE>(), this->nxyz);
    base_device::memory::set_memory_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
        0,
        this->nxyz);
    set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                         this->ig2ixyz_gpu,
                                                         in,
                                                         this->fft_bundle.get_auxr_3d_data<FPTYPE>());
    this->fft_bundle.fft3D_backward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                    this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>()(this->nrxx,
                                                                   add,
                                                                   factor,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);

    ModuleBase::timer::end(this->classname, "recip_to_real gpu");
}
template <typename FPTYPE>
void PW_Basis::recip2real_gpu(const std::complex<FPTYPE>* in,
                              std::complex<FPTYPE>* out,
                              const bool add,
                              const FPTYPE factor) const
{
    ModuleBase::timer::start(this->classname, "recip_to_real gpu");
    assert(this->poolnproc == 1);
    // ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxr_3d_data<double>(), this->nxyz);
    base_device::memory::set_memory_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
        0,
        this->nxyz);

    set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                         this->ig2ixyz_gpu,
                                                         in,
                                                         this->fft_bundle.get_auxr_3d_data<FPTYPE>());
    this->fft_bundle.fft3D_backward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                    this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>()(this->nrxx,
                                                                   add,
                                                                   factor,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);

    ModuleBase::timer::end(this->classname, "recip_to_real gpu");
}
template void PW_Basis::real2recip_gpu<double>(const double* in,
                                               std::complex<double>* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::real2recip_gpu<float>(const float* in,
                                              std::complex<float>* out,
                                              const bool add,
                                              const float factor) const;

template void PW_Basis::real2recip_gpu<double>(const std::complex<double>* in,
                                               std::complex<double>* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::real2recip_gpu<float>(const std::complex<float>* in,
                                              std::complex<float>* out,
                                              const bool add,
                                              const float factor) const;

template void PW_Basis::recip2real_gpu<double>(const std::complex<double>* in,
                                               double* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::recip2real_gpu<float>(const std::complex<float>* in,
                                              float* out,
                                              const bool add,
                                              const float factor) const;

template void PW_Basis::recip2real_gpu<double>(const std::complex<double>* in,
                                               std::complex<double>* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::recip2real_gpu<float>(const std::complex<float>* in,
                                              std::complex<float>* out,
                                              const bool add,
                                              const float factor) const;

#endif

#if (defined(__CUDA) || defined(__ROCM))
bool PW_Basis::host_fft_on_gpu() const
{
    static const bool enabled = []() {
        const char* env = std::getenv("ABACUS_HOST_FFT_ON_GPU");
        return env == nullptr || std::string(env) != "0";
    }();
    // 波動関数の基底（PW_Basis_K）は npw の意味が k ごとに異なるので対象外にする。
    return enabled && this->device == "gpu" && this->poolnproc == 1 && !this->gamma_only && this->npw > 0
           && this->classname != "PW_Basis_K";
}

void PW_Basis::ensure_host_fft_buffers() const
{
    if (this->d_host_fft_g_ == nullptr)
    {
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(this->d_host_fft_g_, this->npw);
    }
    if (this->d_host_fft_r_ == nullptr)
    {
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(this->d_host_fft_r_, this->nrxx);
    }
}

void PW_Basis::free_host_fft_buffers()
{
    if (this->d_host_fft_g_ != nullptr)
    {
        base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(this->d_host_fft_g_);
        this->d_host_fft_g_ = nullptr;
    }
    if (this->d_host_fft_r_ != nullptr)
    {
        base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(this->d_host_fft_r_);
        this->d_host_fft_r_ = nullptr;
    }
}

using h2d_c = base_device::memory::synchronize_memory_op<std::complex<double>, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
using d2h_c = base_device::memory::synchronize_memory_op<std::complex<double>, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
using h2d_r = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
using d2h_r = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

void PW_Basis::recip2real_host_via_gpu(const std::complex<float>*, float*, const bool, const float) const
{
    ModuleBase::WARNING_QUIT("PW_Basis", "host FFT on GPU is implemented only for double precision");
}
void PW_Basis::recip2real_host_via_gpu(const std::complex<float>*, std::complex<float>*, const bool, const float) const
{
    ModuleBase::WARNING_QUIT("PW_Basis", "host FFT on GPU is implemented only for double precision");
}
void PW_Basis::real2recip_host_via_gpu(const float*, std::complex<float>*, const bool, const float) const
{
    ModuleBase::WARNING_QUIT("PW_Basis", "host FFT on GPU is implemented only for double precision");
}
void PW_Basis::real2recip_host_via_gpu(const std::complex<float>*, std::complex<float>*, const bool, const float) const
{
    ModuleBase::WARNING_QUIT("PW_Basis", "host FFT on GPU is implemented only for double precision");
}

void PW_Basis::recip2real_host_via_gpu(const std::complex<double>* in, double* out, const bool add, const double factor) const
{
    ModuleBase::timer::start(this->classname, "recip2real_host_gpu");
    this->ensure_host_fft_buffers();
    double* d_out = reinterpret_cast<double*>(this->d_host_fft_r_);
    h2d_c()(this->d_host_fft_g_, in, this->npw);
    if (add)
    {
        h2d_r()(d_out, out, this->nrxx);
    }
    this->recip2real_gpu<double>(this->d_host_fft_g_, d_out, add, factor);
    d2h_r()(out, d_out, this->nrxx);
    ModuleBase::timer::end(this->classname, "recip2real_host_gpu");
}

void PW_Basis::recip2real_host_via_gpu(const std::complex<double>* in, std::complex<double>* out, const bool add, const double factor) const
{
    ModuleBase::timer::start(this->classname, "recip2real_host_gpu");
    this->ensure_host_fft_buffers();
    h2d_c()(this->d_host_fft_g_, in, this->npw);
    if (add)
    {
        h2d_c()(this->d_host_fft_r_, out, this->nrxx);
    }
    this->recip2real_gpu<double>(this->d_host_fft_g_, this->d_host_fft_r_, add, factor);
    d2h_c()(out, this->d_host_fft_r_, this->nrxx);
    ModuleBase::timer::end(this->classname, "recip2real_host_gpu");
}

void PW_Basis::real2recip_host_via_gpu(const double* in, std::complex<double>* out, const bool add, const double factor) const
{
    ModuleBase::timer::start(this->classname, "real2recip_host_gpu");
    this->ensure_host_fft_buffers();
    double* d_in = reinterpret_cast<double*>(this->d_host_fft_r_);
    h2d_r()(d_in, in, this->nrxx);
    if (add)
    {
        h2d_c()(this->d_host_fft_g_, out, this->npw);
    }
    this->real2recip_gpu<double>(d_in, this->d_host_fft_g_, add, factor);
    d2h_c()(out, this->d_host_fft_g_, this->npw);
    ModuleBase::timer::end(this->classname, "real2recip_host_gpu");
}

void PW_Basis::real2recip_host_via_gpu(const std::complex<double>* in, std::complex<double>* out, const bool add, const double factor) const
{
    ModuleBase::timer::start(this->classname, "real2recip_host_gpu");
    this->ensure_host_fft_buffers();
    h2d_c()(this->d_host_fft_r_, in, this->nrxx);
    if (add)
    {
        h2d_c()(this->d_host_fft_g_, out, this->npw);
    }
    this->real2recip_gpu<double>(this->d_host_fft_r_, this->d_host_fft_g_, add, factor);
    d2h_c()(out, this->d_host_fft_g_, this->npw);
    ModuleBase::timer::end(this->classname, "real2recip_host_gpu");
}
#endif

} // namespace ModulePW