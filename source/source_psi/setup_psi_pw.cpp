#include "source_psi/setup_psi_pw.h"

#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/klist.h"
#include "source_cell/stru_fac.h"
#include "source_cell/unitcell.h"
#include "source_base/global_variable.h"
#include "source_base/tool_quit.h"
#include "source_io/module_parameter/parameter.h"

#include <cstdlib>
#include <string>

namespace
{
// ABACUS_PSI_MANAGED=1（CUDA版）: 全k点の波動関数をCUDA managed memoryに置き、ホスト側（psi_cpu）と
// GPU側で同じ配列を共有する。GPUには計算中のk点だけを置くので、VRAMの常駐量が全k点分から1k点分に減る。
// k点の出し入れはSCF反復ごとに対角化と密度計算の2回で、1k点あたり数GB・PCIeで0.1〜0.3秒程度。
bool psi_managed_requested()
{
    const char* value = std::getenv("ABACUS_PSI_MANAGED");
    return value != nullptr && std::string(value) == "1";
}

// 対応するのはCUDA版の倍精度だけ。それ以外の組み合わせでは nullptr を返す。
template <typename T, typename Device>
psi::Psi<T, Device>* make_managed_view(psi::Psi<std::complex<double>, base_device::DEVICE_CPU>&)
{
    return nullptr;
}

#if defined(__CUDA)
template <>
psi::Psi<std::complex<double>, base_device::DEVICE_GPU>* make_managed_view<std::complex<double>,
                                                                           base_device::DEVICE_GPU>(
    psi::Psi<std::complex<double>, base_device::DEVICE_CPU>& host)
{
    host.use_managed_memory();
    return new psi::Psi<std::complex<double>, base_device::DEVICE_GPU>(host, psi::ManagedView{});
}
#endif
} // namespace

Setup_Psi_pw::Setup_Psi_pw()
{
}

Setup_Psi_pw::~Setup_Psi_pw()
{
}

template <typename T, typename Device>
void Setup_Psi_pw::before_runner_impl(const UnitCell& ucell,
                                      const K_Vectors& kv,
                                      const Structure_Factor& sf,
                                      const ModulePW::PW_Basis_K& pw_wfc,
                                      const int& lmaxkb,
                                      const Input_para& inp)
{
    this->p_psi_init = new psi::PSIPrepare<T, Device>(inp.init_wfc,
                                                      inp.ks_solver,
                                                      inp.basis_type,
                                                      GlobalV::MY_RANK,
                                                      ucell,
                                                      sf,
                                                      kv.ik2iktot,
                                                      kv.get_nkstot(),
                                                      lmaxkb,
                                                      pw_wfc);

    allocate_psi(this->psi_cpu, kv.get_nks(), kv.ngk, PARAM.globalv.nbands_l, pw_wfc.npwk_max);

    auto* p_psi_init = static_cast<psi::PSIPrepare<T, Device>*>(this->p_psi_init);
    // before_runner_impl is invoked only once before the ion dynamics starts,
    // so istep == 0 here ensures any one-time informational warnings are
    // printed from this initial setup call.
    p_psi_init->prepare_init(inp.pw_seed, 0);

    if (std::is_same<T, float>::value)
    {
        precision_type_ = PrecisionType::Float;
    }
    else if (std::is_same<T, double>::value)
    {
        precision_type_ = PrecisionType::Double;
    }
    else if (std::is_same<T, std::complex<float>>::value)
    {
        precision_type_ = PrecisionType::ComplexFloat;
    }
    else
    {
        precision_type_ = PrecisionType::ComplexDouble;
    }

    if (std::is_same<Device, base_device::DEVICE_GPU>::value)
    {
        device_type_ = base_device::GpuDevice;
    }
    else
    {
        device_type_ = base_device::CpuDevice;
    }

    if (inp.device == "gpu" || inp.precision == "single")
    {
        psi::Psi<T, Device>* managed_view = nullptr;
        if (inp.device == "gpu" && psi_managed_requested())
        {
            managed_view = make_managed_view<T, Device>(this->psi_cpu[0]);
            if (managed_view == nullptr)
            {
                ModuleBase::WARNING_QUIT("Setup_Psi_pw",
                                         "ABACUS_PSI_MANAGED=1 supports only double precision in the CUDA build");
            }
            GlobalV::ofs_running << " Wave functions of all k-points share one CUDA managed array between host and GPU"
                                 << " (ABACUS_PSI_MANAGED=1); only the current k-point is kept on the GPU." << std::endl;
        }
        this->psi_t = static_cast<void*>(managed_view != nullptr ? managed_view
                                                                 : new psi::Psi<T, Device>(this->psi_cpu[0]));
    }
    else
    {
        this->psi_t = static_cast<void*>(reinterpret_cast<psi::Psi<T, Device>*>(this->psi_cpu));
    }
}

void Setup_Psi_pw::before_runner(const UnitCell& ucell,
                                 const K_Vectors& kv,
                                 const Structure_Factor& sf,
                                 const ModulePW::PW_Basis_K& pw_wfc,
                                 const int& lmaxkb,
                                 const Input_para& inp)
{
    const bool is_gpu = (inp.device == "gpu");
    const bool is_single = (inp.precision == "single");

#if ((defined __CUDA) || (defined __ROCM))
    if (is_gpu)
    {
        if (is_single)
        {
            before_runner_impl<std::complex<float>, base_device::DEVICE_GPU>(ucell, kv, sf, pw_wfc, lmaxkb, inp);
        }
        else
        {
            before_runner_impl<std::complex<double>, base_device::DEVICE_GPU>(ucell, kv, sf, pw_wfc, lmaxkb, inp);
        }
    }
    else
#endif
    {
        if (is_single)
        {
            before_runner_impl<std::complex<float>, base_device::DEVICE_CPU>(ucell, kv, sf, pw_wfc, lmaxkb, inp);
        }
        else
        {
            before_runner_impl<std::complex<double>, base_device::DEVICE_CPU>(ucell, kv, sf, pw_wfc, lmaxkb, inp);
        }
    }
}

template <typename T, typename Device>
void Setup_Psi_pw::update_psi_d_impl()
{
    if (this->precision_type_ == PrecisionType::ComplexFloat)
    {
        const psi::Psi<T, Device>* native = this->get_psi_t<T, Device>();
        psi::Psi<std::complex<double>, Device>* mirror = this->get_psi_d<T, Device>();
        if (mirror == nullptr)
        {
            this->psi_d = new psi::Psi<std::complex<double>, Device>(*native);
        }
        else if (mirror->get_nk() != native->get_nk() || mirror->get_nbands() != native->get_nbands()
                 || mirror->get_nbasis() != native->get_nbasis() || mirror->get_ngk_pointer() != native->get_ngk_pointer())
        {
            // Preserve the object address borrowed by onsite projectors even when storage changes.
            *mirror = psi::Psi<std::complex<double>, Device>(*native);
        }
        else
        {
            const T* source = native->get_pointer() - native->get_psi_bias();
            std::complex<double>* destination = mirror->get_pointer() - mirror->get_psi_bias();
            base_device::memory::cast_memory_op<std::complex<double>, T, Device, Device>()(destination, source, native->size());
            mirror->fix_k(native->get_current_k());
        }
    }
    else
    {
        this->psi_d = static_cast<void*>(reinterpret_cast<psi::Psi<std::complex<double>, Device>*>(this->psi_t));
    }
}

void Setup_Psi_pw::update_psi_d()
{
#if ((defined __CUDA) || (defined __ROCM))
    if (this->device_type_ == base_device::GpuDevice)
    {
        if (this->precision_type_ == PrecisionType::ComplexFloat)
        {
            update_psi_d_impl<std::complex<float>, base_device::DEVICE_GPU>();
        }
        else
        {
            update_psi_d_impl<std::complex<double>, base_device::DEVICE_GPU>();
        }
    }
    else
#endif
    {
        if (this->precision_type_ == PrecisionType::ComplexFloat)
        {
            update_psi_d_impl<std::complex<float>, base_device::DEVICE_CPU>();
        }
        else
        {
            update_psi_d_impl<std::complex<double>, base_device::DEVICE_CPU>();
        }
    }
}

template <typename T, typename Device>
void Setup_Psi_pw::init_impl(hamilt::Hamilt<T, Device>* p_hamilt)
{
    if (!this->already_initpsi)
    {
        auto* p_psi_init = static_cast<psi::PSIPrepare<T, Device>*>(this->p_psi_init);
        p_psi_init->initialize_psi(this->psi_cpu, this->get_psi_t<T, Device>(), p_hamilt, GlobalV::ofs_running);
        this->already_initpsi = true;
    }
}

void Setup_Psi_pw::init(hamilt::HamiltBase* p_hamilt)
{
    if (this->already_initpsi)
    {
        return;
    }

#if ((defined __CUDA) || (defined __ROCM))
    if (this->device_type_ == base_device::GpuDevice)
    {
        if (this->precision_type_ == PrecisionType::ComplexFloat)
        {
            init_impl<std::complex<float>, base_device::DEVICE_GPU>(
                static_cast<hamilt::Hamilt<std::complex<float>, base_device::DEVICE_GPU>*>(p_hamilt));
        }
        else
        {
            init_impl<std::complex<double>, base_device::DEVICE_GPU>(
                static_cast<hamilt::Hamilt<std::complex<double>, base_device::DEVICE_GPU>*>(p_hamilt));
        }
    }
    else
#endif
    {
        if (this->precision_type_ == PrecisionType::ComplexFloat)
        {
            init_impl<std::complex<float>, base_device::DEVICE_CPU>(
                static_cast<hamilt::Hamilt<std::complex<float>, base_device::DEVICE_CPU>*>(p_hamilt));
        }
        else
        {
            init_impl<std::complex<double>, base_device::DEVICE_CPU>(
                static_cast<hamilt::Hamilt<std::complex<double>, base_device::DEVICE_CPU>*>(p_hamilt));
        }
    }
}

// Transfer data from GPU to CPU in pw basis
template <typename T, typename Device>
void Setup_Psi_pw::copy_d2h_impl()
{
    auto* psi_t = this->get_psi_t<T, Device>();
    std::complex<double>* host = this->psi_cpu[0].get_pointer() - this->psi_cpu[0].get_psi_bias();
    const T* device = psi_t->get_pointer() - psi_t->get_psi_bias();
    if (static_cast<const void*>(host) == static_cast<const void*>(device))
    {
        return; // ABACUS_PSI_MANAGED=1 では同じ配列を共有しているので、写す必要がない。
    }
    this->castmem_d2h_impl<T, Device>(host, device, this->psi_cpu[0].size());
}

void Setup_Psi_pw::copy_d2h()
{
    if (this->device_type_ != base_device::GpuDevice)
    {
        return;
    }

#if ((defined __CUDA) || (defined __ROCM))
    if (this->precision_type_ == PrecisionType::ComplexFloat)
    {
        copy_d2h_impl<std::complex<float>, base_device::DEVICE_GPU>();
    }
    else
    {
        copy_d2h_impl<std::complex<double>, base_device::DEVICE_GPU>();
    }
#endif
}

void Setup_Psi_pw::sync_cpu()
{
    if (this->device_type_ == base_device::GpuDevice)
    {
        this->copy_d2h();
    }
    else if (this->precision_type_ == PrecisionType::ComplexFloat)
    {
        this->copy_d2h_impl<std::complex<float>, base_device::DEVICE_CPU>();
    }
}

template <typename T, typename Device>
void Setup_Psi_pw::castmem_d2h_impl(std::complex<double>* dst, const std::complex<double>* src, const std::size_t size)
{
    base_device::memory::cast_memory_op<std::complex<double>, std::complex<double>, base_device::DEVICE_CPU, Device>()(dst, src, size);
}

template <typename T, typename Device>
void Setup_Psi_pw::castmem_d2h_impl(std::complex<double>* dst, const std::complex<float>* src, const std::size_t size)
{
    base_device::memory::cast_memory_op<std::complex<double>, std::complex<float>, base_device::DEVICE_CPU, Device>()(dst, src, size);
}

template <typename T, typename Device>
void Setup_Psi_pw::clean_impl()
{
    if (this->device_type_ == base_device::GpuDevice || this->precision_type_ == PrecisionType::ComplexFloat)
    {
        delete this->get_psi_t<T, Device>();
    }
    if (this->precision_type_ == PrecisionType::ComplexFloat)
    {
        delete this->get_psi_d<T, Device>();
    }

    delete this->psi_cpu;
    delete this->p_psi_init;
}

void Setup_Psi_pw::clean()
{
#if ((defined __CUDA) || (defined __ROCM))
    if (this->device_type_ == base_device::GpuDevice)
    {
        if (this->precision_type_ == PrecisionType::ComplexFloat)
        {
            clean_impl<std::complex<float>, base_device::DEVICE_GPU>();
        }
        else
        {
            clean_impl<std::complex<double>, base_device::DEVICE_GPU>();
        }
    }
    else
#endif
    {
        if (this->precision_type_ == PrecisionType::ComplexFloat)
        {
            clean_impl<std::complex<float>, base_device::DEVICE_CPU>();
        }
        else
        {
            clean_impl<std::complex<double>, base_device::DEVICE_CPU>();
        }
    }
}

template class psi::PSIPrepare<std::complex<float>, base_device::DEVICE_CPU>;
template class psi::PSIPrepare<std::complex<double>, base_device::DEVICE_CPU>;

template void Setup_Psi_pw::before_runner_impl<std::complex<float>, base_device::DEVICE_CPU>(const UnitCell&,
                                                                                             const K_Vectors&,
                                                                                             const Structure_Factor&,
                                                                                             const ModulePW::PW_Basis_K&,
                                                                                             const int&,
                                                                                             const Input_para&);

template void Setup_Psi_pw::before_runner_impl<std::complex<double>, base_device::DEVICE_CPU>(const UnitCell&,
                                                                                              const K_Vectors&,
                                                                                              const Structure_Factor&,
                                                                                              const ModulePW::PW_Basis_K&,
                                                                                              const int&,
                                                                                              const Input_para&);

template void Setup_Psi_pw::init_impl<std::complex<float>, base_device::DEVICE_CPU>(
    hamilt::Hamilt<std::complex<float>, base_device::DEVICE_CPU>*);

template void Setup_Psi_pw::init_impl<std::complex<double>, base_device::DEVICE_CPU>(
    hamilt::Hamilt<std::complex<double>, base_device::DEVICE_CPU>*);

template void Setup_Psi_pw::update_psi_d_impl<std::complex<float>, base_device::DEVICE_CPU>();

template void Setup_Psi_pw::update_psi_d_impl<std::complex<double>, base_device::DEVICE_CPU>();

template void Setup_Psi_pw::clean_impl<std::complex<float>, base_device::DEVICE_CPU>();

template void Setup_Psi_pw::clean_impl<std::complex<double>, base_device::DEVICE_CPU>();

template void Setup_Psi_pw::castmem_d2h_impl<std::complex<float>, base_device::DEVICE_CPU>(std::complex<double>*,
                                                                                           const std::complex<float>*,
                                                                                           const std::size_t);

template void Setup_Psi_pw::castmem_d2h_impl<std::complex<float>, base_device::DEVICE_CPU>(std::complex<double>*,
                                                                                           const std::complex<double>*,
                                                                                           const std::size_t);

template void Setup_Psi_pw::castmem_d2h_impl<std::complex<double>, base_device::DEVICE_CPU>(std::complex<double>*,
                                                                                            const std::complex<float>*,
                                                                                            const std::size_t);

template void Setup_Psi_pw::castmem_d2h_impl<std::complex<double>, base_device::DEVICE_CPU>(std::complex<double>*,
                                                                                            const std::complex<double>*,
                                                                                            const std::size_t);

#if ((defined __CUDA) || (defined __ROCM))
template class psi::PSIPrepare<std::complex<float>, base_device::DEVICE_GPU>;
template class psi::PSIPrepare<std::complex<double>, base_device::DEVICE_GPU>;

template void Setup_Psi_pw::before_runner_impl<std::complex<float>, base_device::DEVICE_GPU>(const UnitCell&,
                                                                                             const K_Vectors&,
                                                                                             const Structure_Factor&,
                                                                                             const ModulePW::PW_Basis_K&,
                                                                                             const int&,
                                                                                             const Input_para&);

template void Setup_Psi_pw::before_runner_impl<std::complex<double>, base_device::DEVICE_GPU>(const UnitCell&,
                                                                                              const K_Vectors&,
                                                                                              const Structure_Factor&,
                                                                                              const ModulePW::PW_Basis_K&,
                                                                                              const int&,
                                                                                              const Input_para&);

template void Setup_Psi_pw::init_impl<std::complex<float>, base_device::DEVICE_GPU>(
    hamilt::Hamilt<std::complex<float>, base_device::DEVICE_GPU>*);

template void Setup_Psi_pw::init_impl<std::complex<double>, base_device::DEVICE_GPU>(
    hamilt::Hamilt<std::complex<double>, base_device::DEVICE_GPU>*);

template void Setup_Psi_pw::update_psi_d_impl<std::complex<float>, base_device::DEVICE_GPU>();

template void Setup_Psi_pw::update_psi_d_impl<std::complex<double>, base_device::DEVICE_GPU>();

template void Setup_Psi_pw::copy_d2h_impl<std::complex<float>, base_device::DEVICE_GPU>();

template void Setup_Psi_pw::copy_d2h_impl<std::complex<double>, base_device::DEVICE_GPU>();

template void Setup_Psi_pw::clean_impl<std::complex<float>, base_device::DEVICE_GPU>();

template void Setup_Psi_pw::clean_impl<std::complex<double>, base_device::DEVICE_GPU>();

template void Setup_Psi_pw::castmem_d2h_impl<std::complex<float>, base_device::DEVICE_GPU>(std::complex<double>*,
                                                                                           const std::complex<float>*,
                                                                                           const std::size_t);

template void Setup_Psi_pw::castmem_d2h_impl<std::complex<float>, base_device::DEVICE_GPU>(std::complex<double>*,
                                                                                           const std::complex<double>*,
                                                                                           const std::size_t);

template void Setup_Psi_pw::castmem_d2h_impl<std::complex<double>, base_device::DEVICE_GPU>(std::complex<double>*,
                                                                                            const std::complex<float>*,
                                                                                            const std::size_t);

template void Setup_Psi_pw::castmem_d2h_impl<std::complex<double>, base_device::DEVICE_GPU>(std::complex<double>*,
                                                                                            const std::complex<double>*,
                                                                                            const std::size_t);
#endif
