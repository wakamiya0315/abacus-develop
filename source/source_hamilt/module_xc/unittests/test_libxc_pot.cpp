#include "../xc_functional.h"
#include "../libxc_abacus.h"
#include "source_base/constants.h"
#include "source_base/parallel_comm.h"
#include "source_base/parallel_global.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"
#include "source_estate/module_charge/charge.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

// Focused PW/XC target: the fixture owns all charge buffers. The Fourier
// transforms, density derivatives, Libxc evaluation and assembly are real.
Charge::Charge() {}
Charge::~Charge() {}
UnitCell::UnitCell() {}
UnitCell::~UnitCell() {}
Magnetism::Magnetism() {}
Magnetism::~Magnetism() {}
SepPot::SepPot() {}
SepPot::~SepPot() {}
Sep_Cell::Sep_Cell() noexcept {}
Sep_Cell::~Sep_Cell() noexcept {}

namespace
{
int test_rank = 0;
int test_size = 1;
typedef std::tuple<double, double, ModuleBase::matrix, ModuleBase::matrix> MetaResult;

class DensityOnlyMeta : public testing::TestWithParam<std::string>
{
  protected:
    ModulePW::PW_Basis pw;
    UnitCell cell;
    Charge charge;
    std::vector<double> density;
    std::vector<double> direction;
    std::vector<double> core;
    std::vector<std::complex<double>> core_g;
    std::vector<std::complex<double>> density_g;
    std::array<double*, 1> rho_ptr;
    std::array<std::complex<double>*, 1> rhog_ptr;

    void SetUp() override
    {
        pw.initmpi(test_size, test_rank, MPI_COMM_WORLD);
        const ModuleBase::Matrix3 lattice(1, 0, 0, 0, 1, 0, 0, 0, 1);
        pw.initgrids(7.0, lattice, 18, 18, 18);
        pw.initparameters(false, 80.0, 2, false);
        pw.setuptransform();
        pw.collect_local_pw();
        cell.omega = pw.omega;
        cell.tpiba = pw.tpiba;
        density.resize(pw.nrxx);
        direction.resize(pw.nrxx);
        core.resize(pw.nrxx, 0.0);
        core_g.resize(pw.npw, 0.0);
        density_g.resize(pw.npw, 0.0);
        rho_ptr[0] = density.data();
        rhog_ptr[0] = density_g.data();
        charge.rhopw = &pw;
        charge.nrxx = pw.nrxx;
        charge.nxyz = pw.nxyz;
        charge.ngmc = pw.npw;
        charge.nspin = 1;
        charge.rho = rho_ptr.data();
        charge.rhog = rhog_ptr.data();
        charge.rho_core = core.data();
        charge.rhog_core = core_g.data();
        charge.kin_r = nullptr;
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            const double x = ModuleBase::TWO_PI * (ir / (pw.ny * pw.nplane)) / pw.nx;
            const double y = ModuleBase::TWO_PI * ((ir / pw.nplane) % pw.ny) / pw.ny;
            const double z = ModuleBase::TWO_PI * (ir % pw.nplane + pw.startz_current) / pw.nz;
            const double phase = y + z;
            const double phase2 = 2.0 * x + y;
            density[ir] = 0.04 * (1.0 + 0.15 * std::cos(x) + 0.08 * std::sin(phase));
            direction[ir] = 0.005 * std::cos(x) + 0.003 * std::sin(phase) + 0.002 * std::cos(phase2);
        }
        const std::string name = GetParam();
        XC_Functional::set_xc_type(name);
        const std::vector<int>& ids = XC_Functional::get_func_id();
        const bool density_only = XC_Functional::is_density_only_meta_gga(ids);
        ASSERT_TRUE(density_only);
    }

    MetaResult evaluate()
    {
        const std::vector<int>& ids = XC_Functional::get_func_id();
        return XC_Functional_Libxc::v_xc_meta(ids, pw.nrxx, cell.omega, cell.tpiba, &charge, 1, 0.0, 0.0);
    }

    double integral(const ModuleBase::matrix& potential, const std::vector<double>& values)
    {
        double sum = 0.0;
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            sum += potential(0, ir) * values[ir];
        }
        Parallel_Reduce::reduce_pool(sum);
        return sum * cell.omega / pw.nxyz;
    }
};

TEST_P(DensityOnlyMeta, UniformDensityAndTauIndependence)
{
    std::fill(density.begin(), density.end(), 0.04);
    const MetaResult without_tau = evaluate();
    EXPECT_LT(std::get<0>(without_tau), -1.0);
    EXPECT_LT(std::get<2>(without_tau)(0, 0), -0.1);
    const double integral_vrho = integral(std::get<2>(without_tau), density);
    EXPECT_NEAR(std::get<1>(without_tau), integral_vrho, 1e-11);
    std::vector<double> tau(pw.nrxx, 0.0);
    std::array<double*, 1> tau_ptr = {{tau.data()}};
    charge.kin_r = tau_ptr.data();
    const MetaResult zero_tau = evaluate();
    std::fill(tau.begin(), tau.end(), 123.0);
    const MetaResult arbitrary_tau = evaluate();
    EXPECT_DOUBLE_EQ(std::get<0>(without_tau), std::get<0>(zero_tau));
    EXPECT_DOUBLE_EQ(std::get<0>(without_tau), std::get<0>(arbitrary_tau));
    for (int ir = 0; ir < pw.nrxx; ++ir)
    {
        EXPECT_DOUBLE_EQ(std::get<2>(without_tau)(0, ir), std::get<2>(zero_tau)(0, ir));
        EXPECT_DOUBLE_EQ(std::get<2>(without_tau)(0, ir), std::get<2>(arbitrary_tau)(0, ir));
        EXPECT_DOUBLE_EQ(std::get<3>(without_tau)(0, ir), 0.0);
    }
    charge.kin_r = nullptr;
    // Larger steps plus Richardson extrapolation avoid cancellation in a
    // sum of thousands of identical grid values without relaxing tolerance.
    double central[2];
    for (int level = 0; level < 2; ++level)
    {
        const double epsilon = 1e-3 / (1 << level);
        const double plus_density = 0.04 * (1.0 + epsilon);
        const double minus_density = 0.04 * (1.0 - epsilon);
        std::fill(density.begin(), density.end(), plus_density);
        const MetaResult plus = evaluate();
        std::fill(density.begin(), density.end(), minus_density);
        const MetaResult minus = evaluate();
        central[level] = (std::get<0>(plus) - std::get<0>(minus)) / (2.0 * epsilon);
    }
    const double richardson = (4.0 * central[1] - central[0]) / 3.0;
    EXPECT_NEAR(integral_vrho, richardson, 2e-8);
    if (test_rank == 0)
    {
        std::cout << std::setprecision(16) << "XC_UNIFORM_FD," << GetParam() << ","
                  << integral_vrho << "," << richardson << std::endl;
    }
    std::fill(density.begin(), density.end(), 0.04);
}

TEST_P(DensityOnlyMeta, PeriodicDensityDirectionalDerivative)
{
    const MetaResult reference = evaluate();
    const double analytic = integral(std::get<2>(reference), direction);
    EXPECT_GT(std::abs(analytic), 1e-5);
    const std::vector<double> original = density;
    double previous = 1.0;
    for (int level = 0; level < 6; ++level)
    {
        const double epsilon = 0.02 / (1 << level);
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            density[ir] = original[ir] + epsilon * direction[ir];
        }
        const MetaResult plus = evaluate();
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            density[ir] = original[ir] - epsilon * direction[ir];
        }
        const MetaResult minus = evaluate();
        const double finite_difference = (std::get<0>(plus) - std::get<0>(minus)) / (2.0 * epsilon);
        const double error = std::abs(finite_difference - analytic);
        if (test_rank == 0)
        {
            std::cout << std::setprecision(16) << "XC_FD," << GetParam() << "," << epsilon << ","
                      << analytic << "," << finite_difference << "," << error << std::endl;
        }
        if (level > 0 && level <= 3)
        {
            EXPECT_LE(error, 0.4 * previous + 2e-10);
        }
        if (level == 5)
        {
            EXPECT_LT(error, 2e-8);
        }
        previous = error;
    }
    std::copy(original.begin(), original.end(), density.begin());
}

TEST_P(DensityOnlyMeta, IsotropicStressDerivative)
{
    const MetaResult reference = evaluate();
    double ignored_energy = 0.0;
    double ignored_vtxc = 0.0;
    ModuleBase::matrix unused;
    std::vector<double> gradient_stress;
    XC_Functional::gradcorr(ignored_energy, ignored_vtxc, unused, &charge, &pw, &cell,
                            gradient_stress, true, 1, false, false, 0, 0.0, 0.0);
    ASSERT_EQ(gradient_stress.size(), 9U);
    double trace = gradient_stress[0] + gradient_stress[4] + gradient_stress[8];
    Parallel_Reduce::reduce_pool(trace);
    trace /= pw.nxyz;
    trace -= 3.0 * (std::get<0>(reference) - std::get<1>(reference)) / cell.omega;
    const std::vector<double> original_density = density;
    const std::vector<ModuleBase::Vector3<double>> original_g(pw.gcar, pw.gcar + pw.npw);
    const double original_volume = cell.omega;
    const double epsilon = 1e-5;
    double energies[2];
    for (int sign = 0; sign < 2; ++sign)
    {
        const double strain = sign == 0 ? epsilon : -epsilon;
        const double stretch = 1.0 + strain;
        const double determinant = stretch * stretch * stretch;
        cell.omega = original_volume * determinant;
        for (int ir = 0; ir < pw.nrxx; ++ir)
        {
            density[ir] = original_density[ir] / determinant;
        }
        for (int ig = 0; ig < pw.npw; ++ig)
        {
            pw.gcar[ig] = original_g[ig] / stretch;
        }
        const MetaResult result = evaluate();
        energies[sign] = std::get<0>(result);
    }
    const double finite_difference = -(energies[0] - energies[1]) / (2.0 * epsilon * original_volume);
    EXPECT_NEAR(trace, finite_difference, 2e-8);
    if (test_rank == 0)
    {
        std::cout << std::setprecision(16) << "XC_STRESS_FD," << GetParam() << "," << trace << ","
                  << finite_difference << std::endl;
    }
    std::copy(original_density.begin(), original_density.end(), density.begin());
    std::copy(original_g.begin(), original_g.end(), pw.gcar);
    cell.omega = original_volume;
}

TEST(DensityOnlyClassification, RejectsOrbitalDependentMetaGGAs)
{
    std::vector<std::string> rejected = {"SCAN", "PBE"};
#if defined(XC_MGGA_X_R2SCAN) && defined(XC_MGGA_C_R2SCAN)
    rejected.push_back("MGGA_X_R2SCAN+MGGA_C_R2SCAN");
#endif
    for (const std::string& name : rejected)
    {
        XC_Functional::set_xc_type(name);
        const std::vector<int>& ids = XC_Functional::get_func_id();
        const bool density_only = XC_Functional::is_density_only_meta_gga(ids);
        EXPECT_FALSE(density_only);
    }
}

#if defined(XC_MGGA_X_R2SCANL) && defined(XC_MGGA_C_R2SCANL)
INSTANTIATE_TEST_SUITE_P(Deorbitalized, DensityOnlyMeta,
    testing::Values("SCANL", "MGGA_X_R2SCANL+MGGA_C_R2SCANL"));
#else
INSTANTIATE_TEST_SUITE_P(Deorbitalized, DensityOnlyMeta, testing::Values("SCANL"));
#endif
} // namespace

int main(int argc, char** argv)
{
    int threads = 1;
    Parallel_Global::read_pal_param(argc, argv, test_size, threads, test_rank);
    POOL_WORLD = MPI_COMM_WORLD;
    KP_WORLD = MPI_COMM_NULL;
    INT_BGROUP = MPI_COMM_NULL;
    BP_WORLD = MPI_COMM_NULL;
    GRID_WORLD = MPI_COMM_NULL;
    DIAG_WORLD = MPI_COMM_NULL;
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    Parallel_Global::finalize_mpi();
    return result;
}
