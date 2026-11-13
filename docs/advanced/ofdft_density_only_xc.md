# Density-only meta-GGA exchange-correlation in OFDFT

With Libxc enabled, the orbital-free solver accepts SCAN-L and r2SCAN-L in
addition to LDA and GGA. These deorbitalized functionals depend on density,
its gradient and its Laplacian, rather than the orbital kinetic-energy density.
A strict allowlist of their exchange and correlation components is checked
against the installed Libxc metadata. Functionals requiring orbital tau,
including SCAN and r2SCAN, and hybrid functionals remain unsupported by OFDFT.

For example, a non-spin-polarized, fixed-cell calculation can use:

```
INPUT_PARAMETERS
calculation       scf
esolver_type      ofdft
basis_type        pw
nspin             1
dft_functional    MGGA_X_R2SCANL+MGGA_C_R2SCANL
of_kinetic        wt
of_method         tn
of_conv           both
of_tole           1e-8
of_tolp           1e-5
```

Provide a local pseudopotential and the usual structure/grid settings. `SCANL`
is also accepted. The selected Libxc build must contain the requested
functional. This does not extend the existing spin support of the OF solver.

## Potential and kinetic-density handling

For an energy integrand `f(n, sigma, lapl)`, with `sigma = |grad n|^2`, the
multiplicative potential is

```
v_xc = df/dn - div(2 df/dsigma grad n) + laplacian(df/dlapl).
```

The implementation uses the same spectral gradient, divergence and Laplacian
as the KS meta-GGA path, including the final positive Laplacian term and the
Hartree-to-Rydberg conversion. The XC energy and potential do not read or
screen on the orbital/Thomas-Fermi tau buffer for these density-only
functionals. A positive uniform density is a valid input even when its
gradient and Laplacian vanish. The stress path likewise supplies no orbital
tau for these functionals and retains the explicit Laplacian stress term.

`ked_flag` remains a legacy meta-GGA dispatch and buffer-allocation flag: in
particular, `PotXC` uses it to select the meta-GGA interface. It must not be
interpreted as proof that a functional needs orbital tau. The dedicated
`is_density_only_meta_gga` query provides the restricted distinction without
introducing another mutable selection flag.

OFDFT initializes or restarts from density without requesting a tau file.
The common output path does not write a fictitious orbital `tau.cube` or tau
restart for OFDFT. KEDF-based ELF output is a separate existing facility.
KS calculations can still compute and output their orbital kinetic density.

## Verification and numerical scope

The focused `MODULE_HAMILT_XCTest_DENSITY_ONLY_MGGA` target uses real periodic
Fourier transforms and the production Libxc energy/potential assembly. It
checks uniform-density XC, independence from a missing/zero/arbitrary tau
buffer, density directional derivatives, isotropic XC stress derivatives,
and rejection of orbital-dependent meta-GGAs by the classifier.

These checks establish implementation consistency, not the accuracy of a
chosen KEDF or pseudopotential. Density minimization still requires a suitable
grid and optimization setup; large Laplacian responses can make a high-cutoff
calculation stiff. Record convergence of both energy and potential, electron
normalization, and the chosen cutoff rather than treating a normal program
exit as evidence of convergence.

The small HQLPP Al Autotest fixture uses a 20 Ry cutoff and an odd FFT grid.
It establishes a converged stationary density at that discretization, not
cutoff convergence or uniqueness of the minimum. In trial calculations,
50 Ry minimizations were not converged within 300 iterations, and different
initial densities reached different stationary solutions at 20 Ry.
The OF solver also symmetrizes file-based initial densities when symmetry is
enabled, while its optimization does not enforce that symmetry. Use the same
`symmetry 0` setting for an unchanged-density restart check; otherwise a
restart may intentionally alter the initial density and enter another basin.
