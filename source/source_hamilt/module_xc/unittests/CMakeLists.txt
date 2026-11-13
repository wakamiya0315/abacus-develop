addtest(
  TARGET MODULE_HAMILT_XCTest_DENSITY_ONLY_MGGA
  LIBS xc_ planewave parameter MPI::MPI_CXX Libxc::xc psi device container base
  SOURCES test_libxc_pot.cpp)
set_tests_properties(MODULE_HAMILT_XCTest_DENSITY_ONLY_MGGA
                     PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")
