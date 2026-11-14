#include "read_pp.h"
#include "source_base/atom_in.h"
#include "source_base/element_name.h"
#include "source_base/global_variable.h"
#include "source_base/tool_quit.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

int Pseudopot_upf::read_pseudo_blps(std::ifstream &ifs, Atom_pseudo& pp)
{
    // double bohr2a = 0.529177249;
    pp.nlcc = false;
    pp.tvanp = false;
    pp.has_so = false;

    pp.nbeta = 0;
    pp.kkbeta = 0;
    pp.lll = std::vector<int>(pp.nbeta, 0);
    pp.betar.create(0, 0);
    pp.dion.create(pp.nbeta, pp.nbeta);

    pp.nchi = 0;
    pp.nn = std::vector<int>(pp.nchi, 0);
    pp.jchi = std::vector<double>(pp.nchi, 0.0);
    pp.jjj = std::vector<double>(pp.nchi, 0.0);

    ifs >> pp.psd;
    // if(!SCAN_BEGIN(ifs,"BLPS")) WARNING_QUIT("read_pp_blps","Find no PP_HEADER");
    ifs.ignore(300, '\n');

    double zatom = 0.0;
    double zion = 0.0;
    ifs >> zatom >> zion;
    pp.zv = zion;
    ifs.ignore(300, '\n');

    atom_in ai;
    for (auto each_type:  ModuleBase::element_name)
    {
        if (zatom == ai.atom_Z[each_type])
        {
            pp.psd = each_type;
            break;
        }
    }

    int pspcod, pspxc, lloc, r2well;
    ifs >> pspcod >> pspxc >> pp.lmax >> lloc >> pp.mesh >> r2well;
    this->mesh_changed = false;
    if (pp.mesh%2 == 0)
    {
        pp.mesh -= 1;
        this->mesh_changed = true;
    }

    if (pspxc == 2)
    {
        pp.xc_func = "PZ";
    }
    else if (pspxc == 11)
    {
        pp.xc_func = "PBE";
    }
    else
    {
        std::string msg = "Unknown pspxc: " + std::to_string(pspxc);
        ModuleBase::WARNING_QUIT("Pseudopot_upf::read_pseudo_blps", msg);
    }

    if (pspcod == 8)
    {
        for (int i = 0; i < 5; ++i)
        {
            ifs.ignore(300, '\n');
        }
    }
    else if (pspcod == 6)
    {
        for (int i = 0; i < 17; ++i)
        {
            ifs.ignore(300, '\n');
        }
    }
    else
    {
        std::string msg = "Unknown pspcod: " + std::to_string(pspcod);
        ModuleBase::WARNING_QUIT("Pseudopot_upf::read_pseudo_blps", msg);
    }

    assert(pp.mesh > 0);

    pp.r = std::vector<double>(pp.mesh, 0.0); // Bohr
    pp.rab = std::vector<double>(pp.mesh, 0.0);
    pp.vloc_at = std::vector<double>(pp.mesh, 0.0); // Hartree
    int num = 0;
    if (pspcod == 8)
    {
        for(int i = 0;i < pp.mesh; ++i)
        {
            ifs >> num >> pp.r[i] >> pp.vloc_at[i];
            pp.vloc_at[i] = pp.vloc_at[i]*2; // Hartree to Ry
        }
    }
    else if (pspcod == 6)
    {
        double temp = 0.;
        for(int i = 0;i < pp.mesh; ++i)
        {
            ifs >> num >> pp.r[i] >> temp >> pp.vloc_at[i];
            pp.vloc_at[i] = pp.vloc_at[i]*2; // Hartree to Ry
        }
    }
    pp.rab[0] = pp.r[1] - pp.r[0];
    for(int i = 1; i < pp.mesh - 1; ++i)
    {
        pp.rab[i] = (pp.r[i+1] - pp.r[i-1])/2.0;
    }
    pp.rab[pp.mesh - 1] = pp.r[pp.mesh - 1] - pp.r[pp.mesh - 2];

    pp.rho_at = std::vector<double>(pp.mesh, 0.0);
    double charge = zion/pp.r[pp.mesh - 1];
    for(int i = 0;i < pp.mesh; ++i)
    {
        pp.rho_at[i] = charge;
    }
    return 0;
}

//parameters
//read_pp.h <--> blps_real
//nv             -
//psd            head
//pp_type(NC or US) -
//tvanp          False
//nlcc           False
//dft            pspxc 2->lda, 11->gga
//zp             zion
//etotps         -
//ecutwfc        -
//ecutrho        -
//lmax           lmax
//mesh           mmax
//nwfc           -
//nbeta          -
//els            -
//lchi           -
//oc             -

//rab            rab[ir]=(r[ir+1]-r[ir-1])/2.0
//rho_atc(nonlocal) -
//vloc
//chi            -
//rho_at         -

// lll            -
// kbeta         -
// beta           -
// dion           -

//nn             -
//jchi           -
//jjj            -
//nd             -

bool Pseudopot_upf::read_blps_rhoatom(const std::string& fn, Atom_pseudo& pp)
{
    std::ifstream ifs(fn.c_str());
    if (!ifs)
    {
        return false;
    }
    std::vector<double> r_in;
    std::vector<double> rho_in;
    std::string line;
    while (std::getline(ifs, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        std::istringstream iss(line);
        double r = 0.0;
        double rho = 0.0;
        if (iss >> r >> rho)
        {
            r_in.push_back(r);
            rho_in.push_back(rho);
        }
    }
    if (r_in.size() < 2)
    {
        ModuleBase::WARNING_QUIT("read_blps_rhoatom", "too few points in " + fn);
    }
    for (std::size_t i = 1; i < r_in.size(); ++i)
    {
        if (!(r_in[i] > r_in[i - 1]))
        {
            ModuleBase::WARNING_QUIT("read_blps_rhoatom", "r must increase in " + fn);
        }
    }
    // PPのメッシュへ線形補間する（範囲外は0）。
    std::size_t j = 0;
    for (int i = 0; i < pp.mesh; ++i)
    {
        const double r = pp.r[i];
        if (r < r_in.front() || r > r_in.back())
        {
            pp.rho_at[i] = 0.0;
            continue;
        }
        while (j + 1 < r_in.size() - 1 && r_in[j + 1] < r)
        {
            ++j;
        }
        const double t = (r - r_in[j]) / (r_in[j + 1] - r_in[j]);
        pp.rho_at[i] = (1.0 - t) * rho_in[j] + t * rho_in[j + 1];
    }
    // 積分 ∫ 4πr²ρ dr が価電子数 zv になるよう規格化する（rab は read_pseudo_blps と同じ台形の重み）。
    double charge = 0.0;
    for (int i = 0; i < pp.mesh; ++i)
    {
        charge += pp.rho_at[i] * pp.rab[i];
    }
    if (!(charge > 0.0))
    {
        ModuleBase::WARNING_QUIT("read_blps_rhoatom", "non-positive charge in " + fn);
    }
    for (int i = 0; i < pp.mesh; ++i)
    {
        pp.rho_at[i] *= pp.zv / charge;
    }
    GlobalV::ofs_running << " Atomic valence density for the initial charge read from " << fn << " (integral "
                         << charge << " before scaling to zv = " << pp.zv << ")" << std::endl;
    return true;
}
