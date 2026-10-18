/* Native model parsing, radial evaluation, graph evaluation and differentiation
   remain in MLMTPR. This file only translates LAMMPS data and communication. */
#include "pair_sus2_superlinear.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"
#include "sus2-native/dev_src/mtpr.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace LAMMPS_NS;

namespace {
const std::vector<std::string> &chemical_symbols()
{
  static const std::vector<std::string> names = {
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti",
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
    "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
    "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
    "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg",
    "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
  return names;
}

std::vector<std::string> parse_elements(const char *arg)
{
  std::vector<std::string> result;
  std::istringstream input(arg);
  std::string value;
  while (std::getline(input, value, ',')) {
    if (value.empty() || std::find(chemical_symbols().begin() + 1, chemical_symbols().end(), value) == chemical_symbols().end())
      throw std::runtime_error("model_elements must be comma-separated chemical symbols in native model species order");
    if (std::find(result.begin(), result.end(), value) != result.end())
      throw std::runtime_error("model_elements contains a duplicate symbol");
    result.push_back(value);
  }
  if (result.empty() || std::string(arg).back() == ',')
    throw std::runtime_error("model_elements must not be empty or end with a comma");
  return result;
}
}

struct PairSUS2Superlinear::Native {
  std::unique_ptr<MLMTPR> model;
  Configuration cfg;
  Neighborhoods neighborhoods;
};

PairSUS2Superlinear::PairSUS2Superlinear(LAMMPS *lmp) : Pair(lmp), native_(new Native)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
  // Always use the native edge virial, including when LAMMPS requests F dot r.
  no_virial_fdotr_compute = 1;
}

PairSUS2Superlinear::~PairSUS2Superlinear()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairSUS2Superlinear::allocate()
{
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "sus2/superlinear:setflag");
  memory->create(cutsq, n + 1, n + 1, "sus2/superlinear:cutsq");
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j) setflag[i][j] = 0;
}

void PairSUS2Superlinear::settings(int narg, char **arg)
{
  if (native_->model)
    error->all(FLERR, "Set sus2/superlinear options before pair_coeff; reissue pair_coeff after changing pair_style");
  try {
    for (int i = 0; i < narg; i += 2) {
      if (i + 1 >= narg) throw std::runtime_error("sus2/superlinear options require key/value pairs");
      const std::string key(arg[i]);
      if (key == "model_elements") model_elements_ = parse_elements(arg[i + 1]);
      else if (key == "radial_mode") {
        radial_mode_ = arg[i + 1];
        if (radial_mode_ != "auto" && radial_mode_ != "direct" && radial_mode_ != "table")
          throw std::runtime_error("radial_mode must be auto, direct or table");
      } else if (key == "tabstep") {
        size_t consumed = 0;
        const std::string value(arg[i + 1]);
        tabstep_ = std::stod(value, &consumed);
        if (consumed != value.size() || !std::isfinite(tabstep_) || tabstep_ <= 0)
          throw std::runtime_error("tabstep must be a finite positive distance");
      } else if (key == "table_max_mb") {
        size_t consumed = 0;
        const std::string value(arg[i + 1]);
        const double mb = std::stod(value, &consumed);
        if (consumed != value.size() || !std::isfinite(mb) || mb < 0 ||
            mb >= static_cast<double>(std::numeric_limits<std::size_t>::max()) / (1024 * 1024))
          throw std::runtime_error("table_max_mb must be a finite nonnegative memory budget");
        table_max_bytes_ = static_cast<std::size_t>(mb * 1024 * 1024);
      } else throw std::runtime_error("Unknown sus2/superlinear option: " + key);
    }
  } catch (const std::exception &e) { error->all(FLERR, "{}", e.what()); }
}

void PairSUS2Superlinear::coeff(int narg, char **arg)
{
  if (narg != atom->ntypes + 3 || std::strcmp(arg[0], "*") || std::strcmp(arg[1], "*"))
    error->all(FLERR, "Expected pair_coeff * * model.sus2 symbol1 ... symbolN for sus2/superlinear");
  try {
    std::unique_ptr<MLMTPR> model(new MLMTPR(arg[2]));
    if (!model->IsSHPotential()) throw std::runtime_error("sus2/superlinear requires a native SUS2-SH model");
    if (!model->HasCompleteParameters()) throw std::runtime_error("SUS2 model has incomplete parameters");
    const int species = model->species_count;
    std::vector<std::string> metadata;
    if (model->HasZBL()) {
      const auto &numbers = model->ZBL().AtomicNumbers();
      if (static_cast<int>(numbers.size()) != species)
        throw std::runtime_error("ZBL element metadata does not match native species count");
      for (int z : numbers) {
        if (z < 1 || z >= static_cast<int>(chemical_symbols().size()))
          throw std::runtime_error("Invalid atomic number in native ZBL metadata");
        metadata.push_back(chemical_symbols()[z]);
      }
    }
    if (model_elements_.empty()) model_elements_ = metadata;
    if (static_cast<int>(model_elements_.size()) != species)
      throw std::runtime_error("Specify pair_style sus2/superlinear model_elements H,O,... in native model species order; symbol count must match species_count");
    if (!metadata.empty() && model_elements_ != metadata)
      throw std::runtime_error("model_elements disagrees with native ZBL atomic-number metadata");
    type_map_.assign(atom->ntypes + 1, -1);
    for (int i = 1; i <= atom->ntypes; ++i) {
      const auto found = std::find(model_elements_.begin(), model_elements_.end(), arg[i + 2]);
      if (found == model_elements_.end())
        throw std::runtime_error("Unmapped pair_coeff symbol: " + std::string(arg[i + 2]) + "; NULL/hybrid mappings are not supported");
      type_map_[i] = static_cast<int>(found - model_elements_.begin());
    }
    const RadialTableMode radial = radial_mode_ == "direct" ? RadialTableMode::Direct :
      (radial_mode_ == "table" ? RadialTableMode::Table : RadialTableMode::Auto);
    model->ConfigureDistributedRadialTable(radial, tabstep_, table_max_bytes_);
    cutoff_ = model->CutOff();
    if (model->HasZBL()) cutoff_ = std::max(cutoff_, model->ZBL().MaxOuterCutoff());
    comm_forward = model->DistributedForwardWidth();
    comm_reverse = model->DistributedReverseWidth();
    native_->model = std::move(model);
    radial_reported_ = false;
    if (!allocated) allocate();
    for (int i = 1; i <= atom->ntypes; ++i)
      for (int j = 1; j <= atom->ntypes; ++j) setflag[i][j] = 1;
  // Loading/allocating a model can fail on only one rank (e.g. local I/O).
  } catch (const MlipException &e) { error->one(FLERR, "Native SUS2 model: {}", e.What()); }
    catch (const std::exception &e) { error->one(FLERR, "{}", e.what()); }
}

void PairSUS2Superlinear::init_style()
{
  if (std::strcmp(update->unit_style, "metal"))
    error->all(FLERR, "sus2/superlinear requires units metal (native eV and Angstrom model)");
  if (force->newton_pair != 1) error->all(FLERR, "sus2/superlinear requires newton pair on");
  if (!native_->model) error->all(FLERR, "sus2/superlinear requires pair_coeff");
  // A many-body environment must contain every type pair. Hybrid styles can
  // silently install ijskip filters even when the original pair_coeff maps
  // every type, and inum alone cannot detect the missing neighbor entries.
  if (force->pair_style && std::strncmp(force->pair_style, "hybrid", 6) == 0)
    error->all(FLERR, "sus2/superlinear does not support hybrid pair styles");
  if (neighbor->includegroup || neighbor->nex_type || neighbor->nex_group || neighbor->nex_mol)
    error->all(FLERR, "sus2/superlinear requires complete neighborhoods; neigh_modify include/exclude is not supported");
  if (atom->molecular)
    for (int k = 1; k <= 3; ++k)
      if (force->special_lj[k] != 1.0)
        error->all(FLERR, "sus2/superlinear requires special_bonds lj 1 1 1 for molecular atom styles");
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairSUS2Superlinear::init_one(int i, int j)
{
  if (!setflag[i][j]) error->all(FLERR, "All sus2/superlinear pair coefficients must be set");
  return cutoff_;
}

void PairSUS2Superlinear::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (vflag_atom || cvflag_atom)
    error->all(FLERR, "sus2/superlinear supports global virial only; per-atom and centroid virials are not implemented");
  try {
    auto &cfg = native_->cfg;
    auto &nbhs = native_->neighborhoods;
    const int nlocal = atom->nlocal, nall = nlocal + atom->nghost;
    cfg.resize(nall);
    nbhs.resize(nall);
    nbhs.cutoff = cutoff_;
    cfg.unique_elems.clear();
    // LAMMPS already supplies periodic image coordinates. Native neighbor
    // construction is deliberately bypassed; never generate a second halo.
    for (int i = 0; i < nall; ++i) {
      cfg.type(i) = type_map_[atom->type[i]];
      cfg.unique_elems.insert(cfg.type(i));
      for (int a = 0; a < 3; ++a) cfg.pos(i, a) = atom->x[i][a];
      auto &nbh = nbhs[i];
      nbh.my_type = cfg.type(i);
      nbh.count = 0;
      nbh.inds.clear(); nbh.vecs.clear(); nbh.dists.clear(); nbh.types.clear();
    }
    const double cutoff_sq = cutoff_ * cutoff_;
    if (list->inum != nlocal)
      throw std::runtime_error("sus2/superlinear requires a full neighbor list for every owned atom");
    for (int ii = 0; ii < list->inum; ++ii) {
      const int i = list->ilist[ii];
      auto &nbh = nbhs[i];
      for (int jj = 0; jj < list->numneigh[i]; ++jj) {
        const int j = list->firstneigh[i][jj] & NEIGHMASK;
        Vector3 delta;
        double rsq = 0;
        for (int a = 0; a < 3; ++a) {
          delta[a] = atom->x[j][a] - atom->x[i][a];
          rsq += delta[a] * delta[a];
        }
        if (rsq >= cutoff_sq) continue;
        nbh.inds.push_back(j); nbh.vecs.push_back(delta);
        nbh.dists.push_back(std::sqrt(rsq)); nbh.types.push_back(cfg.type(j));
        ++nbh.count;
      }
    }
    auto &model = *native_->model;
    model.BeginDistributedEFS(cfg, nbhs, nlocal);
    if (comm_forward) comm->forward_comm(this);
    model.EvaluateDistributedOuterEFS(cfg, nbhs);
    if (comm_reverse) comm->reverse_comm(this);
    model.FinishDistributedEFS(cfg, nbhs);
    if (!radial_reported_) {
      const auto stats = model.DistributedRadialTableStats();
      if (comm->me == 0)
        utils::logmesg(lmp, "SUS2 native CPU: radial_mode={} tabstep={} forward={} reverse={} "
          "rank0_tables={} bytes={} hits={} direct_fallbacks={} domain_fallbacks={} "
          "memory_fallbacks={} invalid_fallbacks={}\n", radial_mode_, tabstep_,
          comm_forward, comm_reverse, stats.tables, stats.bytes, stats.hits,
          stats.direct_fallbacks, stats.domain_fallbacks, stats.memory_fallbacks,
          stats.invalid_fallbacks);
      radial_reported_ = true;
    }
    // Forces include ghost contributions. LAMMPS performs its normal force
    // reverse after all pair styles finish; the reverse above carries adjoints.
    for (int i = 0; i < nall; ++i)
      for (int a = 0; a < 3; ++a) atom->f[i][a] += cfg.force(i, a);
    if (eflag_global) eng_vdwl += cfg.energy;
    if (eflag_atom)
      for (int i = 0; i < nlocal; ++i) eatom[i] += cfg.cal_se[i];
    if (vflag_global) {
      // Native W_ab = -sum_edge dE/dr_a * r_b is the configurational
      // LAMMPS virial, with NO volume division and NO sign reversal.
      virial[0] += cfg.stresses[0][0]; virial[1] += cfg.stresses[1][1];
      virial[2] += cfg.stresses[2][2]; virial[3] += cfg.stresses[0][1];
      virial[4] += cfg.stresses[0][2]; virial[5] += cfg.stresses[1][2];
    }
  } catch (const MlipException &e) {
    native_->model->AbortDistributedEFS();
    // This may be a rank-local failure while peers are in communication.
    // Error::all has a collective barrier and would deadlock in that case.
    error->one(FLERR, "Native SUS2 evaluation: {}", e.What());
  } catch (const std::exception &e) {
    native_->model->AbortDistributedEFS();
    error->one(FLERR, "{}", e.what());
  }
}

int PairSUS2Superlinear::pack_forward_comm(int n, int *indices, double *buf, int, int *)
{
  for (int k = 0; k < n; ++k)
    native_->model->PackDistributedForward(indices[k], buf + static_cast<size_t>(k) * comm_forward);
  return n * comm_forward;
}
void PairSUS2Superlinear::unpack_forward_comm(int n, int first, double *buf)
{
  for (int k = 0; k < n; ++k)
    native_->model->UnpackDistributedForward(first + k, buf + static_cast<size_t>(k) * comm_forward);
}
int PairSUS2Superlinear::pack_reverse_comm(int n, int first, double *buf)
{
  for (int k = 0; k < n; ++k)
    native_->model->PackDistributedReverse(first + k, buf + static_cast<size_t>(k) * comm_reverse);
  return n * comm_reverse;
}
void PairSUS2Superlinear::unpack_reverse_comm(int n, int *indices, double *buf)
{
  for (int k = 0; k < n; ++k)
    native_->model->AddDistributedReverse(indices[k], buf + static_cast<size_t>(k) * comm_reverse);
}
double PairSUS2Superlinear::memory_usage()
{
  // Lower bound: the native graph and coefficient caches own additional memory.
  const auto stats = native_->model ? native_->model->DistributedRadialTableStats() : RadialTableStats();
  return static_cast<double>(type_map_.capacity() * sizeof(int) + stats.bytes);
}
