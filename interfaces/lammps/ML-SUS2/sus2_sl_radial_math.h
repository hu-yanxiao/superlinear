/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Konstantin Gubaev
 */

#ifndef SUS2_SL_RADIAL_MATH_H
#define SUS2_SL_RADIAL_MATH_H

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <map>

namespace LAMMPS_NS { namespace SUS2SL { namespace radial_math {

enum class RadialTableMode { Direct, Table, Auto };

struct RadialTableStats {
	std::size_t tables = 0, bytes = 0, hits = 0;
	std::size_t direct_fallbacks = 0, domain_fallbacks = 0;
	std::size_t memory_fallbacks = 0, invalid_fallbacks = 0;
};

class AnyRadialBasis
{
private:
	struct TableKey {
		std::uint64_t slope, shift;
		int channel;
		bool operator<(const TableKey& other) const;
	};
	struct HermiteTable {
		bool memory_rejected = false;
		double lower, upper, step;
		std::size_t intervals;
		// Each knot stores [rb_size values, rb_size r-derivatives].
		std::vector<double> knots;
	};
	RadialTableMode inference_table_mode_ = RadialTableMode::Direct;
	bool inference_table_active_ = false, inference_table_building_ = false;
	double inference_table_spacing_ = 0.001;
	std::size_t inference_table_max_bytes_ = 64 * 1024 * 1024;
	std::map<TableKey, HermiteTable> inference_tables_;
	RadialTableStats inference_table_stats_;
	std::uint64_t table_state_min_ = 0, table_state_max_ = 0, table_state_scaling_ = 0;
	int table_state_size_ = -1, table_state_cutoff_p_ = -1;
	std::string table_state_cutoff_;
	void ClearInferenceRadialTables();
protected:
	// Inference only: parameter/mixed derivative blocks are unspecified on hits.
	bool TryInferenceRadialTable(double r, double scal, double s, int k);
public:
	int rb_size;

	double min_dist;
	double max_dist;
	double scaling = 1.0; // all functions are multiplied by scaling
	std::string radial_cutoff_type = "legacy";
	int radial_cutoff_p = 5;

	// values and derivatives, set by calc(r)
	std::vector<double> rb_vals;
	std::vector<double> rb_ders;

	virtual std::string GetRBTypeString()
	{
		return "RBAny";
	}

	void ReadRadialBasis(std::ifstream& ifs);
	void WriteRadialBasis(std::ofstream& ofs);
	void SetRadialCutoff(const std::string& cutoff_type, int cutoff_p);
	void CalcRadialCutoff(double r, double& cutoff_f, double& cutoff_der) const;
	// Both Table and Auto lazily try a table and explicitly fall back to direct
	// evaluation outside [max(0,min_dist),max_dist], on non-finite samples or on
	// the memory limit. min_dist is not a physical lower cutoff: smaller r stays
	// direct. The cap bounds knot storage, not allocator/map overhead.
	void ConfigureInferenceRadialTable(RadialTableMode mode,
		double spacing = 0.001, std::size_t max_bytes = 64 * 1024 * 1024);
	// A model must activate this only around inference, and restore it on exit.
	// Disabled by default; training and parameter derivatives remain direct.
	bool SetInferenceRadialTableActive(bool active);
	bool IsInferenceRadialTableActive() const { return inference_table_active_; }
	RadialTableStats GetInferenceRadialTableStats() const;

	AnyRadialBasis(double _min_dist, double _max_dist, int _size);;
	AnyRadialBasis(std::ifstream& ifs);
	virtual ~AnyRadialBasis() {};
	
	virtual void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) = 0;	// calculates values and derivatives
	virtual void RB_CalcValsOnly(double r,double scal=0.1, double s= 0.1,int k=0); // calculates values only when available
	// Only rb_vals and rb_ders[0:rb_size] are valid after this call. Parameter
	// and mixed derivatives are unspecified; their consumers must use RB_Calc.
	// Un-specialized bases safely fall back to the full evaluator.
	virtual void RB_CalcRDerOnly(double r,double scal=0.1, double s=0.1,int k=0);
};



class RadialBasis_Shapeev : public AnyRadialBasis
{
private:
	static const int ALLOCATED_DEGREE = 11;
	double rb_coeffs[ALLOCATED_DEGREE + 1][ALLOCATED_DEGREE + 3];
	void InitShapeevRB();
	
public:	
	std::string GetRBTypeString() override
	{
		return "RBShapeev";
	}

	RadialBasis_Shapeev(double _min_dist, double _max_dist, int _size);
	RadialBasis_Shapeev(std::ifstream& ifs);

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};



class RadialBasis_Chebyshev : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev";
	}

	RadialBasis_Chebyshev(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Chebyshev_s : public AnyRadialBasis
{
public:
        std::string GetRBTypeString() override
        {
                return "RBChebyshev_s";
        }

        RadialBasis_Chebyshev_s(double _min_dist, double _max_dist, int _size)
                : AnyRadialBasis(_min_dist, _max_dist, _size) {};
        RadialBasis_Chebyshev_s(std::ifstream& ifs)
                : AnyRadialBasis(ifs) {};

        void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Chebyshev_ss : public AnyRadialBasis
{
public:
        std::string GetRBTypeString() override
        {
                return "RBChebyshev_ss";
        }

        RadialBasis_Chebyshev_ss(double _min_dist, double _max_dist, int _size)
                : AnyRadialBasis(_min_dist, _max_dist, _size) {};
        RadialBasis_Chebyshev_ss(std::ifstream& ifs)
                : AnyRadialBasis(ifs) {};

        void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};



class RadialBasis_Chebyshev_ssw : public AnyRadialBasis
{

private:
	static const std::vector<double> arr;


public:
        std::string GetRBTypeString() override
        {
                return "RBChebyshev_ssw";
        }

        RadialBasis_Chebyshev_ssw(double _min_dist, double _max_dist, int _size)
                : AnyRadialBasis(_min_dist, _max_dist, _size) {};
        RadialBasis_Chebyshev_ssw(std::ifstream& ifs)
                : AnyRadialBasis(ifs) {};

        void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Chebyshev_sss : public AnyRadialBasis
{
public:
        std::string GetRBTypeString() override
        {
                return "RBChebyshev_sss";
        }

        RadialBasis_Chebyshev_sss(double _min_dist, double _max_dist, int _size)
                : AnyRadialBasis(_min_dist, _max_dist, _size) {};
        RadialBasis_Chebyshev_sss(std::ifstream& ifs)
                : AnyRadialBasis(ifs) {};

        void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
        void RB_CalcValsOnly(double r,double scal=0.1, double s= 0.1,int k=0) override;
        void RB_CalcRDerOnly(double r,double scal=0.1, double s=0.1,int k=0) override;
};

class RadialBasis_Chebyshev_sss_rational : public AnyRadialBasis
{
public:
        std::string GetRBTypeString() override
        {
                return "RBChebyshev_sss_rational";
        }

        RadialBasis_Chebyshev_sss_rational(double _min_dist, double _max_dist, int _size)
                : AnyRadialBasis(_min_dist, _max_dist, _size) {};
        RadialBasis_Chebyshev_sss_rational(std::ifstream& ifs)
                : AnyRadialBasis(ifs) {};

        void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
        void RB_CalcValsOnly(double r,double scal=0.1, double s= 0.1,int k=0) override;
        void RB_CalcRDerOnly(double r,double scal=0.1, double s=0.1,int k=0) override;
};


class RadialBasis_Chebyshev_sssw : public AnyRadialBasis
{

private:
        static const std::vector<double> arr;

public:
        std::string GetRBTypeString() override
        {
                return "RBChebyshev_sssw";
        }

        RadialBasis_Chebyshev_sssw(double _min_dist, double _max_dist, int _size)
                : AnyRadialBasis(_min_dist, _max_dist, _size) {};
        RadialBasis_Chebyshev_sssw(std::ifstream& ifs)
                : AnyRadialBasis(ifs) {};

        void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Chebyshev_sss_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_sss_lmp";
	}

	RadialBasis_Chebyshev_sss_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_sss_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Chebyshev_sss_rational_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_sss_rational_lmp";
	}

	RadialBasis_Chebyshev_sss_rational_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_sss_rational_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Chebyshev_sssw_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_sssw_lmp";
	}

	RadialBasis_Chebyshev_sssw_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_sssw_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};



class RadialBasis_Chebyshev_ss_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_ss_lmp";
	}

	RadialBasis_Chebyshev_ss_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_ss_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Chebyshev_ssw_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_ssw_lmp";
	}

	RadialBasis_Chebyshev_ssw_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_ssw_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};




class RadialBasis_Chebyshev_s_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_s_lmp";
	}

	RadialBasis_Chebyshev_s_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_s_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Chebyshev_ssss : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_ssss";
	}

	RadialBasis_Chebyshev_ssss(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_ssss(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Chebyshev_sssss : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_sssss";
	}

	RadialBasis_Chebyshev_sssss(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_sssss(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};



class RadialBasis_Chebyshev_sigma : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_sigma";
	}

	RadialBasis_Chebyshev_sigma(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_sigma(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Laguerre_log1p : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBLaguerre_log1p";
	}

	RadialBasis_Laguerre_log1p(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Laguerre_log1p(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Laguerre_log1p_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBLaguerre_log1p_lmp";
	}

	RadialBasis_Laguerre_log1p_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Laguerre_log1p_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Laguerre_log1p_pos : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBLaguerre_log1p_pos";
	}

	RadialBasis_Laguerre_log1p_pos(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Laguerre_log1p_pos(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Laguerre_log1p_pos_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBLaguerre_log1p_pos_lmp";
	}

	RadialBasis_Laguerre_log1p_pos_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Laguerre_log1p_pos_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Laguerre_log1p_noenv : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBLaguerre_log1p_noenv";
	}

	RadialBasis_Laguerre_log1p_noenv(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Laguerre_log1p_noenv(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Laguerre_log1p_noenv_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBLaguerre_log1p_noenv_lmp";
	}

	RadialBasis_Laguerre_log1p_noenv_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Laguerre_log1p_noenv_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Jacobi_sss : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBJacobi_sss";
	}

	RadialBasis_Jacobi_sss(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Jacobi_sss(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Jacobi_sss_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBJacobi_sss_lmp";
	}

	RadialBasis_Jacobi_sss_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Jacobi_sss_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Jacobi_sss_noweight : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBJacobi_sss_noweight";
	}

	RadialBasis_Jacobi_sss_noweight(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Jacobi_sss_noweight(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Jacobi_sss_noweight_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBJacobi_sss_noweight_lmp";
	}

	RadialBasis_Jacobi_sss_noweight_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Jacobi_sss_noweight_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};




class RadialBasis_Bessel : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBBessel";
	}

	RadialBasis_Bessel(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Bessel(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Besselw : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBBesselw";
	}

	RadialBasis_Besselw(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Besselw(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Chebyshev_Tri : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_Tri";
	}

	RadialBasis_Chebyshev_Tri(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_Tri(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};



class RadialBasis_Bessel_sss : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBBessel_sss";
	}

	RadialBasis_Bessel_sss(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Bessel_sss(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Bessel_sssw : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBBessel_sssw";
	}

	RadialBasis_Bessel_sssw(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Bessel_sssw(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Chebyshev_tanhexp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_tanhexp";
	}

	RadialBasis_Chebyshev_tanhexp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_tanhexp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Chebyshev_tanhexp_w : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_tanhexp_w";
	}

	RadialBasis_Chebyshev_tanhexp_w(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_tanhexp_w(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};



class RadialBasis_Chebyshev_repuls : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_repuls";
	}

	RadialBasis_Chebyshev_repuls(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_repuls(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};



class RadialBasis_Taylor : public AnyRadialBasis
{
public:	
	std::string GetRBTypeString() override
	{
		return "RBTaylor";
	}

	RadialBasis_Taylor(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Taylor(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

}}} // LAMMPS_NS::SUS2SL::radial_math
#endif // SUS2_SL_RADIAL_MATH_H
