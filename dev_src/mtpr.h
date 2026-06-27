/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "../src/basic_mlip.h"
#include "../src/common/multidimensional_arrays.h"
#include <map>
#include <random>
#ifndef MLIP_MTPR_H
#define MLIP_MTPR_H

struct BasicIndices {
        std::vector<int> comp0, comp1, comp2, comp3;
    };

struct SHProduct {
	int left;
	int right;
	int target;
	double coeff;
};

struct SHProductRowTerm {
	int left;
	int right;
	double coeff;
};

struct SHProductRow {
	int target;
	int term_begin;
	int term_count;
	int scalar_index;
	bool terminal_scalar;
};

struct SHScalarInfo {
	int body_order = 0;
	int q[5] = {-1, -1, -1, -1, -1};
	int intermediate_l = 0;
};



class MLMTPR : public AnyLocalMLIP
{
protected:
	void MemAlloc();

	void ReadLinearCoeffs(std::ifstream& ifs);				//Read linear regression coeffs from MTP file
	void WriteLinearCoeffs(std::ofstream& ofs);				//Write linear regression coeffs to MTP file


	int alpha_moments_count;						//	/=================================================================================================
	int alpha_index_basic_count;					//	|                                                                                                |
	int(*alpha_index_basic_comp0);
	int(*alpha_index_basic_comp1);
	int(*alpha_index_basic_comp2);
	int(*alpha_index_basic_comp3);
	int(*alpha_index_basic)[4];


    			                        //	|   Internal representation of Moment Tensor Potential basis                                     |
	int alpha_index_times_count;					//	|   These items is required for calculation of basis functions values and their derivatives      |
	int(*alpha_index_times)[4];						//	|                                                                                                |
	int(*alpha_index_times_comp0);
	int(*alpha_index_times_comp1);
	int(*alpha_index_times_comp2);
	int(*alpha_index_times_comp3);
	int *alpha_moment_mapping;						//	\=================================================================================================
	std::string pot_desc;
	std::string rbasis_type;
	std::string scaling_map;
	std::string potential_tag;
	bool is_sh_potential_ = false;
	int sh_l_max_ = 0;
	int sh_k_max_ = 1;
	int sh_body_order_ = 0;
	std::string sh_parity_;
	std::vector<int> sh_body_l_max_;
	std::vector<SHProduct> sh_products_;
	std::vector<SHProductRow> sh_product_rows_;
	std::vector<SHProductRowTerm> sh_product_row_terms_;
	std::vector<int> sh_scalar_index_by_moment_;
	std::vector<char> sh_scalar_terminal_product_;
	std::vector<char> two_layer_gate_required_moments_;
	std::vector<int> two_layer_gate_required_moment_indices_;
	std::vector<int> two_layer_gate_required_basic_indices_;
	std::vector<int> two_layer_gate_required_basic_dense_mu_indices_;
	std::vector<int> two_layer_gate_required_product_indices_;
	std::vector<int> two_layer_gate_required_mu_indices_;
	std::vector<int> two_layer_gate_mu_dense_index_;
	std::vector<int> two_layer_gate_required_radial_eval_blocks_;
	bool sh_product_rows_trace_printed_ = false;
	bool sh_site_der_cache_trace_printed_ = false;
	void EnsureSHScalarInfoForGateUpgrade();
	
	double *moment_vals; //!< Array of basis function values calculated for certain atomic neighborhood
	Array3D moment_ders;//!< Array of basis function derivatives w.r.t. motion of neighboring atoms calculated for certain atomic neighborhood
	double *basis_vals;	//!< Array of the basis functions values calculated for certain neighbor	hood of the certain configuration
	Array3D basis_ders;	//!< Array derivatives w.r.t. each atom in neiborhood of the basis functions values calculated for certain neighborhood of the certain configuration

	Array3D moment_jacobian_;
	std::vector<double> site_energy_ders_wrt_moments_;
	std::vector<double> dist_powers_;
	std::vector<double> coords_powers_x;
	std::vector<double> coords_powers_y;
	std::vector<double> coords_powers_z;
	int max_alpha_index_basic_ = 0;

	std::vector<Vector3> coords_powers_;
	std::vector<double> lmp_radial_vals_buffer_;
	std::vector<double> lmp_radial_ders_buffer_;
	std::vector<double> radial_vals_buffer_;
	std::vector<double> radial_ders_buffer_;
	std::vector<double> basis_radial_ders_buffer_;
	Array3D grad_moment_jacobian_;
	Array3D grad_mom_rad_jacobian_;
	Array3D grad_mom_rad_jacobian_s_;
	Array3D grad_mom_rad_jacobian_ss_;
	Array3D grad_mom_rad_coord_jacobian_;
	Array3D grad_mom_rad_coord_jacobian_s_;
	Array3D grad_mom_rad_coord_jacobian_ss_;
	std::vector<double> grad_mom_vals_;
	std::vector<double> grad_dloss_dsenders_;
	std::vector<double> grad_dloss_dmom_;
	std::vector<double> grad_dist_powers_;
	std::vector<double> grad_coords_powers_x_;
	std::vector<double> grad_coords_powers_y_;
	std::vector<double> grad_coords_powers_z_;
	std::vector<double> grad_mu_contract_vals_;
	std::vector<double> grad_mu_contract_ders_;
	std::vector<double> grad_mu_contract_ders_s_;
	std::vector<double> grad_mu_contract_ders_ss_;
	std::vector<double> grad_mu_contract_coord_ders_s_;
	std::vector<double> grad_mu_contract_coord_ders_ss_;
	std::vector<double> grad_neighbor_dist_powers_cache_;
	std::vector<double> grad_neighbor_coords_powers_x_cache_;
	std::vector<double> grad_neighbor_coords_powers_y_cache_;
	std::vector<double> grad_neighbor_coords_powers_z_cache_;
	std::vector<double> grad_neighbor_radial_vals_cache_;
	std::vector<double> grad_neighbor_radial_ders_cache_;
	std::vector<double> grad_neighbor_mu_contract_vals_cache_;
	std::vector<double> grad_neighbor_mu_contract_ders_cache_;
	std::vector<double> grad_neighbor_mu_contract_ders_s_cache_;
	std::vector<double> grad_neighbor_mu_contract_ders_ss_cache_;
	std::vector<double> grad_neighbor_mu_contract_coord_ders_s_cache_;
	std::vector<double> grad_neighbor_mu_contract_coord_ders_ss_cache_;
	std::vector<double> grad_neighbor_sh_values_cache_;
			std::vector<double> grad_neighbor_sh_ders_cache_;
			std::vector<double> grad_radial_coeff_value_accum_;
			std::vector<double> grad_radial_coeff_coord_accum_;
			std::vector<double> grad_radial_coeff_center_coord_accum_;
			std::vector<double> two_layer_gate_additive_mu_buffer_;
			std::vector<double> two_layer_gate_type_scale_mu_buffer_;
			std::vector<double> two_layer_gate_multiplier_mu_buffer_;
			std::vector<double> two_layer_gate_outer_mu_buffer_;
			std::vector<double> two_layer_gate_additive_param_mu_buffer_;
				std::vector<double> two_layer_gate_gate_residual_mu_buffer_;
				std::vector<double> two_layer_gate_gate_outer_param_mu_buffer_;
				std::vector<double> two_layer_gate_gate_additive_param_mu_buffer_;
				std::vector<double> two_layer_gate_neighbor_multiplier_mu_buffer_;
				std::vector<double> two_layer_gate_neighbor_additive_mu_buffer_;
				std::vector<double> two_layer_gate_neighbor_additive_param_mu_buffer_;
				std::vector<double> two_layer_gate_center_multiplier_mu_buffer_;
				std::vector<double> two_layer_gate_center_additive_mu_buffer_;
				std::vector<double> two_layer_gate_center_additive_param_mu_buffer_;
				std::vector<double> two_layer_gate_center_residual_mu_buffer_;
				std::vector<double> two_layer_gate_center_gate_additive_param_mu_buffer_;
				std::vector<double> two_layer_gate_scaled_mu_vals_buffer_;
				std::vector<double> two_layer_gate_scaled_mu_ders_buffer_;
			std::vector<double> two_layer_gate_tanh_mu_cache_;
			std::vector<char> two_layer_gate_tanh_mu_cache_valid_;
			int two_layer_gate_tanh_mu_cache_atom_count_ = 0;
			int two_layer_gate_tanh_mu_cache_radial_func_count_ = 0;
			std::vector<double> sh_gate_moment_ders_;
		std::vector<double> sh_gate_basis_ders_;
		std::vector<double> sh_gate_scalar_values_;
		std::vector<double> sh_gate_linear_adjoints_;
		std::vector<double> sh_gate_energy_ders_wrt_moments_;
		std::vector<Vector3> sh_gate_component_ders_;
	std::vector<int> basic_total_degree_cache_;
	std::vector<int> basic_scaling_block_cache_;
	std::vector<int> basic_radial_eval_block_cache_;
	std::vector<int> basic_radial_base_cache_;
	std::vector<int> basic_radial_deriv_base_cache_;
	std::vector<int> basic_radial_offset_cache_;
	std::vector<int> basic_mu_cache_;
	std::vector<int> basic_sh_index_cache_;
	std::vector<int> basic_sh_der_index_cache_;
	std::vector<int> basic_indices_by_mu_;
	std::vector<int> basic_mu_offsets_;
	std::vector<int> mu_to_radial_eval_block_;
	std::vector<int> radial_eval_to_scaling_block_;
	Array3D sh_adj_ders_;
	std::vector<double> sh_adj_vals_;

	void CalcSHBasisFuncs(const Neighborhood& nbh, double* bf_vals);
	void CalcSHMomentValuesOnly(const Neighborhood& nbh);
	void CalcSHMomentValuesWithSiteDerivativeCache(const Neighborhood& nbh);
	void CalcSHMomentValuesWithGradientCache(const Neighborhood& nbh);
	void CalcSHBasisFuncsDers(const Neighborhood& nbh);
	void CalcTwoLayerGateScalarValuesOnly(
		const Neighborhood& nbh,
		std::vector<double>& gate_scalar_values,
		int cache_atom_index = -1);
			void CalcTwoLayerGateEdgeL1ValuesOnly(
				const Neighborhood& nbh,
				std::vector<double>& edge_l1_values,
				int cache_atom_index = -1);
			void FillTwoLayerGateEdgeL1ValuesFromMoments(double* edge_l1_values) const;
			void ComputeTwoLayerGateEdgeL1CoordLoss(
				const Neighborhoods& neighborhoods,
				const std::vector<Vector3>& frc_weights,
			const Matrix3& str_weights,
			std::vector<double>& coord_loss);
		void ComputeTwoLayerGateEdgeL1PairCoordTangents(
			const Neighborhood& nbh,
			int neighbor_index,
			const double* edge_sh_values,
			const double* edge_sh_ders,
			const Vector3* edge_der_weight,
			std::vector<double>& source_tangents) const;
	void CalcTwoLayerGateScalarDers(
		const Neighborhood& nbh,
		std::vector<double>& gate_scalar_ders);
	void CalcTwoLayerGateWeightedScalarDers(
		const Neighborhood& nbh,
		std::vector<Vector3>& gate_scalar_ders,
		int cache_atom_index = -1,
		int body_order_filter = 0);
	void CalcTwoLayerGateWeightedScalarDersForBodyOrderAdjoints(
		const Neighborhood& nbh,
		std::vector<Vector3>& gate_scalar_ders,
		int cache_atom_index,
		const double* body_order_adjoints);
	void CalcTwoLayerGateWeightedScalarDersForScalarSeeds(
		const Neighborhood& nbh,
		std::vector<Vector3>& gate_scalar_ders,
		int cache_atom_index,
		const double* scalar_seeds);
	void AccumulateTwoLayerGateScalarParamGrad(
		const Neighborhood& nbh,
		std::vector<double>& out_grad_accumulator,
		double gate_adjoint,
		const Vector3* gate_der_weights,
		int cache_atom_index = -1,
		const double* gate_moment_tangents = nullptr,
		double gate_moment_tangent_scale = 1.0,
		int body_order_filter = 0);
	void AccumulateTwoLayerGateScalarParamGradForBodyOrderAdjoints(
		const Neighborhood& nbh,
		std::vector<double>& out_grad_accumulator,
		const double* body_order_gate_adjoints,
		const Vector3* gate_der_weights,
		int cache_atom_index,
		const double* gate_moment_tangents,
		const double* body_order_energy_adjoints);
	void AccumulateTwoLayerGateScalarParamGradForScalarSeeds(
		const Neighborhood& nbh,
		std::vector<double>& out_grad_accumulator,
		const double* gate_scalar_seeds,
		const Vector3* gate_der_weights,
		int cache_atom_index,
		const double* gate_moment_tangents,
		const double* energy_scalar_seeds);
	void CalcSHBasisGateDers(const Neighborhood& nbh,
	                         std::vector<double>& gate_basis_ders);
	void CalcTwoLayerGateScalarDirectionalDerivatives(
		const Neighborhood& nbh,
		const std::vector<Vector3>& direction_weights,
		std::vector<double>& gate_scalar_tangents,
		std::vector<double>* gate_moment_tangents = nullptr,
		int cache_atom_index = -1);
	void AccumulateSHBasisGateDers(
		const Neighborhood& nbh,
		std::vector<double>& gate_linear_adjoints);
	void CalcSHResidualSiteEnergyDers(const Neighborhood& nbh);
	void CalcSHSiteEnergyDers(const Neighborhood& nbh);
		void AccumulateSHGateTangentGrad(const Neighborhood& nbh,
											std::vector<double>& out_grad_accumulator,
											const std::vector<double>& neighbor_gate_tangent,
											const double* center_gate_tangent = nullptr,
											int cache_atom_index = -1);
	void AccumulateSHCombinationGrad(const Neighborhood& nbh,
										std::vector<double>& out_grad_accumulator,
										const double se_weight = 0.0,
										const Vector3* se_ders_weights = nullptr);
	void ReadSHProductGraph(std::ifstream& ifs, std::string& next_token);
	void WriteSHProductGraph(std::ofstream& ofs);
	void BuildSHProductProgram();
	void BuildTwoLayerGateProductProgram();
	void ClearTwoLayerEdgePrimitiveCache();
	void BuildTwoLayerEdgePrimitiveCache(const Neighborhoods& neighborhoods,
										bool need_derivatives,
										bool need_param_derivatives = true);
	bool HasTwoLayerEdgePrimitiveCache(int cache_atom_index,
										bool need_derivatives,
										bool need_param_derivatives = true) const;
	size_t TwoLayerEdgePrimitiveOffset(int cache_atom_index,
										int neighbor_index) const;
	int TwoLayerGateMuDenseIndex(int mu) const;
	bool UseSHProductRows() const;
	bool UseSHSiteDerivativeCache() const;
	bool UseSHAccumSkipSiteDers() const;
	bool UseSHProductHVTReverse() const;
	void TraceSHProductProgramOnce();
	void TraceSHSiteDerivativeCacheOnce(int neighbor_count,
										int sh_count,
										int radial_func_count);
	void ApplySHProductRowsForward();
	void ApplySHProductRowsDers(const Neighborhood& nbh);
	void AccumulateSHProductRowsForward(const std::vector<double>& input_values,
										std::vector<double>& output_values) const;
	void BackpropSHProductRows(std::vector<double>& adjoints) const;
	void CalcSHEdgeBasicValues(const Neighborhood& nbh,
	                           int neighbor_index,
	                           std::vector<double>& edge_basic_values);
	double CalcSHScalarFirstFactorProjection(
		const SHScalarInfo& scalar_info,
		const std::vector<double>& edge_basic_values) const;
	bool UseFastCalcEijProjection() const;
	void CalcSHDirectedEffectivePairWeights(
		const Neighborhood& nbh,
		std::vector<double>& edge_basic_weights) const;
	double CalcSHDirectedEffectivePairEnergyFromWeights(
		const Neighborhood& nbh,
		int neighbor_index,
		std::vector<double>& edge_basic_values,
		const std::vector<double>& edge_basic_weights);
	double CalcSHDirectedEffectivePairEnergy(
		const Neighborhood& nbh,
		int neighbor_index,
		std::vector<double>& edge_basic_values);

	std::vector<int> radial_eval_to_basis_k_;



	void CalcSiteEnergyDers(const Neighborhood& nbh) override;
	void PrepareEvalCaches() override;
	bool UsesJacobiIndexedBasis() const;
	int JacobiIndexedBlockForMu(int mu) const;

public:
	double scaling = 1.0; //!< how to scale moments
	std::vector<double> regression_coeffs;
	Array3D radial_coeffs;	
							//!< array of radial coefficients
	int L=0;
	int K_ = 1;


    BasicIndices alpha_index_basic_;
    BasicIndices alpha_index_times_;
    Array3D radial_list;
    Array3D radial_der_list;
	Array3D two_layer_gate_radial_list;
	Array3D two_layer_gate_radial_der_list;
    //const Array3D& get_radial_list() const { return radial_list; }
    //const Array3D& get_radial_der_list() const { return radial_der_list; }
    double inv_dr;

	std::vector<double> linear_coeffs;
	//std::map<int, int> mu_to_K;
	std::vector<int> mu_to_K;
	std::vector<int> mu_to_sigma;
	std::vector<double> linear_mults;					//!< array of multiplers for basis functions
	std::vector<double> max_linear;						//!< maximum absolute values of basis functions
	//double cn_cf;
	std::vector<double> max_radial;                         	//!< maximum values of r.b.f on the training set

	bool inited = false;
	bool has_shift_coeffs = false;
	bool has_scal_coeffs = false;
	bool has_radial_coeffs = false;
	bool has_linear_coeffs = false;
	bool has_sh_scalar_info_ = false;
	std::vector<SHScalarInfo> sh_scalar_info_;
		bool two_layer_gate_enabled_ = false;
		int two_layer_gate_body_order_max_ = 0;
		bool two_layer_gate_include_one_body_ = false;
			bool two_layer_gate_shared_radial_ = false;
			std::string two_layer_gate_mode_ = "mu-body-linear-combo";
			std::string two_layer_gate_site_mode_ = "neighbor";
				bool two_layer_residual_enabled_ = false;
			std::string two_layer_gate_scale_mode_ = "legacy";
			double two_layer_gate_bias_ = 1.0;
			double two_layer_gate_tanh_amplitude_ = 0.8;
		// Training-only residual stage hint: 0=full, 1=E0-only, 2=E1-only.
		int two_layer_residual_eval_stage_ = 0;
		bool two_layer_residual_skip_outer_param_grad_ = false;
		std::vector<int> two_layer_gate_scalar_indices_;
		std::vector<int> two_layer_gate_scalar_body_orders_;
		std::vector<int> two_layer_gate_body_order_weight_offsets_;
		std::vector<int> two_layer_gate_body_order_weight_indices_;
			std::vector<int> two_layer_gate_mu_body_orders_;
			std::vector<double> two_layer_gate_radial_coeffs_;
			std::vector<double> two_layer_gate_additive_coeffs_;
			std::vector<double> two_layer_gate_type_coeffs_;
			std::vector<double> two_layer_gate_weights_;
			std::vector<double> two_layer_gate_body_mix_weights_;
				bool two_layer_gate_edge_l1_enabled_ = false;
				std::vector<int> two_layer_gate_edge_l1_source_mu_indices_;
				std::vector<int> two_layer_gate_edge_l1_source_moment_indices_;
					std::vector<double> two_layer_gate_edge_l1_weights_;
					std::vector<double> two_layer_gate_edge_l1_values_;
				std::vector<double> two_layer_gate_edge_l1_weighted_values_;
				std::vector<double> two_layer_gate_edge_l1_adjoints_;
				std::vector<double> two_layer_gate_edge_l1_value_buffer_;
				std::vector<double> two_layer_gate_edge_l1_signal_buffer_;
				std::vector<double> two_layer_gate_edge_l1_chi_buffer_;
				std::vector<double> two_layer_gate_edge_l1_chi_adj_buffer_;
				std::vector<double> two_layer_gate_edge_l1_weights_by_source_;
				std::vector<double> two_layer_gate_edge_l1_coord_loss_;
				std::vector<double> two_layer_gate_edge_l1_energy_adjoint_buffer_;
				const std::vector<double>* active_two_layer_gate_edge_l1_coord_loss_ = nullptr;
				std::vector<double> two_layer_residual_e0_coeffs_;
			std::vector<double> two_layer_gate_values_;
			std::vector<double> two_layer_gate_scalar_values_cache_;
			std::vector<double> two_layer_gate_body_values_cache_;
			std::vector<double> two_layer_gate_moment_values_cache_;
			std::vector<double> two_layer_final_moment_values_cache_;
			std::vector<double> two_layer_final_moment_ders_cache_;
			std::vector<double> two_layer_forward_energy_gate_adjoints_cache_;
			std::vector<double> two_layer_gate_adjoints_;
			bool two_layer_gate_values_from_edge_cache_ready_ = false;
			bool two_layer_forward_final_moment_cache_ready_ = false;
			const std::vector<double>* active_two_layer_gate_values_ = nullptr;
			std::vector<double>* active_two_layer_gate_adjoints_ = nullptr;
			int two_layer_gate_center_mu_buffer_atom_index_ = -1;
			int two_layer_gate_center_mu_buffer_atom_type_ = -1;
			int two_layer_gate_center_mu_buffer_mask_ = 0;
			const double* two_layer_gate_center_mu_buffer_signal_ = nullptr;
			std::vector<size_t> two_layer_edge_offsets_cache_;
			std::vector<double> two_layer_edge_sh_values_cache_;
			std::vector<double> two_layer_edge_sh_ders_cache_;
			std::vector<double> two_layer_edge_radial_vals_cache_;
			std::vector<double> two_layer_edge_radial_ders_cache_;
	std::vector<double> two_layer_edge_mu_vals_cache_;
	std::vector<double> two_layer_edge_mu_ders_cache_;
	std::vector<double> two_layer_edge_mu_ders_s_cache_;
	std::vector<double> two_layer_edge_mu_ders_ss_cache_;
			std::vector<double> two_layer_edge_mu_coord_ders_s_cache_;
			std::vector<double> two_layer_edge_mu_coord_ders_ss_cache_;
			std::vector<double> two_layer_edge_gate_mu_vals_cache_;
			std::vector<double> two_layer_edge_gate_mu_ders_cache_;
			std::vector<double> two_layer_edge_gate_mu_ders_s_cache_;
			std::vector<double> two_layer_edge_gate_mu_ders_ss_cache_;
			std::vector<double> two_layer_edge_gate_mu_coord_ders_s_cache_;
			std::vector<double> two_layer_edge_gate_mu_coord_ders_ss_cache_;
			std::vector<double> two_layer_edge_gate_type_scale_cache_;
			std::vector<double> two_layer_edge_gate_additive_cache_;
			std::vector<double> two_layer_edge_gate_neighbor_multiplier_cache_;
		bool two_layer_edge_cache_ready_ = false;
		bool two_layer_edge_cache_has_derivatives_ = false;
		bool two_layer_edge_cache_has_param_derivatives_ = false;
		bool two_layer_full_edge_cache_for_next_calc_ = false;
		bool two_layer_reuse_full_edge_cache_once_ = false;
		int two_layer_edge_cache_atom_count_ = 0;
		int two_layer_edge_cache_eval_block_count_ = 0;
		int two_layer_edge_cache_sh_count_ = 0;
	int two_layer_edge_cache_radial_func_count_ = 0;
	int two_layer_edge_cache_gate_mu_count_ = 0;
	int two_layer_edge_cache_rb_size_ = 0;
	int active_two_layer_edge_cache_atom_index_ = -1;
        bool shift_ = true;
	double* energy_cmpnts;								// Energy components for SLAE matrix
	Array3D forces_cmpnts;								// Force components for SLAE matrix
	double(*stress_cmpnts)[3][3];						// Stress components for SLAE matrix

		void CalcBasisFuncs(const Neighborhood& Neighborhood, double* bf_vals); //Linear basic functions calculation
		double CalcSiteEnergyValue(const Neighborhood& nbh, double* out_site_energy0);
		void CalcBasisFuncsDers(const Neighborhood& Neighborhood);		//Linear basic functions and their derivatives calculation
	void CalcEFSComponents(Configuration& cfg);						//Calculate the components for linear regression matrix
	void CalcEFSComponents(Configuration& cfg, const Neighborhoods& neighborhoods);
	void CalcEFSComponents(Configuration& cfg, bool need_forces, bool need_stress);
	void CalcEFSComponents(Configuration& cfg, const Neighborhoods& neighborhoods, bool need_forces, bool need_stress);
	void CalcEComponents(Configuration& cfg);			//Calculate the components for linear regression matrix
	void CalcEComponents(Configuration& cfg, const Neighborhoods& neighborhoods);
		void CalcEFS(Configuration& cfg) override;
		void CalcEFS(Configuration& cfg, const Neighborhoods& neighborhoods) override;
		void CalcEnergyAndSiteEnergies(Configuration& cfg) override;
		void CalcEnergyAndSiteEnergies(Configuration& cfg, const Neighborhoods& neighborhoods) override;
		void AccumulateEFSCombinationGrad(Configuration& cfg,
	                                  std::vector<double>& ene_weight,
	                                  const std::vector<Vector3>& frc_weights,
	                                  const Matrix3& str_weights,
	                                  Array1D& out_grads_accumulator) override;
		void AccumulateEFSCombinationGrad(Configuration& cfg,
		                                  std::vector<double>& ene_weight,
		                                  const std::vector<Vector3>& frc_weights,
		                                  const Matrix3& str_weights,
		                                  Array1D& out_grads_accumulator,
		                                  const Neighborhoods& neighborhoods) override;
		void AccumulateEnergyCombinationGrad(Configuration& cfg,
		                                      std::vector<double>& ene_weight,
		                                      Array1D& out_grads_accumulator) override;
		void AccumulateEnergyCombinationGrad(Configuration& cfg,
		                                      std::vector<double>& ene_weight,
		                                      Array1D& out_grads_accumulator,
		                                      const Neighborhoods& neighborhoods) override;
	bool HasNonzeroTwoLayerGateWeights() const;
	bool HasNonzeroTwoLayerGateEdgeL1Weights() const;
	bool RequiresTwoLayerGateEvaluation() const;
	void AddTwoLayerGateX2Penalty(Configuration& cfg,
	                              const Neighborhoods& neighborhoods,
	                              double coeff,
	                              double& out_penalty_accumulator,
	                              Array1D* out_penalty_grad_accumulator);
	void PrepareTwoLayerGateValues(Configuration& cfg, const Neighborhoods& neighborhoods);
	void AccumulateTwoLayerGateForceChain(Configuration& cfg, const Neighborhoods& neighborhoods);
	void AccumulateTwoLayerGateEdgeL1ForceChain(Configuration& cfg, const Neighborhoods& neighborhoods);
	void AccumulateTwoLayerGateValueParamGradFromAdjoints(
		Configuration& cfg,
		const Neighborhoods& neighborhoods,
		Array1D& out_grad_accumulator);
	void AccumulateTwoLayerGateEdgeL1SourceParamGradFromAdjoints(
		Configuration& cfg,
		const Neighborhoods& neighborhoods,
		Array1D& out_grad_accumulator);
	void AccumulateTwoLayerGateEdgeL1WeightGradForPair(
		const Neighborhood& nbh,
		int neighbor_index,
		const double* edge_sh_values,
		const double* gate_adjoint_by_mu,
		std::vector<double>& out_grad_accumulator);
	void AccumulateTwoLayerGateEdgeL1WeightCoordGradForPair(
		const Neighborhood& nbh,
		int neighbor_index,
		const double* edge_sh_values,
		const double* edge_sh_ders,
		const double* energy_gate_adjoint_by_mu,
		const Vector3* edge_der_weight,
		std::vector<double>& out_grad_accumulator);
		void AddTwoLayerGateMuAdjoint(const Neighborhood& nbh, int neighbor_index, int mu, double adjoint);
		double TwoLayerGateNeighborSignal(const Neighborhood& nbh, int neighbor_index, int mu) const;
		const double* TwoLayerGateNeighborSignals(const Neighborhood& nbh, int neighbor_index) const;
		const double* PrepareTwoLayerGateEdgeL1NeighborSignals(
				const Neighborhood& nbh,
				int neighbor_index,
				const double* base_signal_by_mu,
				const double* edge_sh_values,
				int& neighbor_atom_index_for_tanh);
			void PrepareTwoLayerGateEdgeL1WeightedValues(
				int atom_count,
				size_t edge_use_count);
			void PrepareTwoLayerEdgeGateTypeScaleCache(
				const Neighborhoods& neighborhoods);
		void AddTwoLayerGateAtomMuAdjoint(int atom_index, int mu, double adjoint);
		const double* TwoLayerGateAtomSignals(int atom_index) const;
		int TwoLayerGateMuBodyOrder(int mu) const;
	int TwoLayerGateWeightBodyOrder(int weight_index) const;
	void BuildTwoLayerGateBodyOrderBuckets();
	void AccumulateTwoLayerGateBodyOrderScalarDers(
		const Neighborhood& nbh,
		int body_order,
		double adjoint,
		std::vector<Vector3>& gate_scalar_ders,
		int cache_atom_index = -1);
	void InvalidateTwoLayerGateTanhMuCache();
	enum TwoLayerGateMuBufferMask {
		kGateMuBufferAdditive = 1 << 0,
		kGateMuBufferTypeScale = 1 << 1,
		kGateMuBufferMultiplier = 1 << 2,
		kGateMuBufferOuter = 1 << 3,
		kGateMuBufferAdditiveParam = 1 << 4,
		kGateMuBufferGateResidualParam = 1 << 5,
		kGateMuBufferGateOuterParam = 1 << 6,
		kGateMuBufferGateAdditiveParam = 1 << 7,
		kGateMuBufferAll = (1 << 8) - 1
	};
		void PrepareTwoLayerGateNeighborMuBuffers(int type_outer,
		                                          double center_type_coeff,
		                                          double outer_type_coeff,
		                                          const double* gate_signal_by_mu,
		                                          int neighbor_atom_index = -1,
		                                          int buffer_mask = kGateMuBufferAll);
		void PrepareTwoLayerGateAtomMuBuffers(int atom_type,
		                                      const double* gate_signal_by_mu,
		                                      int atom_index,
		                                      int buffer_mask = kGateMuBufferAll);
		void PrepareTwoLayerGatePairMuBuffers(int type_center,
		                                      int type_outer,
		                                      double center_type_coeff,
		                                      double outer_type_coeff,
		                                      const double* center_gate_signal_by_mu,
		                                      int center_atom_index,
		                                      const double* neighbor_gate_signal_by_mu,
		                                      int neighbor_atom_index,
		                                      int buffer_mask = kGateMuBufferAll);
//	void CalcEFS(Configuration& cfg) override
//	{
//		AnyLocalMLIP::CalcEFS(cfg);
//		cfg.features["EFS_by"] = "MultiMTP" + to_string(alpha_count);
//	}

	void AccumulateCombinationGrad(	const Neighborhood& nbh,
									std::vector<double>& out_grad_accumulator,
									const double se_weight = 0.0,
									const Vector3* se_ders_weights = nullptr) override;
	MLMTPR():
		AnyLocalMLIP() {
	}
	MLMTPR(const std::string& mtpfnm) :
		AnyLocalMLIP() {
		Load(mtpfnm);
	}

	~MLMTPR();
        void CalcDescriptors(Configuration& cfg, std::ofstream &ofs);
        void CalcpartialE(Configuration& cfg, std::ofstream &ofs);
        void CalcEij(Configuration& cfg, std::ofstream &ofs);
	void ReadMTPBasis(std::ifstream& ifs);		// Read MTP basis from file	
	void WriteMTPBasis(std::ofstream& ofs);		// Write MTP basis for file

	void Load(const std::string& filename) override;
	void Save(const std::string& filename) override;
        void Save_2(const std::string& filename);
	void RadialCoeffsInit(std::ifstream& ifs_rad);
	void Perform_scaling();
	bool HasCompleteParameters() const;
	void PruneSpecies(const std::vector<int>& old_species_indices);
	int ScalingCoeffCount() const;
		int RadialCoeffOffset() const;
		int RadialCoeffBlockSize() const;
		int BaseNonlinearCoeffCount() const;
						bool TwoLayerGateUsesSharedRadial() const;
						bool TwoLayerGateUsesBodyLinearCombo() const;
						bool TwoLayerGateUsesFullScalarWeights() const;
						bool TwoLayerResidualEnabled() const;
						bool TwoLayerGateUsesDirectScale() const;
						bool TwoLayerGateUsesCenterGate() const;
						const std::string& TwoLayerGateSiteMode() const { return two_layer_gate_site_mode_; }
						void SetTwoLayerGateMode(const std::string& mode);
						double TwoLayerGateTanhAmplitude() const;
						void SetTwoLayerGateTanhAmplitude(double amplitude);
						void SetTwoLayerGateSiteMode(const std::string& mode);
					bool IsSHPotential() const { return is_sh_potential_; }
				bool TwoLayerGateEnabled() const { return two_layer_gate_enabled_; }
					void UpgradePlainSHToTwoLayerGate(int gate_body_order,
					                                  bool independent_gate_radial_coeffs,
					                                  const std::string& gate_site_mode = "neighbor",
					                                  const std::string& gate_mode = "mu-body-linear-combo");
					void UpgradeTwoLayerGateToEdgeL1();
			void RequestTwoLayerFullEdgeCacheForNextCalcEFS();
			int TwoLayerGateRadialCoeffCount() const;
		int TwoLayerGateRadialCoeffOffset() const;
		int TwoLayerGateAdditiveCoeffCount() const;
		int TwoLayerGateAdditiveCoeffOffset() const;
		int TwoLayerGateTypeCoeffCount() const;
		int TwoLayerGateTypeCoeffOffset() const;
		int TwoLayerGateBodyOrderCount() const;
		int TwoLayerGateScalarCount() const;
		int TwoLayerGateWeightCount() const;
			int TwoLayerGateWeightOffset() const;
			int TwoLayerGateBodyMixWeightCount() const;
			int TwoLayerGateBodyMixWeightOffset() const;
				bool TwoLayerGateEdgeL1Enabled() const;
				int TwoLayerGateEdgeL1SourceCount() const;
				int TwoLayerGateEdgeL1WeightCount() const;
				int TwoLayerGateEdgeL1WeightOffset() const;
				int TwoLayerGateEdgeL1SourceMomentIndex(int source_index) const;
				int TwoLayerResidualE0CoeffCount() const;
				int TwoLayerResidualE0CoeffOffset() const;
		int LinearCoeffOffset() const;
		int LinearCoeffCount() const;
		int LinearEquationCount() const;
		int TwoLayerGateRadialCoeffIndex(int mu, int xi) const;
		double TwoLayerGateRadialCoeff(int mu, int xi) const;
		int TwoLayerGateAdditiveCoeffIndex(int type_outer, int mu) const;
		const double* TwoLayerGateAdditiveCoeffRowData(int type_outer) const;
		double TwoLayerGateAdditiveCoeff(int type_outer, int mu) const;
		int TwoLayerGateTypeCoeffIndex(int type_outer) const;
		double TwoLayerGateTypeCoeff(int type_outer) const;
			int TwoLayerGateWeightIndex(int scalar_weight_index) const;
			int TwoLayerGateWeightIndex(int mu, int scalar_weight_index) const;
			int TwoLayerGateBodyMixWeightIndex(int mu, int body_order) const;
			int TwoLayerGateEdgeL1WeightIndex(int mu, int source_index) const;
			double TwoLayerGateScalarWeight(int scalar_weight_index) const;
			double TwoLayerGateWeight(int mu, int scalar_weight_index) const;
				double TwoLayerGateWeight(int weight_index) const;
				double TwoLayerGateBodyMixWeight(int mu, int body_order) const;
					double TwoLayerGateBodyMixWeight(int weight_index) const;
					double TwoLayerGateEdgeL1Weight(int mu, int source_index) const;
					const double* TwoLayerGateEdgeL1WeightData() const;
					int FindTwoLayerGateEdgeL1RawSourceMomentIndex(int mu) const;
					void BuildTwoLayerGateEdgeL1Sources();
					const double* TwoLayerGateScalarWeightData() const;
		void ComputeTwoLayerGateBodySignals(const double* scalar_values,
		                                    double* body_values) const;
		void ComputeTwoLayerGateMuSignals(const double* body_values,
		                                  double* mu_values) const;
		const double* TwoLayerGateBodyMixWeightMatrixData() const;
		const double* TwoLayerGateFullWeightMatrixData() const;
		void ComputeTwoLayerGateFullMuSignalsForAtoms(int atom_count);
		void AccumulateTwoLayerGateScalarSeedsFromMuAdjoints(
			const double* mu_adjoints,
			double* scalar_seeds,
			std::vector<double>& body_scratch) const;
		double TwoLayerResidualE0Coeff(int scalar_index) const;
		void InitializeTwoLayerGateRadialCoeffsFromBase();
		void InitializeTwoLayerGateAdditiveCoeffs(double value = 1.0 / 12.0);
		void InitializeTwoLayerGateTypeCoeffsFromBase();
		int EnforcePositiveRadialFirstCoeffs(double min_value = 1.0e-12);
	bool IsRadialFirstCoeff(int coeff_index) const;
	double RadialFirstCoeffRawToValue(double raw_value) const;
	double RadialFirstCoeffValueToRaw(double coeff_value) const;
	double RadialFirstCoeffDerivativeFromValue(double coeff_value) const;
	bool IsRedundantRadialSpeciesCoeff(int coeff_index) const;
	void BuildActiveCoeffIndices(std::vector<int>& active_coeff_indices, bool exclude_scal_coeffs = false) const;
	int ActiveCoeffCount(bool exclude_scal_coeffs = false) const;
	int ScalingSlopeOffset(int scaling_block, int type_central, int type_outer) const;
	int ScalingShiftOffset(int scaling_block, int type_central, int type_outer) const;
		double OrderedPairStrength(int type_central, int type_outer) const;
		double NormalizedOrderedPairStrength(int type_central, int type_outer) const;
		void SetScalingSlopeRange(double start, double end);
		void SetScalingShiftRange(double start, double end);
		void InitializeDefaultScalingCoeffs();
		void RandomizeScalingCoeffs(std::mt19937_64& generator, double strength_jitter);
		void RandomizeRadialCoeffs(std::mt19937_64& generator, double radial_scale);
		void RandomizeNonlinearCoeffs(std::mt19937_64& generator, double radial_scale, bool include_scaling, double scaling_strength_jitter);

	int CoeffCount() //!< number of coefficients
	{
		return (int)regression_coeffs.size();
	}
	double* Coeff() //!< coefficients themselves
	{
		return &regression_coeffs[0];
	}
	const std::vector<double>& LinCoeff()								//returns linear coefficients
	{
		int Rsize = LinearCoeffOffset();

			for (int i = Rsize; i < Rsize + LinearCoeffCount();i++)
				linear_coeffs[i-Rsize]=regression_coeffs[i];


		return linear_coeffs;
	}
		void DistributeCoeffs()									//Combine radial and linear coefficients in one array
		{
			const int base_nonlinear_size = BaseNonlinearCoeffCount();
					const int gate_radial_count = TwoLayerGateRadialCoeffCount();
						const int gate_additive_count = TwoLayerGateAdditiveCoeffCount();
						const int gate_type_count = TwoLayerGateTypeCoeffCount();
						const int gate_weight_count = TwoLayerGateWeightCount();
					const int gate_body_mix_weight_count = TwoLayerGateBodyMixWeightCount();
					const int gate_edge_l1_weight_count = TwoLayerGateEdgeL1WeightCount();
						const int e0_count = TwoLayerResidualE0CoeffCount();
						const int linear_size = LinearCoeffCount();
					std::vector<double> old_coeffs = regression_coeffs;
					if (static_cast<int>(old_coeffs.size()) < base_nonlinear_size)
						ERROR("DistributeCoeffs found an inconsistent nonlinear coefficient block");
							const int new_gate_additive_offset = base_nonlinear_size + gate_radial_count;
						const int new_gate_type_offset = new_gate_additive_offset + gate_additive_count;
						const int new_gate_offset = new_gate_type_offset + gate_type_count;
						const int new_gate_body_mix_offset = new_gate_offset + gate_weight_count;
						const int new_gate_edge_l1_offset =
							new_gate_body_mix_offset + gate_body_mix_weight_count;
						const int legacy_gate_additive_count =
							two_layer_gate_enabled_ ? species_count : 0;
						const int legacy_gate_offset =
							new_gate_additive_offset + legacy_gate_additive_count;
						const bool old_coeffs_have_release_layout =
							gate_additive_count > 0
							&& static_cast<int>(old_coeffs.size()) >=
								new_gate_body_mix_offset + gate_body_mix_weight_count + e0_count;
						const bool old_coeffs_have_legacy_additive_layout =
							gate_additive_count > 0
							&& static_cast<int>(old_coeffs.size()) >=
								legacy_gate_offset + gate_weight_count
								+ gate_body_mix_weight_count + e0_count;
						const bool old_coeffs_have_edge_l1_layout =
							old_coeffs_have_release_layout
							&& gate_edge_l1_weight_count > 0
							&& static_cast<int>(old_coeffs.size()) >=
								new_gate_edge_l1_offset + gate_edge_l1_weight_count + e0_count;
					if (gate_radial_count > 0
					    && static_cast<int>(old_coeffs.size()) >= base_nonlinear_size + gate_radial_count) {
						two_layer_gate_radial_coeffs_.assign(gate_radial_count, 0.0);
						for (int i = 0; i < gate_radial_count; ++i)
							two_layer_gate_radial_coeffs_[i] =
								old_coeffs[base_nonlinear_size + i];
						}
						if (gate_additive_count > 0) {
						if (static_cast<int>(two_layer_gate_additive_coeffs_.size())
						    == gate_additive_count) {
							// Model metadata is already in the current release layout.
						} else if (old_coeffs_have_release_layout) {
							two_layer_gate_additive_coeffs_.assign(gate_additive_count, 0.0);
							for (int i = 0; i < gate_additive_count; ++i)
								two_layer_gate_additive_coeffs_[i] =
									old_coeffs[new_gate_additive_offset + i];
						} else if (old_coeffs_have_legacy_additive_layout) {
							two_layer_gate_additive_coeffs_.assign(gate_additive_count, 0.0);
							for (int type = 0; type < species_count; ++type)
								for (int mu = 0; mu < radial_func_count; ++mu)
									two_layer_gate_additive_coeffs_[
										type * radial_func_count + mu] =
										old_coeffs[new_gate_additive_offset + type];
						} else if (static_cast<int>(two_layer_gate_additive_coeffs_.size())
						           != gate_additive_count) {
							InitializeTwoLayerGateAdditiveCoeffs();
						}
					}
					if (gate_type_count > 0) {
						if (static_cast<int>(two_layer_gate_type_coeffs_.size())
						    == gate_type_count) {
							// Model metadata is already available.
						} else if (old_coeffs_have_release_layout) {
							two_layer_gate_type_coeffs_.assign(gate_type_count, 0.0);
							for (int i = 0; i < gate_type_count; ++i)
								two_layer_gate_type_coeffs_[i] =
									old_coeffs[new_gate_type_offset + i];
						} else {
							InitializeTwoLayerGateTypeCoeffsFromBase();
						}
					}
					if (gate_weight_count > 0) {
						const int old_gate_offset = old_coeffs_have_release_layout
							? new_gate_offset
							: (old_coeffs_have_legacy_additive_layout
								? legacy_gate_offset
								: base_nonlinear_size + gate_radial_count);
						if (static_cast<int>(old_coeffs.size()) >= old_gate_offset + gate_weight_count) {
							two_layer_gate_weights_.assign(gate_weight_count, 0.0);
							for (int i = 0; i < gate_weight_count; ++i)
								two_layer_gate_weights_[i] = old_coeffs[old_gate_offset + i];
						}
					}
						if (gate_body_mix_weight_count > 0) {
						const int old_gate_body_mix_offset = old_coeffs_have_release_layout
							? new_gate_body_mix_offset
							: (old_coeffs_have_legacy_additive_layout
								? legacy_gate_offset + gate_weight_count
								: base_nonlinear_size + gate_radial_count + gate_weight_count);
						if (static_cast<int>(old_coeffs.size())
						    >= old_gate_body_mix_offset + gate_body_mix_weight_count) {
							two_layer_gate_body_mix_weights_.assign(
								gate_body_mix_weight_count, 0.0);
							for (int i = 0; i < gate_body_mix_weight_count; ++i)
								two_layer_gate_body_mix_weights_[i] =
									old_coeffs[old_gate_body_mix_offset + i];
						} else if (static_cast<int>(two_layer_gate_body_mix_weights_.size())
						           != gate_body_mix_weight_count) {
								two_layer_gate_body_mix_weights_.assign(
									gate_body_mix_weight_count, 1.0);
							}
						}
							if (gate_edge_l1_weight_count > 0) {
								if (static_cast<int>(two_layer_gate_edge_l1_weights_.size())
								    == gate_edge_l1_weight_count) {
									// Metadata is authoritative for edge-L1 weights when present.
								} else if (old_coeffs_have_edge_l1_layout) {
									two_layer_gate_edge_l1_weights_.assign(
										gate_edge_l1_weight_count, 0.0);
									for (int i = 0; i < gate_edge_l1_weight_count; ++i)
										two_layer_gate_edge_l1_weights_[i] =
											old_coeffs[new_gate_edge_l1_offset + i];
								} else {
									two_layer_gate_edge_l1_weights_.assign(
										gate_edge_l1_weight_count, 0.0);
								}
							}
						if (e0_count > 0) {
							const int old_e0_offset =
								old_coeffs_have_edge_l1_layout
									? new_gate_edge_l1_offset + gate_edge_l1_weight_count
									: (old_coeffs_have_release_layout
										? new_gate_body_mix_offset + gate_body_mix_weight_count
										: (old_coeffs_have_legacy_additive_layout
											? legacy_gate_offset + gate_weight_count
												+ gate_body_mix_weight_count
											: base_nonlinear_size + gate_radial_count
												+ gate_weight_count + gate_body_mix_weight_count));
							if (static_cast<int>(old_coeffs.size()) >= old_e0_offset + e0_count) {
							two_layer_residual_e0_coeffs_.assign(e0_count, 0.0);
							for (int i = 0; i < e0_count; ++i)
								two_layer_residual_e0_coeffs_[i] = old_coeffs[old_e0_offset + i];
						}
					}
						regression_coeffs.assign(
							base_nonlinear_size + gate_radial_count
							+ gate_additive_count + gate_type_count + gate_weight_count
							+ gate_body_mix_weight_count + gate_edge_l1_weight_count
							+ e0_count + linear_size, 0.0);
				for (int i = 0; i < base_nonlinear_size; ++i)
					regression_coeffs[i] = old_coeffs[i];
			if (gate_radial_count > 0) {
				if (static_cast<int>(two_layer_gate_radial_coeffs_.size()) != gate_radial_count)
					InitializeTwoLayerGateRadialCoeffsFromBase();
				for (int i = 0; i < gate_radial_count; ++i)
					regression_coeffs[base_nonlinear_size + i] =
						two_layer_gate_radial_coeffs_[i];
			}
			if (gate_additive_count > 0) {
				if (static_cast<int>(two_layer_gate_additive_coeffs_.size()) != gate_additive_count)
					InitializeTwoLayerGateAdditiveCoeffs();
				for (int i = 0; i < gate_additive_count; ++i)
					regression_coeffs[TwoLayerGateAdditiveCoeffOffset() + i] =
						two_layer_gate_additive_coeffs_[i];
			}
			if (gate_type_count > 0) {
				if (static_cast<int>(two_layer_gate_type_coeffs_.size()) != gate_type_count)
					InitializeTwoLayerGateTypeCoeffsFromBase();
				for (int i = 0; i < gate_type_count; ++i)
					regression_coeffs[TwoLayerGateTypeCoeffOffset() + i] =
						two_layer_gate_type_coeffs_[i];
			}
				if (gate_weight_count > 0) {
					if (static_cast<int>(two_layer_gate_weights_.size()) != gate_weight_count)
						ERROR("SUS2-SH two-layer gate metadata has inconsistent sizes");
					for (int i = 0; i < gate_weight_count; ++i)
						regression_coeffs[TwoLayerGateWeightOffset() + i] =
							two_layer_gate_weights_[i];
				}
					if (gate_body_mix_weight_count > 0) {
					if (static_cast<int>(two_layer_gate_body_mix_weights_.size())
					    != gate_body_mix_weight_count)
						two_layer_gate_body_mix_weights_.assign(
							gate_body_mix_weight_count, 1.0);
						for (int i = 0; i < gate_body_mix_weight_count; ++i)
							regression_coeffs[TwoLayerGateBodyMixWeightOffset() + i] =
								two_layer_gate_body_mix_weights_[i];
					}
					if (gate_edge_l1_weight_count > 0) {
						if (static_cast<int>(two_layer_gate_edge_l1_weights_.size())
						    != gate_edge_l1_weight_count)
							two_layer_gate_edge_l1_weights_.assign(
								gate_edge_l1_weight_count, 0.0);
						for (int i = 0; i < gate_edge_l1_weight_count; ++i)
							regression_coeffs[TwoLayerGateEdgeL1WeightOffset() + i] =
								two_layer_gate_edge_l1_weights_[i];
					}
					if (e0_count > 0) {
					if (static_cast<int>(two_layer_residual_e0_coeffs_.size()) != e0_count)
						two_layer_residual_e0_coeffs_.assign(e0_count, 0.0);
					for (int i = 0; i < e0_count; ++i)
						regression_coeffs[TwoLayerResidualE0CoeffOffset() + i] =
							two_layer_residual_e0_coeffs_[i];
				}
				int radial_size = LinearCoeffOffset();
			int max_comp = species_count - 1;				//maximum index of component

		if (linear_coeffs.size() == alpha_count)
		{
			for (int i = 0; i <= max_comp; i++)
				regression_coeffs[radial_size + i] = linear_coeffs[0];				//constants for component's site energy shift

			for (int i = 1; i < alpha_count; i++)
				regression_coeffs[radial_size + i + max_comp] = linear_coeffs[i];

		}
		else
			for (int i = 0; i < (int)linear_coeffs.size(); i++)
				regression_coeffs[radial_size + i] = linear_coeffs[i];

		linear_mults.resize(alpha_scalar_moments);
		max_linear.resize(alpha_scalar_moments);

		for (int i=0;i<alpha_scalar_moments;i++)
		{
			linear_mults[i]=1;
			max_linear[i]=1e-10;

		}


	}
	int Get_RB_size()
	{
		return p_RadialBasis->rb_size;
	}

	void AddPenaltyGrad(const double coeff, 
						double& out_penalty_accumulator, 
						Array1D* out_penalty_grad_accumulator) override;
	void AddRadialSmoothnessPenalty(const double coeff,
						const int grid_size,
						double& out_penalty_accumulator,
						Array1D* out_penalty_grad_accumulator) override;
	void AddFixedAtomicEnergyPenalty(const std::vector<double>& atomic_energies,
						const double coeff,
						double& out_penalty_accumulator,
						Array1D* out_penalty_grad_accumulator) override;
	void AddScalarWeightL2Penalty(const double head_coeff,
						const double gate_scalar_coeff,
						const double gate_mix_coeff,
						const double gate_full_coeff,
						double& out_penalty_accumulator,
						Array1D* out_penalty_grad_accumulator) override;


	int alpha_count;								//!< Basis functions count 
	int alpha_scalar_moments;						//!< = alpha_count-1 (MTP-basis except constant function)
		int radial_func_count;							//!< number of radial basis functions used
		int species_count;							//!< number of components present in the potential
		bool custom_scaling_slope_range = false;
		bool custom_scaling_shift_range = false;
		double scaling_slope_range_start = 0.0;
		double scaling_slope_range_end = 0.0;
		double scaling_shift_range_start = 0.0;
		double scaling_shift_range_end = 0.0;

		void Orthogonalize();						//!<Orthogonalize the basic functions
};



#endif
