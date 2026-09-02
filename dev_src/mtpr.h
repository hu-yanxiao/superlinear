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
	void ResizeLinearComponentStorage();

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
	bool outer_k_rank1_enabled_ = false;
	int outer_k_rank_ = 0;
	bool outer_k_rank1_direct_radial_ = false;
	int outer_k_rank1_basic_begin_ = 0;
	int outer_k_rank1_basic_count_ = 0;
	int outer_k_rank1_product_begin_ = 0;
	std::vector<double> outer_k_rank1_weights_;
	int two_layer_gate_k_rank_ = 0;
	int two_layer_gate_k_rank_basic_begin_ = 0;
	int two_layer_gate_k_rank_basic_count_ = 0;
	int two_layer_gate_k_rank_product_begin_ = 0;
	int two_layer_gate_k_rank_product_end_ = 0;
	std::vector<double> two_layer_gate_k_rank_weights_;
	std::vector<int> two_layer_gate_k_rank_raw_basic_indices_;
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
		std::vector<char> second_layer_gate_required_moments_;
		std::vector<int> second_layer_gate_required_moment_indices_;
		std::vector<int> second_layer_gate_required_basic_indices_;
		std::vector<int> second_layer_gate_required_basic_dense_mu_indices_;
		std::vector<int> second_layer_gate_required_product_indices_;
		std::vector<int> second_layer_gate_required_mu_indices_;
		std::vector<int> second_layer_gate_mu_dense_index_;
		std::vector<int> second_layer_gate_required_radial_eval_blocks_;
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
				std::vector<double> two_layer_gate_radial_mixing_outer_delta_buffer_;
				std::vector<double> two_layer_gate_radial_mixing_gate_rank_buffer_;
				std::vector<double> two_layer_gate_radial_mixing_outer_rank_buffer_;
				std::vector<double> sh_site_energy_radial_mixing_z_scratch_;
				std::vector<double> sh_site_energy_radial_mixing_dz_scratch_;
				std::vector<double> sh_site_energy_radial_mixing_kernel_scratch_;
				std::vector<double> sh_site_energy_radial_mixing_kernel_der_scratch_;
				std::vector<double> sh_site_energy_radial_mixing_outer_scratch_;
				std::vector<double> sh_site_energy_radial_mixing_outer_der_scratch_;
				std::vector<double> sh_site_energy_radial_mixing_kernel_to_raw_scratch_;
				std::vector<double> sh_site_energy_radial_mixing_sum_adj_y_scratch_;
				std::vector<double> sh_gate_tangent_radial_mixing_delta_scratch_;
				std::vector<double> sh_gate_tangent_radial_mixing_rank_scratch_;
				std::vector<double> sh_gate_tangent_radial_mixing_rank_t_cache_scratch_;
				std::vector<double> sh_gate_tangent_radial_mixing_value_adj_scratch_;
				std::vector<double> sh_gate_tangent_radial_mixing_direct_adj_scratch_;
				std::vector<double> sh_gate_tangent_radial_mixing_param_z_scratch_;
				std::vector<double> sh_gate_tangent_radial_mixing_param_z_t_scratch_;
				std::vector<double> sh_gate_tangent_radial_mixing_param_xk_scratch_;
				std::vector<double> sh_gate_tangent_radial_mixing_param_xk_t_scratch_;
				std::vector<double> sh_gate_tangent_radial_mixing_param_kernel_scratch_;
				std::vector<double> sh_gate_tangent_radial_mixing_param_kernel_s_scratch_;
				std::vector<double> sh_gate_tangent_radial_mixing_param_kernel_ss_scratch_;
				std::vector<double> sh_gate_tangent_radial_mixing_param_kernel_to_raw_scratch_;
				std::vector<double> sh_combination_radial_mixing_tangent_delta_scratch_;
				std::vector<double> sh_combination_radial_mixing_tangent_z_scratch_;
				std::vector<double> sh_combination_radial_mixing_tangent_z_t_scratch_;
				std::vector<double> sh_combination_radial_mixing_tangent_z_cache_scratch_;
				std::vector<double> sh_combination_radial_mixing_tangent_z_t_cache_scratch_;
				std::vector<double> sh_combination_radial_mixing_tangent_kernel_scratch_;
				std::vector<double> sh_combination_radial_mixing_tangent_kernel_t_scratch_;
				std::vector<double> sh_combination_radial_mixing_tangent_outer_scratch_;
				std::vector<double> sh_combination_radial_mixing_tangent_outer_t_scratch_;
				std::vector<double> sh_combination_radial_mixing_value_adj_scratch_;
				std::vector<double> sh_combination_radial_mixing_energy_adj_scratch_;
				std::vector<double> sh_combination_radial_mixing_param_z_scratch_;
				std::vector<double> sh_combination_radial_mixing_param_z_t_scratch_;
				std::vector<double> sh_combination_radial_mixing_param_xk_scratch_;
				std::vector<double> sh_combination_radial_mixing_param_xk_t_scratch_;
				std::vector<double> sh_combination_radial_mixing_param_kernel_scratch_;
				std::vector<double> sh_combination_radial_mixing_param_kernel_t_scratch_;
				std::vector<double> sh_combination_radial_mixing_param_kernel_s_scratch_;
				std::vector<double> sh_combination_radial_mixing_param_kernel_coord_s_scratch_;
				std::vector<double> sh_combination_radial_mixing_param_kernel_ss_scratch_;
				std::vector<double> sh_combination_radial_mixing_param_kernel_coord_ss_scratch_;
				std::vector<double> sh_combination_radial_mixing_param_kernel_to_raw_scratch_;
				std::vector<double> sh_combination_radial_mixing_param_kernel_t_to_raw_scratch_;
				bool two_layer_gate_radial_mixing_active_eval_ = false;
			std::vector<double> two_layer_gate_tanh_mu_cache_;
			std::vector<char> two_layer_gate_tanh_mu_cache_valid_;
			int two_layer_gate_tanh_mu_cache_atom_count_ = 0;
			int two_layer_gate_tanh_mu_cache_radial_func_count_ = 0;
			std::vector<double> sh_gate_moment_ders_;
		std::vector<double> sh_gate_basis_ders_;
		std::vector<double> sh_gate_scalar_values_;
		std::vector<double> sh_gate_linear_adjoints_;
		std::vector<double> sh_gate_scalar_ders_;
		std::vector<double> sh_gate_linear_weight_matrix_;
		std::vector<double> sh_gate_linear_mu_coord_ders_;
		std::vector<double> sh_gate_linear_component_ders_;
		std::vector<double> sh_gate_edge_linear_adjoints_;
		std::vector<double> sh_gate_edge_l1_y_linear_adjoints_;
		std::vector<double> sh_gate_edge_l1_projection_linear_adjoints_;
		std::vector<double> sh_gate_edge_l1_source_coord_ders_;
		std::vector<double> sh_gate_edge_l1_weighted_source_coord_ders_;
		std::vector<double> sh_gate_edge_l1_source_component_ders_;
		std::vector<double> sh_gate_energy_ders_wrt_moments_;
		std::vector<Vector3> sh_gate_component_ders_;
		std::vector<Vector3> two_layer_gate_readout_direction_scratch_;
		std::vector<double> two_layer_gate_readout_tangent_scratch_;
		std::vector<double> two_layer_gate_readout_moment_tangent_scratch_;
		std::vector<double> two_layer_gate_readout_value_seed_scratch_;
		std::vector<double> two_layer_gate_readout_energy_seed_scratch_;
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
	// For each synthetic contracted primitive (in graph order), cache the raw
	// primitive node for every k.  This keeps the learned contraction out of
	// the neighbor loop and makes all forward/reverse paths use one mapping.
	std::vector<int> outer_k_rank1_raw_basic_indices_;
	std::vector<int> mu_to_radial_eval_block_;
	std::vector<int> radial_eval_to_scaling_block_;
	Array3D sh_adj_ders_;
	std::vector<double> sh_adj_vals_;

	void CalcSHBasisFuncs(const Neighborhood& nbh, double* bf_vals);
	void CalcSHMomentValuesOnly(const Neighborhood& nbh);
	void CalcSHMomentValuesOnlyLegacy(const Neighborhood& nbh);
	void CalcSHMomentValuesOnlyRadialMixing(const Neighborhood& nbh);
	void CalcSHMomentValuesWithSiteDerivativeCache(const Neighborhood& nbh);
	void CalcSHMomentValuesWithGradientCache(const Neighborhood& nbh);
	void CalcSHMomentValuesWithGradientCacheLegacy(const Neighborhood& nbh);
	void CalcSHMomentValuesWithGradientCacheRadialMixing(const Neighborhood& nbh);
	void AccumulateTwoLayerGateRadialMixingForwardEdge(
		const double* gate_type_scale_by_mu,
		const double* radial_values,
		const double* radial_basis_values,
		const double* sh_values,
		double* outer_delta);
	void CalcSHBasisFuncsDers(const Neighborhood& nbh);
		void CalcTwoLayerGateScalarValuesOnly(
			const Neighborhood& nbh,
			std::vector<double>& gate_scalar_values,
			int cache_atom_index = -1);
		void CalcSecondLayerGateScalarValuesOnly(
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
		void CalcSecondLayerGateWeightedScalarDersForScalarSeeds(
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
			const double* energy_scalar_seeds,
			const std::vector<double>* second_gate_directional_values = nullptr,
			const double* raw_energy_moment_seeds = nullptr);
		void AccumulateSecondLayerGateScalarParamGradForScalarSeeds(
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
		int cache_atom_index = -1,
		const std::vector<double>* second_gate_directional_values = nullptr);
	void CalcSecondLayerGateScalarDirectionalDerivatives(
		const Neighborhood& nbh,
		const std::vector<Vector3>& direction_weights,
		std::vector<double>& gate_scalar_tangents,
		std::vector<double>* gate_moment_tangents = nullptr);
	void AccumulateSHBasisGateDers(
		const Neighborhood& nbh,
		int center_atom_index,
		bool need_forces,
		bool need_stress,
		std::vector<double>& gate_linear_adjoints,
		std::vector<double>& edge_l1_projection_linear_adjoints);
	void ApplySHGateLinearComponentChain(
		Configuration& cfg,
		const Neighborhoods& neighborhoods,
		bool need_forces,
		bool need_stress);
	void CalcSHResidualSiteEnergyDers(const Neighborhood& nbh);
	void CalcSHSiteEnergyDers(const Neighborhood& nbh);
	void CalcSHSiteEnergyDersLegacy(const Neighborhood& nbh);
	void CalcSHSiteEnergyDersRadialMixing(const Neighborhood& nbh);
	void AddTwoLayerGateReadoutSiteEnergyDers(const Neighborhood& nbh);
	void AccumulateTwoLayerGateReadoutCombinationGrad(
		const Neighborhood& nbh,
		std::vector<double>& out_grad_accumulator,
		double se_weight,
		const Vector3* se_ders_weights);
	void AccumulateTwoLayerGateReadoutDirectionalGrad(
		const Neighborhood& nbh,
		std::vector<double>& out_grad_accumulator,
		const Vector3* se_ders_weights,
		const double* gate_scalar_tangents,
		const double* gate_moment_tangents,
		int cache_atom_index);
		void AccumulateSHGateTangentGrad(const Neighborhood& nbh,
											std::vector<double>& out_grad_accumulator,
											const std::vector<double>& neighbor_gate_tangent,
											const double* center_gate_tangent = nullptr,
											int cache_atom_index = -1);
		template <bool RadialMixingEnabled>
		void AccumulateSHGateTangentGradImpl(
			const Neighborhood& nbh,
			std::vector<double>& out_grad_accumulator,
			const std::vector<double>& neighbor_gate_tangent,
			const double* center_gate_tangent,
			int cache_atom_index);
	void AccumulateSHCombinationGrad(const Neighborhood& nbh,
										std::vector<double>& out_grad_accumulator,
										const double se_weight = 0.0,
										const Vector3* se_ders_weights = nullptr);
	template <bool RadialMixingEnabled>
	void AccumulateSHCombinationGradImpl(
		const Neighborhood& nbh,
		std::vector<double>& out_grad_accumulator,
		double se_weight,
		const Vector3* se_ders_weights);
	void ReadSHProductGraph(std::ifstream& ifs, std::string& next_token);
		void WriteSHProductGraph(std::ofstream& ofs);
		void ValidateSHProductGraphDefinitions() const;
		void BuildSHProductProgram();
		void BuildTwoLayerGateProductProgram();
		void BuildSecondLayerGateProductProgram();
		void BuildGateLayerProductProgram(
			const std::vector<int>& scalar_indices,
			const std::vector<int>* explicit_scalar_moment_indices,
			std::vector<char>& required_moments,
			std::vector<int>& required_moment_indices,
			std::vector<int>& required_basic_indices,
			std::vector<int>& required_basic_dense_mu_indices,
			std::vector<int>& required_product_indices,
			std::vector<int>& required_mu_indices,
			std::vector<int>& mu_dense_index,
			std::vector<int>& required_radial_eval_blocks) const;
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
	int OuterKRank1ProductStart() const;
	int OuterKRank1OuterBasicIndex(int channel, int l, int m) const;
	int OuterKRank1RawBasicIndex(int l, int m, int k) const;
	void ContractOuterKRank1Values(double* values) const;
	void ContractOuterKRank1MomentDers(const Neighborhood& nbh);
	void BackpropOuterKRank1Adjoints(std::vector<double>& adjoints) const;
	void AbsorbOuterKRank1WeightsIntoRadialCoeffs();
	int TwoLayerGateKRankBasicIndex(int channel, int l, int m) const;
	int TwoLayerGateKRankRawBasicIndex(int l, int m, int k) const;
	void ContractTwoLayerGateKRankValues(double* values) const;
	void BackpropTwoLayerGateKRankAdjoints(std::vector<double>& adjoints) const;
	void AccumulateTwoLayerGateKRankWeightGrad(
		std::vector<double>& out_grad,
		const std::vector<double>& total_adjoint,
		const std::vector<double>* energy_adjoint,
		const std::vector<double>* raw_tangent) const;
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
	void CalcTwoLayerGateEdgeBasicValues(
		const Neighborhood& nbh,
		int neighbor_index,
		int cache_atom_index,
		std::vector<double>& edge_basic_values);
	double CalcSHScalarFirstFactorProjection(
		const SHScalarInfo& scalar_info,
		const std::vector<double>& edge_basic_values) const;
	bool UseFastCalcEijProjection() const;
	void CalcSHDirectedEffectivePairWeights(
		const Neighborhood& nbh,
		std::vector<double>& edge_basic_weights,
		const std::vector<int>* scalar_indices = nullptr,
		const std::vector<double>* scalar_coeffs = nullptr) const;
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
	int RadialBasisKForMu(int mu) const;
	void CalcEnergyAndSiteEnergiesImpl(
		Configuration& cfg,
		const Neighborhoods& neighborhoods,
		std::vector<double>* out_energy_by_body_order);

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
	    Array3D second_layer_gate_radial_list;
	    Array3D second_layer_gate_radial_der_list;
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
				bool two_layer_gate_decoupled_scaling_ = false;
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
		std::vector<int> two_layer_gate_scalar_moment_indices_;
		std::vector<int> two_layer_gate_scalar_body_orders_;
		std::vector<int> two_layer_gate_body_order_weight_offsets_;
		std::vector<int> two_layer_gate_body_order_weight_indices_;
				std::vector<int> two_layer_gate_mu_body_orders_;
				std::vector<double> two_layer_gate_radial_coeffs_;
				std::vector<double> two_layer_gate_scaling_coeffs_;
				std::vector<double> two_layer_gate_additive_coeffs_;
					std::vector<double> two_layer_gate_type_coeffs_;
					std::vector<double> two_layer_gate_weights_;
					std::vector<double> two_layer_gate_body_mix_weights_;
					std::string two_layer_gate_radial_mixing_ = "none";
					std::vector<double> two_layer_gate_radial_mixing_coeffs_;
					bool two_layer_gate_readout_enabled_ = false;
				std::vector<double> two_layer_gate_readout_coeffs_;
				bool multi_layer_gate_enabled_ = false;
				bool second_layer_gate_shared_radial_ = false;
				int second_layer_gate_body_order_max_ = 0;
				std::string second_layer_gate_mode_ = "mu-body-linear-combo";
				std::vector<int> second_layer_gate_scalar_indices_;
				std::vector<int> second_layer_gate_scalar_body_orders_;
				std::vector<int> second_layer_gate_body_order_weight_offsets_;
				std::vector<int> second_layer_gate_body_order_weight_indices_;
				std::vector<int> second_layer_gate_mu_body_orders_;
				std::vector<double> second_layer_gate_radial_coeffs_;
				std::vector<double> second_layer_gate_additive_coeffs_;
				std::vector<double> second_layer_gate_type_coeffs_;
				std::vector<double> second_layer_gate_weights_;
				std::vector<double> second_layer_gate_body_mix_weights_;
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
					std::vector<double> second_layer_gate_values_;
					std::vector<double> second_layer_gate_scalar_values_cache_;
					std::vector<double> second_layer_gate_body_values_cache_;
					std::vector<double> second_layer_gate_moment_values_cache_;
					std::vector<double> second_layer_gate_adjoints_;
					std::vector<double> two_layer_full_scalar_seed_scratch_;
					std::vector<double> two_layer_full_scalar_energy_seed_scratch_;
					std::vector<double> two_layer_body_combo_seed_scratch_;
					std::vector<double> two_layer_body_combo_energy_seed_scratch_;
					std::vector<double> second_layer_full_scalar_seed_scratch_;
					std::vector<double> second_layer_body_combo_seed_scratch_;
					std::vector<double> second_layer_body_adjoint_scratch_;
					bool two_layer_gate_values_from_edge_cache_ready_ = false;
				bool two_layer_forward_final_moment_cache_ready_ = false;
				const std::vector<double>* active_two_layer_gate_values_ = nullptr;
				std::vector<double>* active_two_layer_gate_adjoints_ = nullptr;
				const std::vector<double>* active_second_layer_gate_values_ = nullptr;
				std::vector<double>* active_second_layer_gate_adjoints_ = nullptr;
				bool second_layer_gate_scope_ = false;
			int two_layer_gate_center_mu_buffer_atom_index_ = -1;
			int two_layer_gate_center_mu_buffer_atom_type_ = -1;
			int two_layer_gate_center_mu_buffer_mask_ = 0;
			const double* two_layer_gate_center_mu_buffer_signal_ = nullptr;
			std::vector<size_t> two_layer_edge_offsets_cache_;
			std::vector<double> two_layer_edge_sh_values_cache_;
			std::vector<double> two_layer_edge_sh_ders_cache_;
			std::vector<double> two_layer_edge_radial_vals_cache_;
			std::vector<double> two_layer_edge_radial_ders_cache_;
			std::vector<double> two_layer_edge_gate_radial_vals_cache_;
			std::vector<double> two_layer_edge_gate_radial_dr_cache_;
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
				double* energy_cmpnts = nullptr;					// Energy components for SLAE matrix
				Array3D forces_cmpnts;								// Force components for SLAE matrix
				double(*stress_cmpnts)[3][3] = nullptr;			// Stress components for SLAE matrix

		void CalcBasisFuncs(const Neighborhood& Neighborhood, double* bf_vals); //Linear basic functions calculation
		double CalcSiteEnergyValue(const Neighborhood& nbh, double* out_site_energy0);
		void CalcBasisFuncsDers(const Neighborhood& Neighborhood);		//Linear basic functions and their derivatives calculation
	void CalcEFSComponents(Configuration& cfg);						//Calculate the components for linear regression matrix
	void CalcEFSComponents(Configuration& cfg, const Neighborhoods& neighborhoods);
	void CalcEFSComponents(Configuration& cfg, bool need_forces, bool need_stress);
	void CalcEFSComponents(Configuration& cfg, const Neighborhoods& neighborhoods, bool need_forces, bool need_stress);
	void CalcEFSComponents(Configuration& cfg,
	                       const Neighborhoods& neighborhoods,
	                       bool need_forces,
	                       bool need_stress,
	                       bool fixed_center_readout_columns);
	void CalcEComponents(Configuration& cfg);			//Calculate the components for linear regression matrix
	void CalcEComponents(Configuration& cfg, const Neighborhoods& neighborhoods);
	void CalcEComponents(Configuration& cfg,
	                     const Neighborhoods& neighborhoods,
	                     bool fixed_center_readout_columns);
		void CalcEFS(Configuration& cfg) override;
		void CalcEFS(Configuration& cfg, const Neighborhoods& neighborhoods) override;
		void CalcEnergyAndSiteEnergies(Configuration& cfg) override;
		void CalcEnergyAndSiteEnergies(Configuration& cfg, const Neighborhoods& neighborhoods) override;
		void CalcEnergyAndSiteEnergiesWithSHBodyOrderContributions(
			Configuration& cfg,
			const Neighborhoods& neighborhoods,
			std::vector<double>& out_energy_by_body_order);
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
	int SHBodyOrderMax() const;
	bool HasSHBodyOrderBlocks() const;
	void CalcSHBodyOrderEnergyContributions(
		Configuration& cfg,
		const Neighborhoods& neighborhoods,
		std::vector<double>& out_energy_by_body_order);
	void AccumulateSHBodyOrderEnergyGrad(
		Configuration& cfg,
		const Neighborhoods& neighborhoods,
		const std::vector<double>& body_order_weights,
		Array1D& out_grads_accumulator);
	bool HasNonzeroTwoLayerGateWeights() const;
	bool HasNonzeroTwoLayerGateEdgeL1Weights() const;
	bool RequiresTwoLayerGateEvaluation() const;
		void AddTwoLayerGateX2Penalty(Configuration& cfg,
		                              const Neighborhoods& neighborhoods,
		                              double coeff,
		                              double& out_penalty_accumulator,
		                              Array1D* out_penalty_grad_accumulator);
		void PrepareTwoLayerGateValues(Configuration& cfg, const Neighborhoods& neighborhoods);
		void PrepareSecondLayerGateValues(Configuration& cfg, const Neighborhoods& neighborhoods);
		void AccumulateTwoLayerGateForceChain(Configuration& cfg, const Neighborhoods& neighborhoods);
		void AccumulateSecondLayerGateForceChain(Configuration& cfg, const Neighborhoods& neighborhoods);
		void AccumulateTwoLayerGateEdgeL1ForceChain(Configuration& cfg, const Neighborhoods& neighborhoods);
		void AccumulateTwoLayerGateValueParamGradFromAdjoints(
			Configuration& cfg,
			const Neighborhoods& neighborhoods,
			Array1D& out_grad_accumulator);
		void AccumulateSecondLayerGateValueParamGradFromAdjoints(
			Configuration& cfg,
			const Neighborhoods& neighborhoods,
			Array1D& out_grad_accumulator,
			const std::vector<Vector3>* force_weights = nullptr,
			const Matrix3* stress_weights = nullptr,
			const std::vector<double>* energy_second_gate_adjoints = nullptr);
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
	void AccumulateTwoLayerGateEdgeL1SourceAdjointsForPair(
		const Neighborhood& nbh,
		int neighbor_index,
		const double* edge_sh_values,
		const double* gate_adjoint_by_mu,
		std::vector<double>& out_source_adjoints);
	void AccumulateTwoLayerGateEdgeL1SourceCoordAdjointsForPair(
		const Neighborhood& nbh,
		int neighbor_index,
		const double* edge_sh_ders,
		const double* energy_gate_adjoint_by_mu,
		const Vector3* edge_der_weight,
		std::vector<double>& out_source_adjoints);
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
			void AddSecondLayerGateMuAdjoint(const Neighborhood& nbh, int neighbor_index, int mu, double adjoint);
			double SecondLayerGateNeighborSignal(const Neighborhood& nbh, int neighbor_index, int mu) const;
			const double* SecondLayerGateNeighborSignals(const Neighborhood& nbh, int neighbor_index) const;
			double* SecondLayerGateNeighborAdjoints(const Neighborhood& nbh, int neighbor_index);
			const double* SecondLayerGateAtomSignals(int atom_index) const;
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
		                                          int type_center,
		                                          const double* gate_signal_by_mu,
		                                          int neighbor_atom_index = -1,
		                                          int buffer_mask = kGateMuBufferAll);
		void PrepareTwoLayerGateAtomMuBuffers(int atom_type,
		                                      const double* gate_signal_by_mu,
		                                      int atom_index,
		                                      int buffer_mask = kGateMuBufferAll);
		void PrepareTwoLayerGatePairMuBuffers(int type_center,
		                                      int type_outer,
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
		int OuterTypeCoeffIndex(int type_outer, int mu) const;
		double OuterTypeCoeff(int type_outer, int mu) const;
		int BaseNonlinearCoeffCount() const;
							bool TwoLayerGateUsesSharedRadial() const;
							bool TwoLayerGateUsesDecoupledScaling() const;
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
					const std::string& ScalingMap() const { return scaling_map; }
						bool OuterKRank1Enabled() const { return outer_k_rank1_enabled_; }
						bool OuterKRankEnabled() const { return outer_k_rank_ > 0; }
						int OuterKRank() const { return outer_k_rank_; }
						bool OuterKRankUsesDirectRadial() const {
							return outer_k_rank_ == 1 && outer_k_rank1_direct_radial_;
						}
						bool OuterKRankHasTrainableWeights() const {
							return OuterKRankEnabled() && !OuterKRankUsesDirectRadial();
						}
					int OuterKRank1BasicBegin() const { return outer_k_rank1_basic_begin_; }
					int OuterKRank1BasicCount() const { return outer_k_rank1_basic_count_; }
					int OuterKRank1ProductBegin() const { return outer_k_rank1_product_begin_; }
					int OuterKRank1WeightCount() const;
					int OuterKRank1WeightOffset() const;
					int OuterKRank1WeightIndex(int channel, int l, int k) const;
					double OuterKRank1Weight(int channel, int l, int k) const;
					const double* OuterKRank1WeightData() const;
					bool TwoLayerGateKRankEnabled() const {
						return two_layer_gate_k_rank_ > 0;
					}
					int TwoLayerGateKRank() const { return two_layer_gate_k_rank_; }
					int TwoLayerGateKRankBasicBegin() const {
						return two_layer_gate_k_rank_basic_begin_;
					}
					int TwoLayerGateKRankBasicCount() const {
						return two_layer_gate_k_rank_basic_count_;
					}
					int TwoLayerGateKRankWeightCount() const;
					int TwoLayerGateKRankWeightOffset() const;
					int TwoLayerGateKRankWeightIndex(int channel, int l, int k) const;
					double TwoLayerGateKRankWeight(int channel, int l, int k) const;
					const double* TwoLayerGateKRankWeightData() const;
					bool TwoLayerGateEnabled() const { return two_layer_gate_enabled_; }
						void UpgradePlainSHToTwoLayerGate(int gate_body_order,
						                                      bool independent_gate_radial_coeffs,
						                                      const std::string& gate_site_mode = "neighbor",
						                                      const std::string& gate_mode = "mu-body-linear-combo",
						                                      bool decoupled_gate_scaling = false,
						                                      const std::string& gate_factor_pruning = "q-total");
						bool TwoLayerGateMatchesFactorPruning(
								const std::string& factor_pruning) const;
								void EnableTwoLayerGateDecoupledScalingFromBase();
								void EnableTwoLayerGateRadialMixingFull();
								void UpgradeTwoLayerGateToEdgeL1();
							void EnableTwoLayerGateReadout();
							bool TwoLayerGateReadoutEnabled() const {
								return two_layer_gate_readout_enabled_;
							}
						bool MultiLayerGateEnabled() const { return multi_layer_gate_enabled_; }
						void UpgradeTwoLayerGateToMultiLayerGate(
							int second_gate_body_order,
							const std::string& second_gate_mode = "mu-body-linear-combo",
							const std::string& second_gate_factor_pruning = "gate1");
						bool SecondLayerGateMatchesFactorPruning(
							const std::string& factor_pruning) const;
				void RequestTwoLayerFullEdgeCacheForNextCalcEFS();
			int TwoLayerGateRadialCoeffCount() const;
		int TwoLayerGateRadialCoeffOffset() const;
		int TwoLayerGateScalingCoeffCount() const;
		int TwoLayerGateScalingCoeffOffset() const;
		int TwoLayerGateAdditiveCoeffCount() const;
		int TwoLayerGateAdditiveCoeffOffset() const;
		int TwoLayerGateTypeCoeffCount() const;
		int TwoLayerGateTypeCoeffOffset() const;
		int TwoLayerGateBodyOrderCount() const;
		int TwoLayerGateScalarCount() const;
		int TwoLayerGateScalarMomentIndex(int scalar_weight_index) const;
		int TwoLayerGateScalarMomentIndexUnchecked(int scalar_weight_index) const
		{
			return two_layer_gate_scalar_moment_indices_[scalar_weight_index];
		}
		int TwoLayerGateScalarBodyOrder(int scalar_weight_index) const;
			int TwoLayerGateWeightCount() const;
			int TwoLayerGateWeightOffset() const;
			int TwoLayerGateBodyMixWeightCount() const;
			int TwoLayerGateBodyMixWeightOffset() const;
			bool TwoLayerGateRadialMixingEnabled() const;
			bool HasNonzeroTwoLayerGateRadialMixingCoeffs() const;
			int TwoLayerGateRadialMixingCoeffCount() const;
			int TwoLayerGateRadialMixingCoeffOffset() const;
			int TwoLayerGateRadialMixingCoeffIndex(int l, int outer_channel, int gate_channel, int xi) const;
			double TwoLayerGateRadialMixingCoeff(int l, int outer_channel, int gate_channel, int xi) const;
				bool TwoLayerGateEdgeL1Enabled() const;
				int TwoLayerGateEdgeL1SourceCount() const;
				int TwoLayerGateEdgeL1WeightCount() const;
				int TwoLayerGateEdgeL1WeightOffset() const;
				int TwoLayerGateEdgeL1SourceMomentIndex(int source_index) const;
					int TwoLayerResidualE0CoeffCount() const;
					int TwoLayerResidualE0CoeffOffset() const;
					int TwoLayerGateReadoutCoeffCount() const;
					int TwoLayerGateReadoutCoeffOffset() const;
					int TwoLayerGateReadoutLinearColumnOffset() const;
					double TwoLayerGateReadoutCoeff(int scalar_index) const;
		int LinearCoeffOffset() const;
		int LinearCoeffCount() const;
		int LinearEquationCount() const;
		int TwoLayerGateRadialCoeffIndex(int mu, int xi) const;
		double TwoLayerGateRadialCoeff(int mu, int xi) const;
		int TwoLayerGateScalingSlopeOffset(int scaling_block, int type_central, int type_outer) const;
		int TwoLayerGateScalingShiftOffset(int scaling_block, int type_central, int type_outer) const;
		double TwoLayerGateScalingSlope(int scaling_block, int type_central, int type_outer) const;
		double TwoLayerGateScalingShift(int scaling_block, int type_central, int type_outer) const;
		int TwoLayerGateAdditiveCoeffIndex(int type_outer, int mu) const;
		const double* TwoLayerGateAdditiveCoeffRowData(int type_outer) const;
		double TwoLayerGateAdditiveCoeff(int type_outer, int mu) const;
		int TwoLayerGateTypeCoeffIndex(int type_outer, int mu) const;
		double TwoLayerGateTypeCoeff(int type_outer, int mu) const;
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
			bool SecondLayerGateUsesSharedRadial() const;
			bool SecondLayerGateUsesBodyLinearCombo() const;
			bool SecondLayerGateUsesFullScalarWeights() const;
			int SecondLayerGateRadialCoeffCount() const;
			int SecondLayerGateRadialCoeffOffset() const;
			int SecondLayerGateAdditiveCoeffCount() const;
			int SecondLayerGateAdditiveCoeffOffset() const;
			int SecondLayerGateTypeCoeffCount() const;
			int SecondLayerGateTypeCoeffOffset() const;
			int SecondLayerGateWeightCount() const;
			int SecondLayerGateWeightOffset() const;
			int SecondLayerGateBodyMixWeightCount() const;
			int SecondLayerGateBodyMixWeightOffset() const;
			int SecondLayerGateBodyOrderCount() const;
			int SecondLayerGateScalarCount() const;
			int SecondLayerGateMuBodyOrder(int mu) const;
			int SecondLayerGateWeightBodyOrder(int weight_index) const;
			void BuildSecondLayerGateBodyOrderBuckets();
			int SecondLayerGateRadialCoeffIndex(int mu, int xi) const;
			double SecondLayerGateRadialCoeff(int mu, int xi) const;
			int SecondLayerGateAdditiveCoeffIndex(int type_outer, int mu) const;
			double SecondLayerGateAdditiveCoeff(int type_outer, int mu) const;
			int SecondLayerGateTypeCoeffIndex(int type_outer, int mu) const;
			double SecondLayerGateTypeCoeff(int type_outer, int mu) const;
			int SecondLayerGateWeightIndex(int scalar_weight_index) const;
			int SecondLayerGateWeightIndex(int mu, int scalar_weight_index) const;
			int SecondLayerGateBodyMixWeightIndex(int mu, int body_order) const;
			double SecondLayerGateWeight(int mu, int scalar_weight_index) const;
			double SecondLayerGateWeight(int weight_index) const;
			double SecondLayerGateBodyMixWeight(int mu, int body_order) const;
			double SecondLayerGateBodyMixWeight(int weight_index) const;
			const double* SecondLayerGateScalarWeightData() const;
			const double* SecondLayerGateBodyMixWeightMatrixData() const;
			const double* SecondLayerGateFullWeightMatrixData() const;
			void ComputeSecondLayerGateBodySignals(const double* scalar_values,
			                                       double* body_values) const;
			void ComputeSecondLayerGateMuSignals(const double* body_values,
			                                     double* mu_values) const;
			void ComputeSecondLayerGateFullMuSignalsForAtoms(int atom_count);
			void AccumulateSecondLayerGateScalarSeedsFromMuAdjoints(
				const double* mu_adjoints,
				double* scalar_seeds,
				std::vector<double>& body_scratch) const;
			void AccumulateTwoLayerGateScalarSeedsFromMuAdjoints(
				const double* mu_adjoints,
				double* scalar_seeds,
				std::vector<double>& body_scratch) const;
			double TwoLayerResidualE0Coeff(int scalar_index) const;
			void InitializeTwoLayerGateRadialCoeffsFromBase();
			void InitializeTwoLayerGateScalingCoeffsFromBase();
			void InitializeTwoLayerGateAdditiveCoeffs(double value = 1.0 / 12.0);
			void InitializeTwoLayerGateTypeCoeffsFromBase();
			void InitializeSecondLayerGateRadialCoeffsFromFirstGate();
			void InitializeSecondLayerGateAdditiveCoeffs(double value = 1.0 / 12.0);
			void InitializeSecondLayerGateTypeCoeffsFromFirstGate();
			int CountRadialFirstCoeffsNeedingPositiveRepair(
				double min_value = 1.0e-12) const;
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
		void RandomizeTwoLayerGateRadialMixingCoeffs(
			std::mt19937_64& generator, double scale);
		void SyncTwoLayerGateRadialMixingCoeffsFromRegression();

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
						const int gate_scaling_count = TwoLayerGateScalingCoeffCount();
							const int gate_additive_count = TwoLayerGateAdditiveCoeffCount();
							const int gate_type_count = TwoLayerGateTypeCoeffCount();
							const int gate_weight_count = TwoLayerGateWeightCount();
							const int gate_body_mix_weight_count = TwoLayerGateBodyMixWeightCount();
							const int gate_radial_mixing_count =
								TwoLayerGateRadialMixingCoeffCount();
							const int gate_edge_l1_weight_count = TwoLayerGateEdgeL1WeightCount();
						const int second_gate_radial_count = SecondLayerGateRadialCoeffCount();
						const int second_gate_additive_count = SecondLayerGateAdditiveCoeffCount();
						const int second_gate_type_count = SecondLayerGateTypeCoeffCount();
						const int second_gate_weight_count = SecondLayerGateWeightCount();
						const int second_gate_body_mix_weight_count =
							SecondLayerGateBodyMixWeightCount();
							const int gate_readout_count = TwoLayerGateReadoutCoeffCount();
								const int e0_count = TwoLayerResidualE0CoeffCount();
							const int gate_k_rank_weight_count =
								TwoLayerGateKRankWeightCount();
							const int outer_k_rank1_weight_count = OuterKRank1WeightCount();
						const int linear_size = LinearCoeffCount();
					std::vector<double> old_coeffs = regression_coeffs;
					if (static_cast<int>(old_coeffs.size()) < base_nonlinear_size)
						ERROR("DistributeCoeffs found an inconsistent nonlinear coefficient block");
								const int new_gate_scaling_offset = base_nonlinear_size + gate_radial_count;
								const int new_gate_additive_offset =
									new_gate_scaling_offset + gate_scaling_count;
							const int new_gate_type_offset = new_gate_additive_offset + gate_additive_count;
						const int new_gate_offset = new_gate_type_offset + gate_type_count;
							const int new_gate_body_mix_offset = new_gate_offset + gate_weight_count;
								const int new_gate_edge_l1_offset =
									new_gate_body_mix_offset + gate_body_mix_weight_count
									+ gate_radial_mixing_count;
							const int new_second_gate_radial_offset =
								new_gate_edge_l1_offset + gate_edge_l1_weight_count;
							const int new_second_gate_additive_offset =
								new_second_gate_radial_offset + second_gate_radial_count;
							const int new_second_gate_type_offset =
								new_second_gate_additive_offset + second_gate_additive_count;
							const int new_second_gate_weight_offset =
								new_second_gate_type_offset + second_gate_type_count;
							const int new_second_gate_body_mix_offset =
								new_second_gate_weight_offset + second_gate_weight_count;
							const int new_gate_readout_offset =
								new_second_gate_body_mix_offset + second_gate_body_mix_weight_count;
								const int new_gate_k_rank_weight_offset =
									new_gate_readout_offset + gate_readout_count + e0_count;
								const int new_outer_k_rank1_weight_offset =
									new_gate_k_rank_weight_offset + gate_k_rank_weight_count;
							const int expected_new_size =
								new_outer_k_rank1_weight_offset
								+ outer_k_rank1_weight_count + linear_size;
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
							const bool old_coeffs_have_second_gate_layout =
								old_coeffs_have_release_layout
								&& multi_layer_gate_enabled_
								&& static_cast<int>(old_coeffs.size()) >=
									new_second_gate_body_mix_offset
									+ second_gate_body_mix_weight_count + e0_count;
						if (gate_radial_count > 0
					    && static_cast<int>(old_coeffs.size()) >= base_nonlinear_size + gate_radial_count) {
						two_layer_gate_radial_coeffs_.assign(gate_radial_count, 0.0);
						for (int i = 0; i < gate_radial_count; ++i)
							two_layer_gate_radial_coeffs_[i] =
								old_coeffs[base_nonlinear_size + i];
						}
						if (gate_scaling_count > 0) {
							if (static_cast<int>(two_layer_gate_scaling_coeffs_.size())
							    == gate_scaling_count) {
								// Model metadata is already in the current decoupled-scaling layout.
							} else if (old_coeffs_have_release_layout
							           && static_cast<int>(old_coeffs.size())
							              >= new_gate_scaling_offset + gate_scaling_count) {
								two_layer_gate_scaling_coeffs_.assign(gate_scaling_count, 0.0);
								for (int i = 0; i < gate_scaling_count; ++i)
									two_layer_gate_scaling_coeffs_[i] =
										old_coeffs[new_gate_scaling_offset + i];
							} else {
								InitializeTwoLayerGateScalingCoeffsFromBase();
							}
						} else {
							two_layer_gate_scaling_coeffs_.clear();
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
						if (gate_radial_mixing_count > 0
						    && static_cast<int>(two_layer_gate_radial_mixing_coeffs_.size())
						       != gate_radial_mixing_count) {
							two_layer_gate_radial_mixing_coeffs_.assign(
								gate_radial_mixing_count, 0.0);
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
								if (second_gate_radial_count > 0) {
									if (static_cast<int>(second_layer_gate_radial_coeffs_.size())
									    == second_gate_radial_count) {
										// Metadata is authoritative for release3 second-gate radial coeffs.
									} else if (static_cast<int>(old_coeffs.size())
									           >= new_second_gate_radial_offset
									              + second_gate_radial_count) {
										second_layer_gate_radial_coeffs_.assign(
											second_gate_radial_count, 0.0);
										for (int i = 0; i < second_gate_radial_count; ++i)
											second_layer_gate_radial_coeffs_[i] =
												old_coeffs[new_second_gate_radial_offset + i];
									} else {
										InitializeSecondLayerGateRadialCoeffsFromFirstGate();
									}
								}
								if (second_gate_additive_count > 0) {
									if (static_cast<int>(second_layer_gate_additive_coeffs_.size())
									    == second_gate_additive_count) {
										// Metadata is authoritative.
									} else if (static_cast<int>(old_coeffs.size())
									           >= new_second_gate_additive_offset
									              + second_gate_additive_count) {
										second_layer_gate_additive_coeffs_.assign(
											second_gate_additive_count, 0.0);
										for (int i = 0; i < second_gate_additive_count; ++i)
											second_layer_gate_additive_coeffs_[i] =
												old_coeffs[new_second_gate_additive_offset + i];
									} else {
										InitializeSecondLayerGateAdditiveCoeffs();
									}
								}
								if (second_gate_type_count > 0) {
									if (static_cast<int>(second_layer_gate_type_coeffs_.size())
									    == second_gate_type_count) {
										// Metadata is authoritative.
									} else if (static_cast<int>(old_coeffs.size())
									           >= new_second_gate_type_offset
									              + second_gate_type_count) {
										second_layer_gate_type_coeffs_.assign(
											second_gate_type_count, 0.0);
										for (int i = 0; i < second_gate_type_count; ++i)
											second_layer_gate_type_coeffs_[i] =
												old_coeffs[new_second_gate_type_offset + i];
									} else {
										InitializeSecondLayerGateTypeCoeffsFromFirstGate();
									}
								}
								if (second_gate_weight_count > 0) {
									if (static_cast<int>(second_layer_gate_weights_.size())
									    == second_gate_weight_count) {
										// Metadata is authoritative.
									} else if (static_cast<int>(old_coeffs.size())
									           >= new_second_gate_weight_offset
									              + second_gate_weight_count) {
										second_layer_gate_weights_.assign(
											second_gate_weight_count, 0.0);
										for (int i = 0; i < second_gate_weight_count; ++i)
											second_layer_gate_weights_[i] =
												old_coeffs[new_second_gate_weight_offset + i];
									}
								}
								if (second_gate_body_mix_weight_count > 0) {
									if (static_cast<int>(second_layer_gate_body_mix_weights_.size())
									    == second_gate_body_mix_weight_count) {
										// Metadata is authoritative.
									} else if (static_cast<int>(old_coeffs.size())
									           >= new_second_gate_body_mix_offset
									              + second_gate_body_mix_weight_count) {
										second_layer_gate_body_mix_weights_.assign(
											second_gate_body_mix_weight_count, 0.0);
										for (int i = 0; i < second_gate_body_mix_weight_count; ++i)
											second_layer_gate_body_mix_weights_[i] =
												old_coeffs[new_second_gate_body_mix_offset + i];
									} else {
										second_layer_gate_body_mix_weights_.assign(
											second_gate_body_mix_weight_count, 1.0);
									}
								}
							if (gate_readout_count > 0) {
								if (static_cast<int>(old_coeffs.size()) == expected_new_size) {
									two_layer_gate_readout_coeffs_.assign(
										gate_readout_count, 0.0);
									for (int i = 0; i < gate_readout_count; ++i)
										two_layer_gate_readout_coeffs_[i] =
											old_coeffs[new_gate_readout_offset + i];
								} else if (static_cast<int>(two_layer_gate_readout_coeffs_.size())
								           != gate_readout_count) {
									two_layer_gate_readout_coeffs_.assign(
										gate_readout_count, 0.0);
								}
							}
							if (e0_count > 0) {
								const int old_e0_offset =
									old_coeffs_have_second_gate_layout
										? new_second_gate_body_mix_offset
											+ second_gate_body_mix_weight_count
										: (old_coeffs_have_edge_l1_layout
											? new_gate_edge_l1_offset + gate_edge_l1_weight_count
										: (old_coeffs_have_release_layout
											? new_gate_body_mix_offset + gate_body_mix_weight_count
											: (old_coeffs_have_legacy_additive_layout
												? legacy_gate_offset + gate_weight_count
													+ gate_body_mix_weight_count
												: base_nonlinear_size + gate_radial_count
													+ gate_weight_count + gate_body_mix_weight_count)));
							if (static_cast<int>(old_coeffs.size()) >= old_e0_offset + e0_count) {
							two_layer_residual_e0_coeffs_.assign(e0_count, 0.0);
							for (int i = 0; i < e0_count; ++i)
								two_layer_residual_e0_coeffs_[i] = old_coeffs[old_e0_offset + i];
						}
					}
							if (gate_k_rank_weight_count > 0) {
								if (static_cast<int>(old_coeffs.size()) == expected_new_size) {
									two_layer_gate_k_rank_weights_.assign(
										gate_k_rank_weight_count, 0.0);
									for (int i = 0; i < gate_k_rank_weight_count; ++i)
										two_layer_gate_k_rank_weights_[i] =
											old_coeffs[new_gate_k_rank_weight_offset + i];
								} else if (static_cast<int>(
									       two_layer_gate_k_rank_weights_.size())
								           != gate_k_rank_weight_count) {
									ERROR("SUS2-SH gate K-rank weight metadata has inconsistent sizes");
								}
							}
								if (outer_k_rank1_weight_count > 0) {
								if (static_cast<int>(old_coeffs.size()) == expected_new_size) {
									outer_k_rank1_weights_.assign(
										outer_k_rank1_weight_count, 0.0);
									for (int i = 0; i < outer_k_rank1_weight_count; ++i)
										outer_k_rank1_weights_[i] =
											old_coeffs[new_outer_k_rank1_weight_offset + i];
								} else if (static_cast<int>(outer_k_rank1_weights_.size())
								           != outer_k_rank1_weight_count) {
									ERROR("SUS2-SH outer K-rank1 weight metadata has inconsistent sizes");
								}
							}
								regression_coeffs.assign(
									base_nonlinear_size + gate_radial_count
								+ gate_scaling_count + gate_additive_count
								+ gate_type_count + gate_weight_count
								+ gate_body_mix_weight_count
								+ gate_radial_mixing_count
								+ gate_edge_l1_weight_count
								+ second_gate_radial_count + second_gate_additive_count
								+ second_gate_type_count + second_gate_weight_count
									+ second_gate_body_mix_weight_count
									+ gate_readout_count + e0_count
									+ gate_k_rank_weight_count
									+ outer_k_rank1_weight_count + linear_size, 0.0);
				for (int i = 0; i < base_nonlinear_size; ++i)
					regression_coeffs[i] = old_coeffs[i];
				if (gate_radial_count > 0) {
				if (static_cast<int>(two_layer_gate_radial_coeffs_.size()) != gate_radial_count)
					InitializeTwoLayerGateRadialCoeffsFromBase();
				for (int i = 0; i < gate_radial_count; ++i)
					regression_coeffs[base_nonlinear_size + i] =
						two_layer_gate_radial_coeffs_[i];
			}
			if (gate_scaling_count > 0) {
				if (static_cast<int>(two_layer_gate_scaling_coeffs_.size())
				    != gate_scaling_count)
					InitializeTwoLayerGateScalingCoeffsFromBase();
				for (int i = 0; i < gate_scaling_count; ++i)
					regression_coeffs[TwoLayerGateScalingCoeffOffset() + i] =
						two_layer_gate_scaling_coeffs_[i];
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
					if (gate_radial_mixing_count > 0) {
						if (static_cast<int>(two_layer_gate_radial_mixing_coeffs_.size())
						    != gate_radial_mixing_count)
							two_layer_gate_radial_mixing_coeffs_.assign(
								gate_radial_mixing_count, 0.0);
						for (int i = 0; i < gate_radial_mixing_count; ++i)
							regression_coeffs[TwoLayerGateRadialMixingCoeffOffset() + i] =
								two_layer_gate_radial_mixing_coeffs_[i];
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
						if (second_gate_radial_count > 0) {
							if (static_cast<int>(second_layer_gate_radial_coeffs_.size())
							    != second_gate_radial_count)
								InitializeSecondLayerGateRadialCoeffsFromFirstGate();
							for (int i = 0; i < second_gate_radial_count; ++i)
								regression_coeffs[SecondLayerGateRadialCoeffOffset() + i] =
									second_layer_gate_radial_coeffs_[i];
						}
						if (second_gate_additive_count > 0) {
							if (static_cast<int>(second_layer_gate_additive_coeffs_.size())
							    != second_gate_additive_count)
								InitializeSecondLayerGateAdditiveCoeffs();
							for (int i = 0; i < second_gate_additive_count; ++i)
								regression_coeffs[SecondLayerGateAdditiveCoeffOffset() + i] =
									second_layer_gate_additive_coeffs_[i];
						}
						if (second_gate_type_count > 0) {
							if (static_cast<int>(second_layer_gate_type_coeffs_.size())
							    != second_gate_type_count)
								InitializeSecondLayerGateTypeCoeffsFromFirstGate();
							for (int i = 0; i < second_gate_type_count; ++i)
								regression_coeffs[SecondLayerGateTypeCoeffOffset() + i] =
									second_layer_gate_type_coeffs_[i];
						}
						if (second_gate_weight_count > 0) {
							if (static_cast<int>(second_layer_gate_weights_.size())
							    != second_gate_weight_count)
								ERROR("SUS2-SH second-layer gate metadata has inconsistent sizes");
							for (int i = 0; i < second_gate_weight_count; ++i)
								regression_coeffs[SecondLayerGateWeightOffset() + i] =
									second_layer_gate_weights_[i];
						}
							if (second_gate_body_mix_weight_count > 0) {
							if (static_cast<int>(second_layer_gate_body_mix_weights_.size())
							    != second_gate_body_mix_weight_count)
								second_layer_gate_body_mix_weights_.assign(
									second_gate_body_mix_weight_count, 1.0);
							for (int i = 0; i < second_gate_body_mix_weight_count; ++i)
								regression_coeffs[SecondLayerGateBodyMixWeightOffset() + i] =
									second_layer_gate_body_mix_weights_[i];
							}
							if (gate_readout_count > 0) {
								if (static_cast<int>(two_layer_gate_readout_coeffs_.size())
								    != gate_readout_count)
									ERROR("SUS2-SH gate readout metadata has inconsistent sizes");
								for (int i = 0; i < gate_readout_count; ++i)
									regression_coeffs[TwoLayerGateReadoutCoeffOffset() + i] =
										two_layer_gate_readout_coeffs_[i];
							}
					if (e0_count > 0) {
					if (static_cast<int>(two_layer_residual_e0_coeffs_.size()) != e0_count)
						two_layer_residual_e0_coeffs_.assign(e0_count, 0.0);
					for (int i = 0; i < e0_count; ++i)
						regression_coeffs[TwoLayerResidualE0CoeffOffset() + i] =
							two_layer_residual_e0_coeffs_[i];
				}
							if (gate_k_rank_weight_count > 0) {
								if (static_cast<int>(two_layer_gate_k_rank_weights_.size())
								    != gate_k_rank_weight_count)
									ERROR("SUS2-SH gate K-rank weight metadata has inconsistent sizes");
								for (int i = 0; i < gate_k_rank_weight_count; ++i)
									regression_coeffs[TwoLayerGateKRankWeightOffset() + i] =
										two_layer_gate_k_rank_weights_[i];
							}
							if (outer_k_rank1_weight_count > 0) {
							if (static_cast<int>(outer_k_rank1_weights_.size())
							    != outer_k_rank1_weight_count)
								ERROR("SUS2-SH outer K-rank1 weight metadata has inconsistent sizes");
							for (int i = 0; i < outer_k_rank1_weight_count; ++i)
								regression_coeffs[OuterKRank1WeightOffset() + i] =
									outer_k_rank1_weights_[i];
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
