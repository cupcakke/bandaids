pragma circom 2.1.8;

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/comparators.circom";
include "circomlib/circuits/bitify.circom";
include "circomlib/circuits/mux1.circom";

function FIXED_POINT_SCALE() {
    return 1000000;
}

function TAYLOR_COEFF_LINEAR() {
    return 1;
}

function TAYLOR_COEFF_QUADRATIC() {
    return FIXED_POINT_SCALE() / 2;
}

function TAYLOR_COEFF_CUBIC() {
    return FIXED_POINT_SCALE() / 6;
}

function REMAINDER_BIT_SIZE() {
    return 21;
}

function SHA256_DIGEST_SIZE() {
    return 32;
}

function DEFAULT_BATCH_SIZE() {
    return 64;
}

function EXP_INPUT_RANGE_BIT_SIZE() {
    return 22;
}

function EXP_INPUT_SIGNED_BIT_SIZE() {
    return 20;
}

function VALUE_BIT_SIZE() {
    return 64;
}

function COUNT_BIT_SIZE() {
    return 32;
}

function FIXED_PRODUCT_BIT_SIZE() {
    return 160;
}

function TAYLOR_FINAL_REMAINDER_BIT_SIZE() {
    return 63;
}

function DP_PRODUCT_BIT_SIZE() {
    return 96;
}

function TWO_TO_THE_POWER(n) {
    var result = 1;
    for (var i = 0; i < n; i++) {
        result = result * 2;
    }
    return result;
}

function CEIL_LOG2(n) {
    var value = 1;
    var result = 0;
    while (value < n) {
        value = value * 2;
        result = result + 1;
    }
    return result;
}

template SafeIsZero() {
    signal input in;
    signal output out;

    signal inv;
    inv <-- in != 0 ? 1 / in : 0;

    signal prod;
    prod <== in * inv;

    out <== 1 - prod;

    in * out === 0;
}

template SafeIsEqual() {
    signal input a;
    signal input b;
    signal output out;

    signal diff;
    diff <== a - b;

    component isz = SafeIsZero();
    isz.in <== diff;

    out <== isz.out;
}

template SignedAbs(bits) {
    assert(bits > 0);
    assert(bits < 251);

    signal input in;
    signal output abs;
    signal output is_negative;

    var offset = TWO_TO_THE_POWER(bits);

    signal shifted;
    shifted <== in + offset;

    component shifted_bits = Num2Bits(bits + 1);
    shifted_bits.in <== shifted;

    component negative_check = LessThan(bits + 1);
    negative_check.in[0] <== shifted;
    negative_check.in[1] <== offset;

    is_negative <== negative_check.out;

    signal negated;
    negated <== 0 - in;

    component abs_mux = Mux1();
    abs_mux.c[0] <== in;
    abs_mux.c[1] <== negated;
    abs_mux.s <== is_negative;

    abs <== abs_mux.out;

    component abs_bits = Num2Bits(bits + 1);
    abs_bits.in <== abs;
}

template SignedDivByConstant(bits, divisor, remainder_bits) {
    assert(bits > 0);
    assert(bits < 251);
    assert(divisor > 0);
    assert(remainder_bits > 0);
    assert(remainder_bits < 251);
    assert(divisor < TWO_TO_THE_POWER(remainder_bits));

    signal input in;
    signal output quotient;
    signal output remainder;

    component abs_input = SignedAbs(bits);
    abs_input.in <== in;

    signal quotient_abs;
    quotient_abs <-- abs_input.abs \ divisor;

    remainder <-- abs_input.abs % divisor;

    quotient_abs * divisor + remainder === abs_input.abs;

    component quotient_abs_bits = Num2Bits(bits);
    quotient_abs_bits.in <== quotient_abs;

    component remainder_check = LessThan(remainder_bits);
    remainder_check.in[0] <== remainder;
    remainder_check.in[1] <== divisor;
    remainder_check.out === 1;

    signal quotient_abs_negated;
    quotient_abs_negated <== 0 - quotient_abs;

    component quotient_mux = Mux1();
    quotient_mux.c[0] <== quotient_abs;
    quotient_mux.c[1] <== quotient_abs_negated;
    quotient_mux.s <== abs_input.is_negative;

    quotient <== quotient_mux.out;
}

template PoseidonCommit() {
    signal input value;
    signal input blinding;
    signal output commitment;

    component hasher = Poseidon(2);
    hasher.inputs[0] <== value;
    hasher.inputs[1] <== blinding;

    commitment <== hasher.out;
}

template PoseidonChain(n) {
    assert(n > 0);

    signal input in[n];
    signal output out;

    var num_chunks = (n + 5) \ 6;

    signal intermediate[num_chunks];
    component hashers[num_chunks];

    for (var chunk = 0; chunk < num_chunks; chunk++) {
        hashers[chunk] = Poseidon(8);

        for (var j = 0; j < 6; j++) {
            if (chunk * 6 + j < n) {
                hashers[chunk].inputs[j] <== in[chunk * 6 + j];
            } else {
                hashers[chunk].inputs[j] <== 0;
            }
        }

        if (chunk > 0) {
            hashers[chunk].inputs[6] <== intermediate[chunk - 1];
        } else {
            hashers[chunk].inputs[6] <== 0;
        }

        hashers[chunk].inputs[7] <== n;

        intermediate[chunk] <== hashers[chunk].out;
    }

    out <== intermediate[num_chunks - 1];
}

template VerifyMerkleProof(depth) {
    assert(depth > 0);

    signal input leaf;
    signal input path_elements[depth];
    signal input path_indices[depth];
    signal output root;

    signal hashes[depth + 1];
    hashes[0] <== leaf;

    component hashers[depth];
    component muxers_left[depth];
    component muxers_right[depth];

    for (var i = 0; i < depth; i++) {
        path_indices[i] * (path_indices[i] - 1) === 0;

        muxers_left[i] = Mux1();
        muxers_left[i].c[0] <== hashes[i];
        muxers_left[i].c[1] <== path_elements[i];
        muxers_left[i].s <== path_indices[i];

        muxers_right[i] = Mux1();
        muxers_right[i].c[0] <== path_elements[i];
        muxers_right[i].c[1] <== hashes[i];
        muxers_right[i].s <== path_indices[i];

        hashers[i] = Poseidon(2);
        hashers[i].inputs[0] <== muxers_left[i].out;
        hashers[i].inputs[1] <== muxers_right[i].out;

        hashes[i + 1] <== hashers[i].out;
    }

    root <== hashes[depth];
}

template RangeProof(bits) {
    assert(bits > 0);
    assert(bits < 251);

    signal input value;
    signal input min_value;
    signal input max_value;
    signal input commitments[bits];
    signal input openings[bits];
    signal output valid;

    signal normalized;
    normalized <== value - min_value;

    component bit_decomp = Num2Bits(bits);
    bit_decomp.in <== normalized;

    component lt_check = LessThan(bits + 1);
    lt_check.in[0] <== normalized;
    lt_check.in[1] <== max_value - min_value + 1;

    component hash_commits[bits];
    component eq_checks[bits];
    signal bit_valid[bits];

    for (var i = 0; i < bits; i++) {
        hash_commits[i] = PoseidonCommit();
        hash_commits[i].value <== bit_decomp.out[i];
        hash_commits[i].blinding <== openings[i];

        eq_checks[i] = SafeIsEqual();
        eq_checks[i].a <== hash_commits[i].commitment;
        eq_checks[i].b <== commitments[i];

        bit_valid[i] <== eq_checks[i].out;
    }

    signal all_valid[bits + 1];
    all_valid[0] <== 1;

    for (var i = 0; i < bits; i++) {
        all_valid[i + 1] <== all_valid[i] * bit_valid[i];
    }

    valid <== all_valid[bits] * lt_check.out;
}

template RSFLayerComputation(dim) {
    assert(dim > 0);
    assert(dim % 2 == 0);

    signal input x[dim];
    signal input weights_s[dim / 2][dim / 2 + 1];
    signal input weights_t[dim / 2][dim / 2 + 1];
    signal input expected_commitment;
    signal output y[dim];
    signal output valid_commitment;

    var half = dim / 2;

    signal x1[half];
    signal x2[half];

    for (var i = 0; i < half; i++) {
        x1[i] <== x[i];
        x2[i] <== x[half + i];
    }

    signal s_partial[half][half + 1];
    signal s_x2[half];

    for (var i = 0; i < half; i++) {
        s_partial[i][0] <== 0;

        for (var j = 0; j < half; j++) {
            s_partial[i][j + 1] <== s_partial[i][j] + weights_s[i][j] * x2[j];
        }

        s_x2[i] <== s_partial[i][half] + weights_s[i][half];
    }

    signal y1[half];

    signal s_val[half];
    signal s_sq[half];
    signal s_cu[half];
    signal exp_numerator[half];
    signal x1_exp_numerator[half];

    component s_val_range_low[half];
    component s_val_range_high[half];
    component exp_numerator_bits[half];
    component final_division[half];

    for (var i = 0; i < half; i++) {
        s_val[i] <== s_x2[i];

        s_val_range_low[i] = LessThan(EXP_INPUT_RANGE_BIT_SIZE());
        s_val_range_low[i].in[0] <== FIXED_POINT_SCALE() - s_val[i];
        s_val_range_low[i].in[1] <== 2 * FIXED_POINT_SCALE() + 1;
        s_val_range_low[i].out === 1;

        s_val_range_high[i] = LessThan(EXP_INPUT_RANGE_BIT_SIZE());
        s_val_range_high[i].in[0] <== s_val[i] + FIXED_POINT_SCALE();
        s_val_range_high[i].in[1] <== 2 * FIXED_POINT_SCALE() + 1;
        s_val_range_high[i].out === 1;

        s_sq[i] <== s_val[i] * s_val[i];
        s_cu[i] <== s_sq[i] * s_val[i];

        exp_numerator[i] <== FIXED_POINT_SCALE() * 6 * FIXED_POINT_SCALE() * FIXED_POINT_SCALE() + s_val[i] * 6 * FIXED_POINT_SCALE() * FIXED_POINT_SCALE() + s_sq[i] * 3 * FIXED_POINT_SCALE() + s_cu[i];

        exp_numerator_bits[i] = Num2Bits(65);
        exp_numerator_bits[i].in <== exp_numerator[i];

        x1_exp_numerator[i] <== x1[i] * exp_numerator[i];

        final_division[i] = SignedDivByConstant(FIXED_PRODUCT_BIT_SIZE(), 6 * FIXED_POINT_SCALE() * FIXED_POINT_SCALE() * FIXED_POINT_SCALE(), TAYLOR_FINAL_REMAINDER_BIT_SIZE());
        final_division[i].in <== x1_exp_numerator[i];

        y1[i] <== final_division[i].quotient;
    }

    signal t_partial[half][half + 1];
    signal t_y1[half];

    for (var i = 0; i < half; i++) {
        t_partial[i][0] <== 0;

        for (var j = 0; j < half; j++) {
            t_partial[i][j + 1] <== t_partial[i][j] + weights_t[i][j] * y1[j];
        }

        t_y1[i] <== t_partial[i][half] + weights_t[i][half];
    }

    signal y2[half];

    for (var i = 0; i < half; i++) {
        y2[i] <== x2[i] + t_y1[i];
    }

    for (var i = 0; i < half; i++) {
        y[i] <== y1[i];
        y[half + i] <== y2[i];
    }

    component output_hash = PoseidonChain(dim);

    for (var i = 0; i < dim; i++) {
        output_hash.in[i] <== y[i];
    }

    component commit_check = SafeIsEqual();
    commit_check.a <== output_hash.out;
    commit_check.b <== expected_commitment;

    valid_commitment <== commit_check.out;
}

template VerifyBatchInference(batch_size, dim) {
    assert(batch_size > 0);
    assert(dim > 0);

    signal input inputs[batch_size][dim];
    signal input outputs[batch_size][dim];
    signal input commitments[batch_size];
    signal input expected_root;
    signal output valid;

    signal computed_commits[batch_size];
    component hashers[batch_size];
    component commit_checks[batch_size];

    for (var b = 0; b < batch_size; b++) {
        hashers[b] = PoseidonChain(dim);

        for (var i = 0; i < dim; i++) {
            hashers[b].in[i] <== outputs[b][i];
        }

        computed_commits[b] <== hashers[b].out;

        commit_checks[b] = SafeIsEqual();
        commit_checks[b].a <== computed_commits[b];
        commit_checks[b].b <== commitments[b];
    }

    var tree_depth = 0;
    var temp_size = batch_size;

    while (temp_size > 1) {
        temp_size = (temp_size + 1) \ 2;
        tree_depth = tree_depth + 1;
    }

    var tree_depth_alloc = 1;

    if (tree_depth > 0) {
        tree_depth_alloc = tree_depth;
    }

    signal tree_nodes[tree_depth + 1][batch_size];

    for (var i = 0; i < batch_size; i++) {
        tree_nodes[0][i] <== commitments[i];
    }

    component tree_hashers[tree_depth_alloc][batch_size];

    var current_width = batch_size;

    for (var level = 0; level < tree_depth_alloc; level++) {
        if (level < tree_depth) {
            var next_width = (current_width + 1) \ 2;

            for (var i = 0; i < batch_size; i++) {
                tree_hashers[level][i] = Poseidon(2);

                if (i < next_width) {
                    tree_hashers[level][i].inputs[0] <== tree_nodes[level][i * 2];

                    if (i * 2 + 1 < current_width) {
                        tree_hashers[level][i].inputs[1] <== tree_nodes[level][i * 2 + 1];
                    } else {
                        tree_hashers[level][i].inputs[1] <== tree_nodes[level][i * 2];
                    }

                    tree_nodes[level + 1][i] <== tree_hashers[level][i].out;
                } else {
                    tree_hashers[level][i].inputs[0] <== 0;
                    tree_hashers[level][i].inputs[1] <== 0;
                    tree_nodes[level + 1][i] <== 0;
                }
            }

            current_width = next_width;
        } else {
            for (var i = 0; i < batch_size; i++) {
                tree_hashers[level][i] = Poseidon(2);
                tree_hashers[level][i].inputs[0] <== 0;
                tree_hashers[level][i].inputs[1] <== 0;
            }
        }
    }

    component root_check = SafeIsEqual();
    root_check.a <== tree_nodes[tree_depth][0];
    root_check.b <== expected_root;

    signal commit_valid[batch_size + 1];
    commit_valid[0] <== 1;

    for (var b = 0; b < batch_size; b++) {
        commit_valid[b + 1] <== commit_valid[b] * commit_checks[b].out;
    }

    valid <== commit_valid[batch_size] * root_check.out;
}

template VerifyNoiseBound(dim, precision_bits) {
    assert(dim > 0);
    assert(precision_bits > 0);
    assert(precision_bits < 251);

    signal input original[dim];
    signal input noisy[dim];
    signal input max_noise;
    signal output valid;

    signal noise[dim];
    signal abs_noise[dim];

    component abs_components[dim];
    component bound_checks[dim];

    for (var i = 0; i < dim; i++) {
        noise[i] <== noisy[i] - original[i];

        abs_components[i] = SignedAbs(precision_bits);
        abs_components[i].in <== noise[i];

        abs_noise[i] <== abs_components[i].abs;

        bound_checks[i] = LessThan(precision_bits + 1);
        bound_checks[i].in[0] <== abs_noise[i];
        bound_checks[i].in[1] <== max_noise + 1;
    }

    signal all_valid[dim + 1];
    all_valid[0] <== 1;

    for (var i = 0; i < dim; i++) {
        all_valid[i + 1] <== all_valid[i] * bound_checks[i].out;
    }

    valid <== all_valid[dim];
}

template VerifyAggregation(num_participants, dim) {
    assert(num_participants > 0);
    assert(dim > 0);

    signal input contributions[num_participants][dim];
    signal input participant_commitments[num_participants];
    signal input aggregated_result[dim];
    signal input min_threshold;
    signal input max_contribution;
    signal output valid;

    component commit_checks[num_participants];
    component commit_eq[num_participants];
    signal commit_valid[num_participants];

    for (var p = 0; p < num_participants; p++) {
        commit_checks[p] = PoseidonChain(dim);

        for (var i = 0; i < dim; i++) {
            commit_checks[p].in[i] <== contributions[p][i];
        }

        commit_eq[p] = SafeIsEqual();
        commit_eq[p].a <== commit_checks[p].out;
        commit_eq[p].b <== participant_commitments[p];
        commit_valid[p] <== commit_eq[p].out;
    }

    component contribution_abs[num_participants][dim];
    component contribution_bounds[num_participants][dim];
    signal contribution_valid[num_participants][dim];

    for (var p = 0; p < num_participants; p++) {
        for (var i = 0; i < dim; i++) {
            contribution_abs[p][i] = SignedAbs(VALUE_BIT_SIZE());
            contribution_abs[p][i].in <== contributions[p][i];

            contribution_bounds[p][i] = LessThan(VALUE_BIT_SIZE() + 1);
            contribution_bounds[p][i].in[0] <== contribution_abs[p][i].abs;
            contribution_bounds[p][i].in[1] <== max_contribution + 1;

            contribution_valid[p][i] <== contribution_bounds[p][i].out;
        }
    }

    signal partial_sums[dim][num_participants + 1];
    signal sums[dim];

    for (var i = 0; i < dim; i++) {
        partial_sums[i][0] <== 0;

        for (var p = 0; p < num_participants; p++) {
            partial_sums[i][p + 1] <== partial_sums[i][p] + contributions[p][i];
        }

        sums[i] <== partial_sums[i][num_participants];
    }

    component result_checks[dim];
    signal result_valid[dim];

    for (var i = 0; i < dim; i++) {
        result_checks[i] = SafeIsEqual();
        result_checks[i].a <== sums[i];
        result_checks[i].b <== aggregated_result[i];
        result_valid[i] <== result_checks[i].out;
    }

    component threshold_upper_check = LessThan(COUNT_BIT_SIZE());
    threshold_upper_check.in[0] <== min_threshold;
    threshold_upper_check.in[1] <== num_participants + 1;

    component threshold_zero_check = SafeIsZero();
    threshold_zero_check.in <== min_threshold;

    signal threshold_nonzero;
    threshold_nonzero <== 1 - threshold_zero_check.out;

    signal threshold_valid;
    threshold_valid <== threshold_upper_check.out * threshold_nonzero;

    signal all_commits[num_participants + 1];
    all_commits[0] <== 1;

    for (var p = 0; p < num_participants; p++) {
        all_commits[p + 1] <== all_commits[p] * commit_valid[p];
    }

    signal participant_bound_products[num_participants][dim + 1];

    for (var p = 0; p < num_participants; p++) {
        participant_bound_products[p][0] <== 1;

        for (var i = 0; i < dim; i++) {
            participant_bound_products[p][i + 1] <== participant_bound_products[p][i] * contribution_valid[p][i];
        }
    }

    signal all_contribution_bounds[num_participants + 1];
    all_contribution_bounds[0] <== 1;

    for (var p = 0; p < num_participants; p++) {
        all_contribution_bounds[p + 1] <== all_contribution_bounds[p] * participant_bound_products[p][dim];
    }

    signal all_results[dim + 1];
    all_results[0] <== 1;

    for (var i = 0; i < dim; i++) {
        all_results[i + 1] <== all_results[i] * result_valid[i];
    }

    signal commits_and_results_valid;
    commits_and_results_valid <== all_commits[num_participants] * all_results[dim];

    signal bounds_and_threshold_valid;
    bounds_and_threshold_valid <== all_contribution_bounds[num_participants] * threshold_valid;

    valid <== commits_and_results_valid * bounds_and_threshold_valid;
}

template DifferentialPrivacyProof(dim) {
    assert(dim > 0);

    signal input original[dim];
    signal input noisy[dim];
    signal input epsilon;
    signal input sensitivity;
    signal input noise_commitments[dim];
    signal input noise_blindings[dim];
    signal output valid;

    component epsilon_bits = Num2Bits(VALUE_BIT_SIZE());
    epsilon_bits.in <== epsilon;

    component sensitivity_bits = Num2Bits(VALUE_BIT_SIZE());
    sensitivity_bits.in <== sensitivity;

    component epsilon_zero_check = SafeIsZero();
    epsilon_zero_check.in <== epsilon;

    signal epsilon_is_zero;
    epsilon_is_zero <== epsilon_zero_check.out;

    signal epsilon_nonzero;
    epsilon_nonzero <== 1 - epsilon_is_zero;

    signal epsilon_denominator;
    epsilon_denominator <== epsilon + epsilon_is_zero;

    signal max_noise_numerator;
    max_noise_numerator <== sensitivity * FIXED_POINT_SCALE();

    component numerator_bits = Num2Bits(DP_PRODUCT_BIT_SIZE());
    numerator_bits.in <== max_noise_numerator;

    signal max_noise_quotient;
    signal max_noise_remainder;

    max_noise_quotient <-- epsilon != 0 ? max_noise_numerator \ epsilon : max_noise_numerator;
    max_noise_remainder <-- epsilon != 0 ? max_noise_numerator % epsilon : 0;

    max_noise_quotient * epsilon_denominator + max_noise_remainder === max_noise_numerator;

    component quotient_bits = Num2Bits(DP_PRODUCT_BIT_SIZE());
    quotient_bits.in <== max_noise_quotient;

    component remainder_check = LessThan(VALUE_BIT_SIZE() + 1);
    remainder_check.in[0] <== max_noise_remainder;
    remainder_check.in[1] <== epsilon_denominator;

    signal noise[dim];
    signal abs_noise[dim];

    component abs_components[dim];
    component bound_checks[dim];
    component commit_checks[dim];
    component commit_eq_check[dim];
    component blinding_zero_check[dim];

    signal commit_eq[dim];
    signal blinding_nonzero[dim];

    for (var i = 0; i < dim; i++) {
        noise[i] <== noisy[i] - original[i];

        abs_components[i] = SignedAbs(VALUE_BIT_SIZE());
        abs_components[i].in <== noise[i];

        abs_noise[i] <== abs_components[i].abs;

        bound_checks[i] = LessThan(DP_PRODUCT_BIT_SIZE());
        bound_checks[i].in[0] <== abs_noise[i];
        bound_checks[i].in[1] <== max_noise_quotient + 1;

        commit_checks[i] = PoseidonCommit();
        commit_checks[i].value <== noise[i];
        commit_checks[i].blinding <== noise_blindings[i];

        commit_eq_check[i] = SafeIsEqual();
        commit_eq_check[i].a <== commit_checks[i].commitment;
        commit_eq_check[i].b <== noise_commitments[i];

        commit_eq[i] <== commit_eq_check[i].out;

        blinding_zero_check[i] = SafeIsZero();
        blinding_zero_check[i].in <== noise_blindings[i];

        blinding_nonzero[i] <== 1 - blinding_zero_check[i].out;
    }

    signal all_bounds[dim + 1];
    signal all_commits[dim + 1];
    signal all_blindings[dim + 1];

    all_bounds[0] <== 1;
    all_commits[0] <== 1;
    all_blindings[0] <== 1;

    for (var i = 0; i < dim; i++) {
        all_bounds[i + 1] <== all_bounds[i] * bound_checks[i].out;
        all_commits[i + 1] <== all_commits[i] * commit_eq[i];
        all_blindings[i + 1] <== all_blindings[i] * blinding_nonzero[i];
    }

    signal epsilon_and_remainder_valid;
    epsilon_and_remainder_valid <== epsilon_nonzero * remainder_check.out;

    signal privacy_commitments_valid;
    privacy_commitments_valid <== all_commits[dim] * all_blindings[dim];

    signal privacy_bounds_valid;
    privacy_bounds_valid <== all_bounds[dim] * epsilon_and_remainder_valid;

    valid <== privacy_commitments_valid * privacy_bounds_valid;
}

template SecureAggregationProof(num_participants, dim) {
    assert(num_participants > 0);
    assert(dim > 0);

    signal input contributions[num_participants][dim];
    signal input participant_commitments[num_participants];
    signal input aggregated_result[dim];
    signal input threshold;
    signal input max_contribution;
    signal output valid;

    component threshold_upper_check = LessThan(COUNT_BIT_SIZE());
    threshold_upper_check.in[0] <== threshold;
    threshold_upper_check.in[1] <== num_participants + 1;

    component threshold_zero_check = SafeIsZero();
    threshold_zero_check.in <== threshold;

    signal threshold_nonzero;
    threshold_nonzero <== 1 - threshold_zero_check.out;

    signal threshold_valid;
    threshold_valid <== threshold_upper_check.out * threshold_nonzero;

    component commit_hashes[num_participants];
    component commit_eq[num_participants];
    signal commit_valid[num_participants];

    for (var p = 0; p < num_participants; p++) {
        commit_hashes[p] = PoseidonChain(dim);

        for (var i = 0; i < dim; i++) {
            commit_hashes[p].in[i] <== contributions[p][i];
        }

        commit_eq[p] = SafeIsEqual();
        commit_eq[p].a <== commit_hashes[p].out;
        commit_eq[p].b <== participant_commitments[p];

        commit_valid[p] <== commit_eq[p].out;
    }

    component contribution_abs[num_participants][dim];
    component contribution_bounds[num_participants][dim];
    signal contribution_valid[num_participants][dim];

    for (var p = 0; p < num_participants; p++) {
        for (var i = 0; i < dim; i++) {
            contribution_abs[p][i] = SignedAbs(VALUE_BIT_SIZE());
            contribution_abs[p][i].in <== contributions[p][i];

            contribution_bounds[p][i] = LessThan(VALUE_BIT_SIZE() + 1);
            contribution_bounds[p][i].in[0] <== contribution_abs[p][i].abs;
            contribution_bounds[p][i].in[1] <== max_contribution + 1;

            contribution_valid[p][i] <== contribution_bounds[p][i].out;
        }
    }

    signal partial_sums[dim][num_participants + 1];
    signal sums[dim];

    for (var i = 0; i < dim; i++) {
        partial_sums[i][0] <== 0;

        for (var p = 0; p < num_participants; p++) {
            partial_sums[i][p + 1] <== partial_sums[i][p] + contributions[p][i];
        }

        sums[i] <== partial_sums[i][num_participants];
    }

    component result_eq[dim];
    signal result_valid[dim];

    for (var i = 0; i < dim; i++) {
        result_eq[i] = SafeIsEqual();
        result_eq[i].a <== sums[i];
        result_eq[i].b <== aggregated_result[i];

        result_valid[i] <== result_eq[i].out;
    }

    signal all_commits[num_participants + 1];
    all_commits[0] <== 1;

    for (var p = 0; p < num_participants; p++) {
        all_commits[p + 1] <== all_commits[p] * commit_valid[p];
    }

    signal participant_bound_products[num_participants][dim + 1];

    for (var p = 0; p < num_participants; p++) {
        participant_bound_products[p][0] <== 1;

        for (var i = 0; i < dim; i++) {
            participant_bound_products[p][i + 1] <== participant_bound_products[p][i] * contribution_valid[p][i];
        }
    }

    signal all_contribution_bounds[num_participants + 1];
    all_contribution_bounds[0] <== 1;

    for (var p = 0; p < num_participants; p++) {
        all_contribution_bounds[p + 1] <== all_contribution_bounds[p] * participant_bound_products[p][dim];
    }

    signal all_results[dim + 1];
    all_results[0] <== 1;

    for (var i = 0; i < dim; i++) {
        all_results[i + 1] <== all_results[i] * result_valid[i];
    }

    signal commits_and_results_valid;
    commits_and_results_valid <== all_commits[num_participants] * all_results[dim];

    signal bounds_and_threshold_valid;
    bounds_and_threshold_valid <== all_contribution_bounds[num_participants] * threshold_valid;

    valid <== commits_and_results_valid * bounds_and_threshold_valid;
}

template FullInferenceProof(num_layers, dim, precision_bits) {
    assert(num_layers > 0);
    assert(dim > 0);
    assert(dim % 2 == 0);
    assert(precision_bits > 0);
    assert(precision_bits < 120);

    signal input tokens[dim];
    signal input layer_weights_s[num_layers][dim / 2][dim / 2 + 1];
    signal input layer_weights_t[num_layers][dim / 2][dim / 2 + 1];
    signal input expected_output[dim];
    signal input input_commitment;
    signal input output_commitment;
    signal input layer_commitments[num_layers];
    signal input max_error_squared;
    signal output y[dim];
    signal output is_valid;

    signal layer_outputs[num_layers + 1][dim];

    for (var i = 0; i < dim; i++) {
        layer_outputs[0][i] <== tokens[i];
    }

    component input_hash = PoseidonChain(dim);

    for (var i = 0; i < dim; i++) {
        input_hash.in[i] <== tokens[i];
    }

    component input_check = SafeIsEqual();
    input_check.a <== input_hash.out;
    input_check.b <== input_commitment;

    component rsf_layers[num_layers];
    signal layer_valid[num_layers];

    for (var layer = 0; layer < num_layers; layer++) {
        rsf_layers[layer] = RSFLayerComputation(dim);

        for (var i = 0; i < dim / 2; i++) {
            for (var j = 0; j < dim / 2 + 1; j++) {
                rsf_layers[layer].weights_s[i][j] <== layer_weights_s[layer][i][j];
                rsf_layers[layer].weights_t[i][j] <== layer_weights_t[layer][i][j];
            }
        }

        for (var i = 0; i < dim; i++) {
            rsf_layers[layer].x[i] <== layer_outputs[layer][i];
        }

        rsf_layers[layer].expected_commitment <== layer_commitments[layer];

        for (var i = 0; i < dim; i++) {
            layer_outputs[layer + 1][i] <== rsf_layers[layer].y[i];
        }

        layer_valid[layer] <== rsf_layers[layer].valid_commitment;
    }

    for (var i = 0; i < dim; i++) {
        y[i] <== layer_outputs[num_layers][i];
    }

    component output_hash = PoseidonChain(dim);

    for (var i = 0; i < dim; i++) {
        output_hash.in[i] <== layer_outputs[num_layers][i];
    }

    component output_check = SafeIsEqual();
    output_check.a <== output_hash.out;
    output_check.b <== output_commitment;

    signal diff[dim];
    signal abs_diff[dim];
    signal diff_squared[dim];

    component diff_abs_components[dim];

    for (var i = 0; i < dim; i++) {
        diff[i] <== layer_outputs[num_layers][i] - expected_output[i];

        diff_abs_components[i] = SignedAbs(precision_bits);
        diff_abs_components[i].in <== diff[i];

        abs_diff[i] <== diff_abs_components[i].abs;
        diff_squared[i] <== abs_diff[i] * abs_diff[i];
    }

    signal error_sum[dim + 1];

    error_sum[0] <== 0;

    for (var i = 0; i < dim; i++) {
        error_sum[i + 1] <== error_sum[i] + diff_squared[i];
    }

    var error_sum_bits = 2 * precision_bits + CEIL_LOG2(dim) + 1;

    component error_check = LessThan(error_sum_bits);
    error_check.in[0] <== error_sum[dim];
    error_check.in[1] <== max_error_squared + 1;

    signal all_layers_valid[num_layers + 1];

    all_layers_valid[0] <== 1;

    for (var layer = 0; layer < num_layers; layer++) {
        all_layers_valid[layer + 1] <== all_layers_valid[layer] * layer_valid[layer];
    }

    signal input_and_output_valid;
    input_and_output_valid <== input_check.out * output_check.out;

    signal io_and_error_valid;
    io_and_error_valid <== input_and_output_valid * error_check.out;

    is_valid <== io_and_error_valid * all_layers_valid[num_layers];
}

template InferenceTraceWithBatch(num_layers, dim, batch_size, precision_bits) {
    assert(num_layers > 0);
    assert(dim > 0);
    assert(dim % 2 == 0);
    assert(batch_size > 0);
    assert(precision_bits > 0);
    assert(precision_bits < 120);

    signal input tokens[batch_size][dim];
    signal input layer_weights_s[num_layers][dim / 2][dim / 2 + 1];
    signal input layer_weights_t[num_layers][dim / 2][dim / 2 + 1];
    signal input expected_outputs[batch_size][dim];
    signal input input_commitments[batch_size];
    signal input output_commitments[batch_size];
    signal input layer_commitments[batch_size][num_layers];
    signal input max_error_squared;
    signal input batch_root;
    signal output is_valid;

    component inference_proofs[batch_size];
    signal batch_valid[batch_size];

    for (var b = 0; b < batch_size; b++) {
        inference_proofs[b] = FullInferenceProof(num_layers, dim, precision_bits);

        for (var i = 0; i < dim; i++) {
            inference_proofs[b].tokens[i] <== tokens[b][i];
            inference_proofs[b].expected_output[i] <== expected_outputs[b][i];
        }

        for (var layer = 0; layer < num_layers; layer++) {
            for (var i = 0; i < dim / 2; i++) {
                for (var j = 0; j < dim / 2 + 1; j++) {
                    inference_proofs[b].layer_weights_s[layer][i][j] <== layer_weights_s[layer][i][j];
                    inference_proofs[b].layer_weights_t[layer][i][j] <== layer_weights_t[layer][i][j];
                }
            }

            inference_proofs[b].layer_commitments[layer] <== layer_commitments[b][layer];
        }

        inference_proofs[b].input_commitment <== input_commitments[b];
        inference_proofs[b].output_commitment <== output_commitments[b];
        inference_proofs[b].max_error_squared <== max_error_squared;

        batch_valid[b] <== inference_proofs[b].is_valid;
    }

    component batch_verify = VerifyBatchInference(batch_size, dim);

    for (var b = 0; b < batch_size; b++) {
        for (var i = 0; i < dim; i++) {
            batch_verify.inputs[b][i] <== tokens[b][i];
            batch_verify.outputs[b][i] <== inference_proofs[b].y[i];
        }

        batch_verify.commitments[b] <== output_commitments[b];
    }

    batch_verify.expected_root <== batch_root;

    signal all_batch_valid[batch_size + 1];

    all_batch_valid[0] <== 1;

    for (var b = 0; b < batch_size; b++) {
        all_batch_valid[b + 1] <== all_batch_valid[b] * batch_valid[b];
    }

    is_valid <== all_batch_valid[batch_size] * batch_verify.valid;
}

component main {public [tokens, expected_output, input_commitment, output_commitment]} = FullInferenceProof(8, 32, 64);
