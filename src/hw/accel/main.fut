let matmul_f16 [m][n][k] (a: [m][k]f16) (b: [k][n]f16) : *[m][n]f16 =
  let bt = transpose b
  in map (\a_row ->
    map (\b_col ->
      f16.sum (map2 (f16.*) a_row b_col)
    ) bt
  ) a

entry rsf_forward [n][half] (input: [n][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16) : *[n][half*2]f16 =
  let d = half * 2
  in map (\row ->
    let x1 = map f32.f16 (row[0:half] :> [half]f16)
    let x2 = map f32.f16 (row[half:d] :> [half]f16)
    let scale = map (\j ->
      let sum = f32.f16 weights_s[j][half] + f32.sum (map2 (\w x -> f32.f16 w * x) (weights_s[j][0:half] :> [half]f16) x2)
      let clipped = f32.max (f32.f16 clip_min) (f32.min (f32.f16 clip_max) sum)
      in f32.exp clipped
    ) (iota half)
    let y1 = map2 (*) x1 scale
    let trans = map (\j ->
      let value = f32.f16 weights_t[j][half] + f32.sum (map2 (\w x -> f32.f16 w * x) (weights_t[j][0:half] :> [half]f16) y1)
      in if f32.isnan value || f32.isinf value then 0f32 else value
    ) (iota half)
    let y2 = map2 (+) x2 trans
    let output = map (\value ->
      let safe_value = if f32.isnan value || f32.isinf value then 0f32 else f32.max (-60000f32) (f32.min 60000f32 value)
      in f16.f32 safe_value
    ) (y1 ++ y2)
    in output :> [half*2]f16
  ) input

entry rsf_backward [n][half] (input: [n][half*2]f16) (grad_output: [n][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : ([half][half+1]f16, [half][half+1]f16) =
  let weights_t_body = map (\row -> row[0:half] :> [half]f16) weights_t
  let weights_t_t = transpose weights_t_body
  let per_tok = map2 (\row g_row ->
    let x1 = row[0:half] :> [half]f16
    let x2 = row[half:half*2] :> [half]f16
    let pre_scale = map (\j ->
      weights_s[j][half] f16.+ f16.sum (map2 (f16.*) (weights_s[j][0:half] :> [half]f16) x2)
    ) (iota half)
    let scale = map (\ps -> f16.exp (f16.max clip_min (f16.min clip_max ps))) pre_scale
    let y1 = map2 (f16.*) x1 scale
    let dy1 = g_row[0:half] :> [half]f16
    let dy2 = g_row[half:half*2] :> [half]f16
    let dy1_total = map2 (\dy1_j wt_t_row ->
      dy1_j f16.+ f16.sum (map2 (f16.*) wt_t_row dy2)
    ) dy1 weights_t_t
    let ds = map2 (\ps j ->
      if ps f16.>= clip_min && ps f16.<= clip_max
      then dy1_total[j] f16.* y1[j]
      else f16.i32 0
    ) pre_scale (iota half)
    in (ds, x2, dy2, y1)
  ) input grad_output
  let ds_all  = map (\(a,_,_,_) -> a) per_tok
  let x2_all  = map (\(_,b,_,_) -> b) per_tok
  let dy2_all = map (\(_,_,c,_) -> c) per_tok
  let y1_all  = map (\(_,_,_,d) -> d) per_tok
  let ds_t  = transpose ds_all
  let x2_t  = transpose x2_all
  let dy2_t = transpose dy2_all
  let y1_t  = transpose y1_all
  let acc_ws = map2 (\ds_row bias ->
    let inner = map (\x2_row -> f16.sum (map2 (f16.*) ds_row x2_row)) x2_t
    in inner ++ [bias] :> [half+1]f16
  ) ds_t (map f16.sum ds_t)
  let acc_wt = map2 (\dy2_row bias ->
    let inner = map (\y1_row -> f16.sum (map2 (f16.*) dy2_row y1_row)) y1_t
    in inner ++ [bias] :> [half+1]f16
  ) dy2_t (map f16.sum dy2_t)
  in (acc_ws, acc_wt)

entry sfd_update_mat [d][e] (weights: *[d][e]f16) (gradients: [d][e]f16) (learning_rate: f16) (momentum: f16) (velocity: *[d][e]f16) : (*[d][e]f16, *[d][e]f16) =
  let new_velocity = map2 (map2 (\v g ->
    let vf = f32.f16 v
    let gf = f32.f16 g
    let safe_v = if f32.isnan vf || f32.isinf vf then 0f32 else vf
    let safe_g = if f32.isnan gf || f32.isinf gf then 0f32 else f32.max (-1f32) (f32.min 1f32 gf)
    let next = f32.f16 momentum * safe_v + f32.f16 learning_rate * safe_g
    in f16.f32 (f32.max (-65504f32) (f32.min 65504f32 next))
  )) velocity gradients
  let new_weights = map2 (map2 (\w v ->
    let wf = f32.f16 w
    let vf = f32.f16 v
    let safe_w = if f32.isnan wf || f32.isinf wf then 0f32 else wf
    let safe_v = if f32.isnan vf || f32.isinf vf then 0f32 else vf
    in f16.f32 (f32.max (-60000f32) (f32.min 60000f32 (safe_w - safe_v)))
  )) weights (copy new_velocity)
  in (new_weights, new_velocity)

entry compute_loss [n][d] (output: [n][d]f16) (target: [n][d]f16) : f16 =
  let squared_diff = map2 (map2 (\o t -> (o f16.- t) f16.* (o f16.- t))) output target
  let total = f16.sum (flatten squared_diff)
  let count = f16.i64 (n * d)
  in total f16./ count

entry batch_forward [batch_size][seq_len][half] (inputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16) : *[batch_size][seq_len][half*2]f16 =
  map (\sample -> rsf_forward sample weights_s weights_t clip_min clip_max) inputs

entry batch_compute_loss [batch_size][seq_len][d] (outputs: [batch_size][seq_len][d]f16) (targets: [batch_size][seq_len][d]f16) : f16 =
  let squared_diff_f32 = map2 (map2 (map2 (\o t ->
    let diff = (f32.f16 o) - (f32.f16 t)
    in diff * diff
  ))) outputs targets
  let total_f32 = f32.sum (flatten (flatten squared_diff_f32))
  let count_f32 = f32.i64 (batch_size * seq_len * d)
  let mean_f32 = total_f32 / count_f32
  in f16.f32 mean_f32

entry batch_gradients [batch_size][seq_len][half] (inputs: [batch_size][seq_len][half*2]f16)
  (grad_outputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : ([half][half+1]f16, [half][half+1]f16) =
  let results = map2 (\inp g_out ->
    rsf_backward inp g_out weights_s weights_t clip_min clip_max
  ) inputs grad_outputs
  let gs_list = map (\(gs, _) -> gs) results
  let gt_list = map (\(_, gt) -> gt) results
  let gs_total = reduce (map2 (map2 (f16.+))) (replicate half (replicate (half+1) (f16.i32 0))) gs_list
  let gt_total = reduce (map2 (map2 (f16.+))) (replicate half (replicate (half+1) (f16.i32 0))) gt_list
  in (copy gs_total, copy gt_total)

let rsf_backward_full [n][half]
  (input: [n][half*2]f16) (grad_output: [n][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : ([half][half+1]f16, [half][half+1]f16, *[n][half*2]f16) =
  let ws_body = map (\row -> row[0:half] :> [half]f16) weights_s
  let ws_bias = map (\row -> row[half]) weights_s
  let wt_body = map (\row -> row[0:half] :> [half]f16) weights_t
  let x1_all  = map (\row -> row[0:half]      :> [half]f16) input
  let x2_all  = map (\row -> row[half:half*2] :> [half]f16) input
  let dy1_all = map (\row -> row[0:half]      :> [half]f16) grad_output
  let dy2_all = map (\row -> row[half:half*2] :> [half]f16) grad_output
  let ws_body_t     = transpose ws_body
  let x2_ws_t       = matmul_f16 x2_all ws_body_t
  let pre_scale_all = map (\row -> map2 (f16.+) row ws_bias) x2_ws_t
  let scale_all     = map (map (\ps -> f16.exp (f16.max clip_min (f16.min clip_max ps)))) pre_scale_all
  let y1_all        = map2 (map2 (f16.*)) x1_all scale_all
  let dy2_wt        = matmul_f16 dy2_all wt_body
  let dy1_total_all = map2 (map2 (f16.+)) dy1_all dy2_wt
  let ds_all = map2 (map2 (\ps_val prod ->
    if ps_val f16.>= clip_min && ps_val f16.<= clip_max then prod else f16.i32 0
  )) pre_scale_all (map2 (map2 (f16.*)) dy1_total_all y1_all)
  let dx1_all  = map2 (map2 (f16.*)) dy1_total_all scale_all
  let ds_ws    = matmul_f16 ds_all ws_body
  let dx2_all  = map2 (map2 (f16.+)) dy2_all ds_ws
  let g_in     = map2 (\r1 r2 -> r1 ++ r2 :> [half*2]f16) dx1_all dx2_all
  let ds_t     = transpose ds_all
  let dy2_t    = transpose dy2_all
  let acc_ws_body = matmul_f16 ds_t x2_all
  let acc_ws_bias = map f16.sum ds_t
  let acc_ws      = map2 (\row bias -> row ++ [bias] :> [half+1]f16) acc_ws_body acc_ws_bias
  let acc_wt_body = matmul_f16 dy2_t y1_all
  let acc_wt_bias = map f16.sum dy2_t
  let acc_wt      = map2 (\row bias -> row ++ [bias] :> [half+1]f16) acc_wt_body acc_wt_bias
  in (copy acc_ws, copy acc_wt, copy g_in)

entry batch_rsf_inverse [batch_size][seq_len][half]
  (outputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : *[batch_size][seq_len][half*2]f16 =
  let flat_outputs = flatten outputs
  let ws_body   = map (\row -> row[0:half] :> [half]f16) weights_s
  let ws_bias   = map (\row -> row[half]) weights_s
  let wt_body   = map (\row -> row[0:half] :> [half]f16) weights_t
  let wt_bias   = map (\row -> row[half]) weights_t
  let wt_body_t = transpose wt_body
  let ws_body_t = transpose ws_body
  let y1_all       = map (\row -> row[0:half]      :> [half]f16) flat_outputs
  let y2_all       = map (\row -> row[half:half*2] :> [half]f16) flat_outputs
  let y1_wt_t      = matmul_f16 y1_all wt_body_t
  let trans_all    = map (\row -> map2 (f16.+) row wt_bias) y1_wt_t
  let x2_all       = map2 (map2 (f16.-)) y2_all trans_all
  let x2_ws_t      = matmul_f16 x2_all ws_body_t
  let pre_scale_all = map (\row -> map2 (f16.+) row ws_bias) x2_ws_t
  let scale_all    = map (map (\ps -> f16.exp (f16.max clip_min (f16.min clip_max ps)))) pre_scale_all
  let x1_all       = map2 (map2 (f16./)) y1_all scale_all
  let flat_inputs  = map2 (\r1 r2 -> r1 ++ r2 :> [half*2]f16) x1_all x2_all
  in copy (unflatten flat_inputs :> [batch_size][seq_len][half*2]f16)

entry batch_gradients_full [batch_size][seq_len][half]
  (inputs: [batch_size][seq_len][half*2]f16)
  (grad_outputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : ([half][half+1]f16, [half][half+1]f16, *[batch_size][seq_len][half*2]f16) =
  let flat_inputs = flatten inputs
  let flat_grads  = flatten grad_outputs
  let (gs, gt, flat_g_in) = rsf_backward_full flat_inputs flat_grads weights_s weights_t clip_min clip_max
  let g_in = unflatten flat_g_in :> [batch_size][seq_len][half*2]f16
  in (copy gs, copy gt, copy g_in)

entry compute_initial_grad_l2 [batch_size][seq_len][d]
  (outputs: [batch_size][seq_len][d]f16) (targets: [batch_size][seq_len][d]f16)
  : *[batch_size][seq_len][d]f16 =
  map2 (map2 (map2 (\o t -> (f16.f32 2.0) f16.* (o f16.- t)))) outputs targets

entry batch_compute_loss_masked [batch_size][seq_len][d]
  (outputs: [batch_size][seq_len][d]f16)
  (targets: [batch_size][seq_len][d]f16)
  (lengths: [batch_size]i64) : f16 =
  let squared_diff_f32 = map2 (\length bi ->
    let sample = outputs[bi]
    let target = targets[bi]
    in map (\j ->
      let active = j < i64.max 0 (i64.min seq_len length)
      in map2 (\o t ->
        if active then
          let diff = f32.f16 o - f32.f16 t
           in if f32.isnan diff || f32.isinf diff then 0f32 else diff * diff
        else 0f32
      ) sample[j] target[j]
    ) (iota seq_len)
  ) lengths (iota batch_size)
  let valid_tokens = i64.sum (map (\length -> i64.max 0 (i64.min seq_len length)) lengths)
  let count = valid_tokens * d
  let total = f32.sum (flatten (flatten squared_diff_f32))
  let safe_total = if f32.isnan total || f32.isinf total then 0f32 else total
  in if count <= 0
     then f16.i32 0
     else f16.f32 (safe_total / f32.i64 count)

entry compute_initial_grad_l2_masked [batch_size][seq_len][d]
  (outputs: [batch_size][seq_len][d]f16)
  (targets: [batch_size][seq_len][d]f16)
  (lengths: [batch_size]i64) : *[batch_size][seq_len][d]f16 =
  map2 (\length bi ->
    let sample = outputs[bi]
    let target = targets[bi]
    in map (\j ->
      let active = j < i64.max 0 (i64.min seq_len length)
      in if active
          then map2 (\o t ->
            let diff = f32.f16 o - f32.f16 t
            let safe_diff = if f32.isnan diff || f32.isinf diff then 0f32 else f32.max (-100f32) (f32.min 100f32 diff)
            in f16.f32 (2f32 * safe_diff)
          ) sample[j] target[j]
         else replicate d (f16.i32 0)
    ) (iota seq_len)
  ) lengths (iota batch_size)

entry xavier_fill_inplace [d] (_weights: *[d][d]f16) (seed: i32) : *[d][d]f16 =
  let scale = f16.sqrt (f16.f32 2.0 f16./ f16.i64 d)
  in map (\i ->
    map (\j ->
      let hash = (seed + i32.i64 i * 73856093 + i32.i64 j * 19349663) % 1000000
      let normalized = (f16.i32 hash) f16./ (f16.i32 1000000) f16.- f16.f32 0.5
      in normalized f16.* scale
    ) (iota d)
  ) (iota d)

entry scale_weights_inplace [d] (weights: *[d][d]f16) (scale_factor: f16) : *[d][d]f16 =
  map (map (\w -> w f16./ scale_factor)) weights

entry scale_matrix_f16 [rows][columns] (values: *[rows][columns]f16) (scale_factor: f16) : *[rows][columns]f16 =
  map (map (\value -> value f16.* scale_factor)) values

entry accumulate_gradients [d] (grad1: *[d][d]f16) (grad2: [d][d]f16) : *[d][d]f16 =
  map2 (map2 (f16.+)) grad1 grad2

entry training_step [batch_size][seq_len][half]
  (inputs: [batch_size][seq_len][half*2]f16)
  (targets: [batch_size][seq_len][half*2]f16)
  (weights_s: *[half][half+1]f16)
  (weights_t: *[half][half+1]f16)
  (velocity_s: *[half][half+1]f16)
  (velocity_t: *[half][half+1]f16)
  (learning_rate: f16)
  (momentum: f16)
  (clip_min: f16)
  (clip_max: f16) : (*[half][half+1]f16, *[half][half+1]f16, *[half][half+1]f16, *[half][half+1]f16, f16) =
  let outputs = batch_forward inputs weights_s weights_t clip_min clip_max
  let loss = batch_compute_loss outputs targets
  let grad_outputs = map2 (map2 (map2 (\o t -> (f16.f32 2.0) f16.* (o f16.- t)))) outputs targets
  let (grad_s, grad_t) = batch_gradients inputs grad_outputs weights_s weights_t clip_min clip_max
  let grad_s_c = copy grad_s
  let grad_t_c = copy grad_t
  let (new_weights_s, new_velocity_s) = sfd_update_mat weights_s grad_s_c learning_rate momentum velocity_s
  let (new_weights_t, new_velocity_t) = sfd_update_mat weights_t grad_t_c learning_rate momentum velocity_t
  in (new_weights_s, new_weights_t, new_velocity_s, new_velocity_t, loss)

let oftb_scale : f16 = f16.f32 0.7071067811865476

entry oftb_forward_single [seq_len][dim] (input: [seq_len][dim]f16) : *[seq_len][dim]f16 =
  let half = dim / 2
  in map (\row ->
    let x1 = map f32.f16 (row[0:half] :> [half]f16)
    let x2 = map f32.f16 (row[half:dim] :> [half]f16)
    let new_x1 = map2 (\a b -> (a - b) * f32.f16 oftb_scale) x1 x2
    let new_x2 = map2 (\a b -> (a + b) * f32.f16 oftb_scale) x1 x2
    let output = map (\value ->
      let safe_value = if f32.isnan value || f32.isinf value then 0f32 else f32.max (-60000f32) (f32.min 60000f32 value)
      in f16.f32 safe_value
    ) (new_x1 ++ new_x2)
    in output :> [dim]f16
  ) input

entry oftb_backward_single [seq_len][dim] (grad_output: [seq_len][dim]f16) : *[seq_len][dim]f16 =
  let half = dim / 2
  in map (\row ->
    let g1 = map f32.f16 (row[0:half] :> [half]f16)
    let g2 = map f32.f16 (row[half:dim] :> [half]f16)
    let new_g1 = map2 (\a b -> (a + b) * f32.f16 oftb_scale) g1 g2
    let new_g2 = map2 (\a b -> (b - a) * f32.f16 oftb_scale) g1 g2
    let output = map (\value ->
      let safe_value = if f32.isnan value || f32.isinf value then 0f32 else f32.max (-100f32) (f32.min 100f32 value)
      in f16.f32 safe_value
    ) (new_g1 ++ new_g2)
    in output :> [dim]f16
  ) grad_output

entry oftb_forward [batch_size][seq_len][dim] (inputs: [batch_size][seq_len][dim]f16) : *[batch_size][seq_len][dim]f16 =
  map (\sample -> oftb_forward_single sample) inputs

entry oftb_backward [batch_size][seq_len][dim] (grad_outputs: [batch_size][seq_len][dim]f16) : *[batch_size][seq_len][dim]f16 =
  map (\sample -> oftb_backward_single sample) grad_outputs

entry batch_oftb_forward [batch_size][seq_len][dim] (inputs: [batch_size][seq_len][dim]f16) : *[batch_size][seq_len][dim]f16 =
  oftb_forward inputs

entry batch_oftb_backward [batch_size][seq_len][dim] (grad_outputs: [batch_size][seq_len][dim]f16) : *[batch_size][seq_len][dim]f16 =
  oftb_backward grad_outputs

entry embedding_forward [n][vocab_size][dim] (tokens: [n]i64) (weight: [vocab_size][dim]f16) : *[n][dim]f16 =
  map (\tok ->
    let t = if tok >= 0 && tok < vocab_size then tok else 0
    in weight[t]
  ) tokens

entry embedding_forward_padded [n][batch_size][seq_len][vocab_size][dim]
  (tokens: [n]i64)
  (lengths: [batch_size]i64)
  (positions: [seq_len]i64)
  (weight: [vocab_size][dim]f16) : *[batch_size][seq_len][dim]f16 =
  map2 (\batch_index length ->
    map (\sequence_index ->
      let flat_index = batch_index * seq_len + sequence_index
      in if sequence_index < i64.max 0 (i64.min seq_len length) && flat_index < n
         then let token = tokens[flat_index]
              let safe_token = if token >= 0 && token < vocab_size then token else 0
              in weight[safe_token]
         else replicate dim (f16.i32 0)
    ) positions
  ) (iota batch_size) lengths

entry embedding_backward [n][vocab_size][dim] (tokens: [n]i64) (grad_output: [n][dim]f16) (grad_weight: [vocab_size][dim]f16) : *[vocab_size][dim]f16 =
  loop gw = copy grad_weight for i < n do
    let t = if tokens[i] >= 0 && tokens[i] < vocab_size then tokens[i] else 0
    let row_update = map2 (\g acc -> acc f16.+ g) grad_output[i] gw[t]
    in gw with [t] = row_update

entry embedding_backward_padded [n][batch_size][seq_len][dim][vocab_size]
  (tokens: [n]i64)
  (lengths: [batch_size]i64)
  (grad_output: [batch_size][seq_len][dim]f16)
  (grad_weight: [vocab_size][dim]f16)
  (clip_norm: f32) : *[vocab_size][dim]f16 =
  let flat_size = batch_size * seq_len
  let masked_gradient_f32 = map2 (\length bi ->
    map (\j ->
      let active = j < i64.max 0 (i64.min seq_len length)
      in if active
         then map f32.f16 grad_output[bi][j]
         else replicate dim 0f32
    ) (iota seq_len)
  ) lengths (iota batch_size)
  let flat_values = flatten (flatten masked_gradient_f32)
  let maximum_absolute_value = reduce f32.max 0f32 (map f32.abs flat_values)
  let scaled_norm_squared =
    if maximum_absolute_value > 0f32
    then f32.sum (map (\value ->
      let scaled = value / maximum_absolute_value
      in scaled * scaled
    ) flat_values)
    else 0f32
  let norm = maximum_absolute_value * f32.sqrt scaled_norm_squared
  let gradient_scale =
    if clip_norm > 0f32 && norm > clip_norm && norm > 1e-12f32
    then clip_norm / norm
    else 1f32
  let flat_gradient = map (\bi ->
    map (\j ->
      map (\v -> f16.f32 (v * gradient_scale)) masked_gradient_f32[bi][j]
    ) (iota seq_len)
  ) (iota batch_size)
  in loop gw = copy grad_weight for flat_index < flat_size do
    let bi = flat_index / seq_len
    let j = flat_index % seq_len
    let length = lengths[bi]
    let valid_position = j < i64.max 0 (i64.min seq_len length) && flat_index < n
    in if valid_position
       then let token = tokens[flat_index]
            in if token >= 0 && token < vocab_size
               then let row_update = map2 (\gradient accumulated -> accumulated f16.+ gradient) flat_gradient[bi][j] gw[token]
                    in gw with [token] = row_update
               else gw
       else gw

entry embedding_update [vocab_size][dim] (weight: *[vocab_size][dim]f16) (grad_weight: [vocab_size][dim]f16) (lr: f16) : *[vocab_size][dim]f16 =
  map2 (map2 (\w g -> w f16.- lr f16.* g)) weight grad_weight

entry embedding_spectral_normalize [vocab_size][dim]
  (weight: *[vocab_size][dim]f16)
  (u: *[vocab_size]f32)
  (v: *[dim]f32)
  (power_iters: i64) : (*[vocab_size][dim]f16, *[vocab_size]f32, *[dim]f32) =
  let wf32 : [vocab_size][dim]f32 = map (map f32.f16) weight
  let (final_u, final_v) =
    loop (ua, va) = (u, v) for loop_k < power_iters do
      let _ = loop_k
      let _ = va
      let raw_v = map (\j -> f32.sum (map (\i -> wf32[i][j] * ua[i]) (iota vocab_size))) (iota dim)
      let v_norm = f32.sqrt (f32.sum (map (\x -> x * x) raw_v))
      let v_safe = if v_norm < 1e-12f32 then 1.0f32 else v_norm
      let nv = map (/ v_safe) raw_v
      let raw_u = map (\row -> f32.sum (map2 (*) row nv)) wf32
      let u_norm = f32.sqrt (f32.sum (map (\x -> x * x) raw_u))
      let u_safe = if u_norm < 1e-12f32 then 1.0f32 else u_norm
      let nu = map (/ u_safe) raw_u
      let nv_ret = copy nv
      in (nu, nv_ret)
  let sigma = f32.sum (map2 (\ui row -> ui * f32.sum (map2 (*) row final_v)) final_u wf32)
  let out_weight =
    if sigma > 1.0f32
    then map (map (\w -> f16.f32 (w / sigma))) wf32
    else map (map f16.f32) wf32
  in (out_weight, copy final_u, copy final_v)

let graph_derive_qubit_states [n] (hashes: [n]u64) : ([n]f32, [n]f32, [n]f32, [n]f32) =
  let pi = 3.14159265358979323846f32
  let inv_m = 1f32 / 1000000f32
  let raw_re_a = map (\h -> f32.cos (pi * f32.u64 (h % 1000000u64) * inv_m)) hashes
  let raw_im_a = map (\h -> f32.sin (pi * f32.u64 ((h >> 20u64) % 1000000u64) * inv_m)) hashes
  let raw_re_b = map (\h -> f32.cos (pi * 2f32 * f32.u64 ((h >> 40u64) % 1000000u64) * inv_m)) hashes
  let raw_im_b = map (\h -> f32.sin (pi * 2f32 * f32.u64 ((h >> 32u64) % 1000000u64) * inv_m)) hashes
  let norms = map4 (\ra ia rb ib ->
    let s = ra * ra + ia * ia + rb * rb + ib * ib
    in if s > 1e-30f32 then f32.sqrt s else 1f32
  ) raw_re_a raw_im_a raw_re_b raw_im_b
  in (map2 (/) raw_re_a norms, map2 (/) raw_im_a norms, map2 (/) raw_re_b norms, map2 (/) raw_im_b norms)

entry graph_batch_encode [n] (data_hashes: [n]u64) (_seed: u64) : ([]u64, []f32, []f32, []f32, []f32, []i64, []i64) =
  let (re_a, im_a, re_b, im_b) = graph_derive_qubit_states data_hashes
  let ne = n * 3
  let edge_srcs = tabulate ne (\flat_i ->
    let node_i = flat_i / 3
    let pred_k = flat_i % 3
    in if node_i > pred_k then node_i else -1i64
  )
  let edge_tgts = tabulate ne (\flat_i ->
    let node_i = flat_i / 3
    let pred_k = flat_i % 3
    in if node_i > pred_k then node_i - pred_k - 1 else -1i64
  )
  in (copy data_hashes, re_a, im_a, re_b, im_b, edge_srcs, edge_tgts)

entry batch_add_reconstruction_delta_masked [batch_size][seq_len][d]
  (forward_delta: [batch_size][seq_len][d]f16)
  (reconstructed: [batch_size][seq_len][d]f16)
  (original: [batch_size][seq_len][d]f16)
  (lengths: [batch_size]i64)
  (alpha: f16)
  (forward_scale: f16)
  : *[batch_size][seq_len][d]f16 =
  let valid_tokens = i64.sum (map (\length -> i64.max 0 (i64.min seq_len length)) lengths)
  let count = if valid_tokens > 0 then valid_tokens * d else 1
  let count_f32 = f32.i64 count
  let alpha_f32 = f32.f16 alpha
  let forward_scale_f32 = f32.f16 forward_scale
  in map2 (\length bi ->
    let fd = forward_delta[bi]
    let rc = reconstructed[bi]
    let og = original[bi]
    let limit = i64.max 0 (i64.min seq_len length)
    in map (\j ->
      let active = j < limit
      in map3 (\f r o ->
        let f_f32 = f32.f16 f
        let safe_f = if f32.isnan f_f32 || f32.isinf f_f32 then 0f32 else f_f32
        in if active
           then
             let diff = f32.f16 r - f32.f16 o
             let safe_diff = if f32.isnan diff || f32.isinf diff
                             then 0f32
                             else f32.max (-100f32) (f32.min 100f32 diff)
             let combined = forward_scale_f32 * safe_f
                            + alpha_f32 * 2f32 * safe_diff / count_f32
             let bounded = f32.max (-65504f32) (f32.min 65504f32 combined)
             in f16.f32 bounded
           else
             let scaled = forward_scale_f32 * safe_f
             let bounded = f32.max (-65504f32) (f32.min 65504f32 scaled)
             in f16.f32 bounded
      ) fd[j] rc[j] og[j]
    ) (iota seq_len)
  ) lengths (iota batch_size)

entry batch_compute_reconstruction_loss_masked [batch_size][seq_len][d]
  (reconstructed: [batch_size][seq_len][d]f16)
  (original: [batch_size][seq_len][d]f16)
  (lengths: [batch_size]i64)
  : f16 =
  let squared_diff_f32 = map2 (\length bi ->
    let rc = reconstructed[bi]
    let og = original[bi]
    let limit = i64.max 0 (i64.min seq_len length)
    in map (\j ->
      let active = j < limit
      in map2 (\r o ->
        if active
        then
          let diff = f32.f16 r - f32.f16 o
          in if f32.isnan diff || f32.isinf diff then 0f32 else diff * diff
        else 0f32
      ) rc[j] og[j]
    ) (iota seq_len)
  ) lengths (iota batch_size)
  let valid_tokens = i64.sum (map (\length -> i64.max 0 (i64.min seq_len length)) lengths)
  let count = valid_tokens * d
  let total = f32.sum (flatten (flatten squared_diff_f32))
  let safe_total = if f32.isnan total || f32.isinf total then 0f32 else total
  in if count <= 0
     then f16.i32 0
     else f16.f32 (safe_total / f32.i64 count)

entry embedding_sum_squares [vocab_size][dim] (source: [vocab_size][dim]f16) : f32 =
  let squared = map (\row ->
    map (\v ->
      let x = f32.f16 v
      in if f32.isnan x || f32.isinf x then 0f32 else x * x
    ) row
  ) source
  let total = f32.sum (flatten squared)
  in if f32.isnan total || f32.isinf total then 0f32 else total
