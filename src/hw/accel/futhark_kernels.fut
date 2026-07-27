import "lib/github.com/diku-dk/sorts/radix_sort"

let float_pow (base: f32) (exp_val: i64): f32 =
  loop result = 1f32 for _i < (i64.max 0 exp_val) do result * base

let f32_total_order (x: f32): u32 =
  let bits = f32.to_bits x
  in if (bits >> 31) != 0u32
     then bits ^ 0xFFFFFFFFu32
     else bits ^ 0x80000000u32

let matmul_tiled [m][n][k] (a: [m][k]f32) (b: [k][n]f32): [m][n]f32 =
  let bt = transpose b
  in map (\a_row ->
    map (\b_col ->
      reduce (+) 0f32 (map2 (*) a_row b_col)
    ) bt
  ) a

let batched_matmul [batch][m][n][k] (a: [batch][m][k]f32) (c: [batch][k][n]f32): [batch][m][n]f32 =
  map2 (\a_mat c_mat -> matmul_tiled a_mat c_mat) a c

let dot_product [n] (a: [n]f32) (b: [n]f32): f32 =
  let products = map2 (\x y ->
    let p = x * y
    in if f32.isnan p then 0f32 else p
  ) a b
  let result = reduce (+) 0f32 products
  in if f32.isnan result then 0f32 else result

let spectral_clip [n] (fisher: [n]f32) (clip_val: f32): [n]f32 =
  let safe_clip = f32.max clip_val 0f32
  in map (\f ->
    if f32.isnan f then safe_clip else f32.max f safe_clip
  ) fisher

let batch_reduce [b][n] (gradients: [b][n]f32): [n]f32 =
  if b == 0 then replicate n 0f32
  else
    let gt = transpose gradients
    in map (\col -> reduce (+) 0f32 col) gt

let fisher_diagonal_update [n] (fisher: [n]f32) (gradient: [n]f32) (decay: f32): [n]f32 =
  let safe_decay = f32.max 0f32 (f32.min 1f32 decay)
  in map2 (\f g ->
    let safe_f = if f32.isnan f then 0f32 else f
    let g_sq = g * g
    let safe_g_sq = if f32.isnan g_sq || f32.isinf g_sq then 0f32 else g_sq
    in safe_decay * safe_f + (1f32 - safe_decay) * safe_g_sq
  ) fisher gradient

let spectral_natural_gradient [n] (gradient: [n]f32) (fisher: [n]f32) (damping: f32): [n]f32 =
  let safe_damping = f32.max damping 1e-8f32
  in map2 (\g f ->
    let denom = f32.max (f32.abs f + safe_damping) 1e-8f32
    let result = g / denom
    in if f32.isnan result || f32.isinf result then 0f32 else result
  ) gradient fisher

let score_segments [n] (query_hash: u64) (segment_hashes: [n]u64) (base_scores: [n]f32): [n]f32 =
  map2 (\hash score ->
    let base = if f32.isnan score then 0f32 else score
    let match_bonus = if hash == query_hash then 1f32 else 0f32
    let result = base + match_bonus
    in f32.max (-1e30f32) (f32.min 1e30f32 result)
  ) segment_hashes base_scores

let topk [n] (k: i64) (scores: [n]f32) (indices: [n]i64): ([k]f32, [k]i64) =
  let safe_k = i64.max 0 k
  let safe_scores = map (\s -> if f32.isnan s then -f32.inf else s) scores
  let pairs = zip safe_scores indices
  let sorted_asc = radix_sort_by_key (\(s, _) -> f32_total_order s) u32.num_bits u32.get_bit pairs
  let pad_len = i64.max n safe_k
  let desc_scores = tabulate pad_len (\i ->
    if i < n then let (s, _) = sorted_asc[n - 1 - i] in s else -f32.inf
  )
  let desc_indices = tabulate pad_len (\i ->
    if i < n then let (_, idx) = sorted_asc[n - 1 - i] in idx else -1i64
  )
  in (take safe_k desc_scores :> [k]f32, take safe_k desc_indices :> [k]i64)

let rsf_scatter [n] (x: [n]f32) (indices: [n]i64): [n]f32 =
  if n < 2 then copy x
  else
    let half = n / 2
    let inv_sqrt2 = 1f32 / f32.sqrt 2f32
    in tabulate n (\i ->
      if i < half then
        let j = (i64.abs indices[i]) % half
        in inv_sqrt2 * (x[j] + x[j + half])
      else if i < half * 2 then
        let j = (i64.abs indices[i]) % half
        in inv_sqrt2 * (x[j] - x[j + half])
      else x[i]
    )

let rsf_flow [half] (x: [half*2]f32) (s_weight: [half][half+1]f32) (t_weight: [half][half+1]f32) (clip_min: f32) (clip_max: f32): [half*2]f32 =
  let d = half * 2
  let x1 = x[0:half] :> [half]f32
  let x2 = x[half:d] :> [half]f32
  let scale = tabulate half (\j ->
    let raw = s_weight[j][half] + reduce (+) 0f32 (map2 (*) (s_weight[j][0:half] :> [half]f32) x2)
    let clipped = f32.max clip_min (f32.min clip_max raw)
    in f32.exp clipped
  )
  let y1 = map2 (*) x1 scale
  let trans = tabulate half (\j ->
    t_weight[j][half] + reduce (+) 0f32 (map2 (*) (t_weight[j][0:half] :> [half]f32) y1)
  )
  let y2 = map2 (+) x2 trans
  in (y1 ++ y2) :> [half*2]f32

let rsf_forward_layer [half] (x: [half*2]f32) (s_weight: [half][half+1]f32) (t_weight: [half][half+1]f32) (perm_indices: [half*2]i64) (clip_min: f32) (clip_max: f32): [half*2]f32 =
  let scattered = rsf_scatter x perm_indices
  in rsf_flow scattered s_weight t_weight clip_min clip_max

let rsf_forward_multi [num_layers][half] (x: [half*2]f32) (s_ws: [num_layers][half][half+1]f32) (t_ws: [num_layers][half][half+1]f32) (perms: [num_layers][half*2]i64) (clip_min: f32) (clip_max: f32): [half*2]f32 =
  loop acc = x for i < num_layers do
    rsf_forward_layer acc s_ws[i] t_ws[i] perms[i] clip_min clip_max

let rsf_backward_scatter [n] (grad: [n]f32) (indices: [n]i64): [n]f32 =
  if n < 2 then copy grad
  else
    let half = n / 2
    let inv_sqrt2 = 1f32 / f32.sqrt 2f32
    let safe_idx = tabulate n (\i -> (i64.abs indices[i]) % half)
    let idx_first = tabulate half (\i -> safe_idx[i])
    let idx_second = tabulate half (\i -> safe_idx[i + half])
    let grad_first = tabulate half (\i -> inv_sqrt2 * grad[i])
    let grad_second = tabulate half (\i -> inv_sqrt2 * grad[i + half])
    let grad_lower = reduce_by_index (replicate half 0f32) (+) 0f32
      (idx_first ++ idx_second) (grad_first ++ grad_second)
    let grad_upper = reduce_by_index (replicate half 0f32) (+) 0f32
      (idx_first ++ idx_second)
      (grad_first ++ map (\g -> -g) grad_second)
    let base = grad_lower ++ grad_upper
    in tabulate n (\i ->
      if i < half * 2 then base[i] else grad[i]
    )

let rsf_backward_flow [half] (grad_out: [half*2]f32) (x: [half*2]f32) (s_weight: [half][half+1]f32) (t_weight: [half][half+1]f32) (clip_min: f32) (clip_max: f32): ([half*2]f32, [half][half+1]f32, [half][half+1]f32) =
  let d = half * 2
  let x1 = x[0:half] :> [half]f32
  let x2 = x[half:d] :> [half]f32
  let pre_scale = tabulate half (\j ->
    s_weight[j][half] + reduce (+) 0f32 (map2 (*) (s_weight[j][0:half] :> [half]f32) x2)
  )
  let scale = map (\ps ->
    let clipped = f32.max clip_min (f32.min clip_max ps)
    in f32.exp clipped
  ) pre_scale
  let y1 = map2 (*) x1 scale
  let dy1 = grad_out[0:half] :> [half]f32
  let dy2 = grad_out[half:d] :> [half]f32
  let dy1_total = tabulate half (\j ->
    dy1[j] + reduce (+) 0f32 (tabulate half (\k -> t_weight[k][j] * dy2[k]))
  )
  let ds = tabulate half (\j ->
    let in_range = pre_scale[j] >= clip_min && pre_scale[j] <= clip_max
    in if in_range then dy1_total[j] * y1[j] else 0f32
  )
  let dx1 = map2 (*) dy1_total scale
  let dx2 = tabulate half (\j ->
    dy2[j] + reduce (+) 0f32 (tabulate half (\k -> s_weight[k][j] * ds[k]))
  )
  let grad_x = (dx1 ++ dx2) :> [half*2]f32
  let grad_ws = tabulate half (\j ->
    tabulate (half+1) (\k -> if k < half then ds[j] * x2[k] else ds[j])
  )
  let grad_wt = tabulate half (\j ->
    tabulate (half+1) (\k -> if k < half then dy2[j] * y1[k] else dy2[j])
  )
  in (grad_x, grad_ws, grad_wt)

let rsf_backward_layer [half] (grad_out: [half*2]f32) (x: [half*2]f32) (s_weight: [half][half+1]f32) (t_weight: [half][half+1]f32) (perm_indices: [half*2]i64) (clip_min: f32) (clip_max: f32): ([half*2]f32, [half][half+1]f32, [half][half+1]f32) =
  let scattered_x = rsf_scatter x perm_indices
  let (grad_flow, grad_s_w, grad_t_w) = rsf_backward_flow grad_out scattered_x s_weight t_weight clip_min clip_max
  let grad_x = rsf_backward_scatter grad_flow perm_indices
  in (grad_x, grad_s_w, grad_t_w)

let hash_sequence [m] (tokens: [m]u32): u64 =
  loop h = 14695981039346656037u64 for i < m do
    (h ^ u64.u32 tokens[i]) * 1099511628211u64

let ssi_hash_insert [n] (hashes: [n]u64) (new_hash: u64): [n+1]u64 =
  let pos = reduce (+) 0i64 (map (\h -> if h < new_hash then 1i64 else 0i64) hashes)
  in tabulate (n + 1) (\i ->
    if i < pos then hashes[i]
    else if i == pos then new_hash
    else hashes[i - 1]
  )

let ssi_search [n][m] (tree_hashes: [n]u64) (query: [m]u32): i64 =
  if n == 0 then -1i64
  else
    let query_hash = hash_sequence query
    let (_, best_idx) = reduce (\(d1, i1) (d2, i2) ->
      if d1 < d2 then (d1, i1)
      else if d2 < d1 then (d2, i2)
      else (d1, i64.min i1 i2)
    ) (u64.highest, -1i64)
      (map2 (\h i ->
        let diff = if h > query_hash then h - query_hash else query_hash - h
        in (diff, i)
      ) tree_hashes (iota n))
    in best_idx

let ssi_retrieve_topk [n][m] (tree_hashes: [n]u64) (scores: [n]f32) (query: [m]u32) (k: i64): ([k]u64, [k]f32) =
  let safe_k = i64.max 0 k
  in if n == 0 then (replicate safe_k 0u64 :> [k]u64, replicate safe_k 0f32 :> [k]f32)
  else
    let query_hash = hash_sequence query
    let adjusted = map2 (\h score ->
      let safe_score = if f32.isnan score then 0f32 else score
      let match_bonus = if h == query_hash then 10f32 else 0f32
      let diff = if h > query_hash then h - query_hash else query_hash - h
      let proximity = 1f32 / (1f32 + f32.u64 (u64.min diff 1000000u64))
      in safe_score + match_bonus + proximity
    ) tree_hashes scores
    let safe_adj = map (\s -> if f32.isnan s then -f32.inf else s) adjusted
    let pairs = zip safe_adj (iota n)
    let sorted = radix_sort_by_key (\(s, _) -> f32_total_order s) u32.num_bits u32.get_bit pairs
    let pad_len = i64.max n safe_k
    let desc_hashes = tabulate pad_len (\i ->
      if i < n then let (_, idx) = sorted[n - 1 - i] in tree_hashes[idx] else 0u64
    )
    let desc_scores = tabulate pad_len (\i ->
      if i < n then let (s, _) = sorted[n - 1 - i] in s else -f32.inf
    )
    in (take safe_k desc_hashes :> [k]u64, take safe_k desc_scores :> [k]f32)

let ssi_compute_similarity [m] (query: [m]u32) (candidate: [m]u32): f32 =
  if m == 0 then 0f32
  else
    let matches = reduce (+) 0i64 (map2 (\q c -> if q == c then 1i64 else 0i64) query candidate)
    in f32.i64 matches / f32.i64 m

let ngram_hash [n] (tokens: [n]u32) (ngram_size: i64): []u64 =
  let safe_size = i64.max 1 (i64.min ngram_size n)
  let num_ngrams = i64.max 0 (n - safe_size + 1)
  in tabulate num_ngrams (\i ->
    hash_sequence tokens[i:i+safe_size]
  )

let lsh_hash [n] (vec: [n]f32) (num_tables: i64) (seed: u64): [num_tables]u64 =
  if n == 0 then replicate num_tables 0u64
  else
    let inv_n = f32.sqrt (1f32 / f32.i64 (i64.max 1 n))
    in tabulate num_tables (\table_idx ->
      let table_seed = seed + u64.i64 table_idx
      let proj = reduce (+) 0f32 (map2 (\v i ->
        let hash_raw = (table_seed * 2654435761u64 + u64.i64 i * 2246822519u64) % 1000003u64
        let centered = f32.u64 hash_raw / 500001.5f32 - 1f32
        in v * centered * inv_n
      ) vec (iota n))
      in if proj > 0f32 then 1u64 else 0u64
    )

let rsf_relational_context [seq_len][d_model] (spectral_input: [seq_len][d_model]f32) (_temporal_input: [seq_len][d_model]f32) (value: [seq_len][d_model]f32) (s_weight: [d_model]f32) (t_weight: [d_model]f32) (_eps: f32): [seq_len][d_model]f32 =
  if seq_len == 0 || d_model == 0 then copy value
  else
    let half = d_model / 2
    let clip_min = -5.0f32
    let clip_max = 5.0f32
    in map (\s_row ->
      let x1 = take half s_row :> [half]f32
      let x2 = drop half s_row :> [half]f32
      let s_w1 = take half s_weight :> [half]f32
      let t_w1 = take half t_weight :> [half]f32
      let scale = map2 (\xi wi ->
        let raw = xi * wi
        let clipped = f32.max clip_min (f32.min clip_max raw)
        in f32.exp clipped
      ) x2 s_w1
      let y1 = map2 (*) x1 scale
      let trans = map2 (*) y1 t_w1
      let y2 = map2 (+) x2 trans
      in (y1 ++ y2) :> [d_model]f32
    ) spectral_input

let elem_add [n] (a: [n]f32) (b: [n]f32): [n]f32 = map2 (+) a b
let elem_mul [n] (a: [n]f32) (b: [n]f32): [n]f32 = map2 (*) a b
let elem_div [n] (a: [n]f32) (b: [n]f32): [n]f32 =
  map2 (\x y -> if y == 0f32 || f32.isnan y then 0f32 else x / y) a b
let elem_sub [n] (a: [n]f32) (b: [n]f32): [n]f32 = map2 (-) a b

let scalar_add [n] (a: [n]f32) (s: f32): [n]f32 = map (+ s) a
let scalar_mul [n] (a: [n]f32) (s: f32): [n]f32 = map (* s) a
let scalar_div [n] (a: [n]f32) (s: f32): [n]f32 =
  if s == 0f32 || f32.isnan s then replicate n 0f32 else map (/ s) a

let sum [n] (x: [n]f32): f32 = reduce (+) 0f32 x
let mean [n] (x: [n]f32): f32 =
  if n == 0 then 0f32 else (reduce (+) 0f32 x) / f32.i64 n
let max [n] (x: [n]f32): f32 =
  if n == 0 then 0f32 else reduce f32.max (-f32.inf) x
let min [n] (x: [n]f32): f32 =
  if n == 0 then 0f32 else reduce f32.min f32.inf x

entry matmul [m][n][k] (a: [m][k]f32) (b: [k][n]f32): [m][n]f32 = matmul_tiled a b
entry batch_matmul [b][m][n][k] (a: [b][m][k]f32) (c: [b][k][n]f32): [b][m][n]f32 = batched_matmul a c
entry dot [n] (a: [n]f32) (b: [n]f32): f32 = dot_product a b

entry clip_fisher [n] (fisher: [n]f32) (clip_val: f32): [n]f32 = spectral_clip fisher clip_val
entry reduce_gradients [b][n] (gradients: [b][n]f32): [n]f32 = batch_reduce gradients
entry update_fisher [n] (fisher: [n]f32) (grad: [n]f32) (decay: f32): [n]f32 = fisher_diagonal_update fisher grad decay
entry compute_natural_grad [n] (grad: [n]f32) (fisher: [n]f32) (damping: f32): [n]f32 = spectral_natural_gradient grad fisher damping

entry rank_segments [n] (query_hash: u64) (segment_hashes: [n]u64) (base_scores: [n]f32): [n]f32 = score_segments query_hash segment_hashes base_scores
entry select_topk [n] (k: i64) (scores: [n]f32): ([]f32, []i64) =
  let safe_k = i64.max 0 k
  in topk safe_k scores (iota n)

entry rsf_forward [half] (x: [half*2]f32) (s_w: [half][half+1]f32) (t_w: [half][half+1]f32) (perm: [half*2]i64) (clip_min: f32) (clip_max: f32): [half*2]f32 = rsf_forward_layer x s_w t_w perm clip_min clip_max
entry rsf_forward_multilayer [num_layers][half] (x: [half*2]f32) (s_ws: [num_layers][half][half+1]f32) (t_ws: [num_layers][half][half+1]f32) (perms: [num_layers][half*2]i64) (clip_min: f32) (clip_max: f32): [half*2]f32 = rsf_forward_multi x s_ws t_ws perms clip_min clip_max
entry rsf_backward [half] (grad: [half*2]f32) (x: [half*2]f32) (s_w: [half][half+1]f32) (t_w: [half][half+1]f32) (perm: [half*2]i64) (clip_min: f32) (clip_max: f32): ([half*2]f32, [half][half+1]f32, [half][half+1]f32) = rsf_backward_layer grad x s_w t_w perm clip_min clip_max

entry ssi_hash_tokens [m] (tokens: [m]u32): u64 = hash_sequence tokens
entry ssi_find_nearest [n][m] (tree: [n]u64) (query: [m]u32): i64 = ssi_search tree query
entry ssi_get_topk [n][m] (tree: [n]u64) (scores: [n]f32) (query: [m]u32) (k: i64): ([]u64, []f32) =
  let safe_k = i64.max 0 k
  in ssi_retrieve_topk tree scores query safe_k
entry ssi_similarity [m] (query: [m]u32) (candidate: [m]u32): f32 = ssi_compute_similarity query candidate

entry compute_ngram_hashes [n] (tokens: [n]u32) (ngram_size: i64): []u64 = ngram_hash tokens ngram_size
entry compute_lsh [n] (vec: [n]f32) (num_tables: i64) (seed: u64): [num_tables]u64 = lsh_hash vec num_tables seed

entry compute_rsf_context [seq_len][d_model] (spectral_input: [seq_len][d_model]f32) (temporal_input: [seq_len][d_model]f32) (value: [seq_len][d_model]f32) (s_weight: [d_model]f32) (t_weight: [d_model]f32) (eps: f32): [seq_len][d_model]f32 = rsf_relational_context spectral_input temporal_input value s_weight t_weight eps

entry add_arrays [n] (a: [n]f32) (b: [n]f32): [n]f32 = elem_add a b
entry mul_arrays [n] (a: [n]f32) (b: [n]f32): [n]f32 = elem_mul a b
entry div_arrays [n] (a: [n]f32) (b: [n]f32): [n]f32 = elem_div a b
entry sub_arrays [n] (a: [n]f32) (b: [n]f32): [n]f32 = elem_sub a b

entry add_scalar [n] (a: [n]f32) (s: f32): [n]f32 = scalar_add a s
entry mul_scalar [n] (a: [n]f32) (s: f32): [n]f32 = scalar_mul a s
entry div_scalar [n] (a: [n]f32) (s: f32): [n]f32 = scalar_div a s

entry array_sum [n] (x: [n]f32): f32 = sum x
entry array_mean [n] (x: [n]f32): f32 = mean x
entry array_max [n] (x: [n]f32): f32 = max x
entry array_min [n] (x: [n]f32): f32 = min x

type complex = {re: f32, im: f32}

let complex_add (a: complex) (b: complex): complex =
  {re = a.re + b.re, im = a.im + b.im}

let complex_sub (a: complex) (b: complex): complex =
  {re = a.re - b.re, im = a.im - b.im}

let complex_mul (a: complex) (b: complex): complex =
  {re = a.re * b.re - a.im * b.im, im = a.re * b.im + a.im * b.re}

let complex_conj (a: complex): complex =
  {re = a.re, im = -a.im}

let complex_abs (a: complex): f32 =
  let abs_re = f32.abs a.re
  let abs_im = f32.abs a.im
  let s = f32.max abs_re abs_im
  in if s == 0f32 then 0f32
     else
       let scaled_re = abs_re / s
       let scaled_im = abs_im / s
       in s * f32.sqrt (scaled_re * scaled_re + scaled_im * scaled_im)

let complex_abs_sq (a: complex): f32 =
  let abs_re = f32.abs a.re
  let abs_im = f32.abs a.im
  let s = f32.max abs_re abs_im
  in if s == 0f32 then 0f32
     else
       let scaled_re = abs_re / s
       let scaled_im = abs_im / s
       in s * s * (scaled_re * scaled_re + scaled_im * scaled_im)

let complex_scale (sc: f32) (a: complex): complex =
  {re = sc * a.re, im = sc * a.im}

let complex_from_polar (r: f32) (theta: f32): complex =
  {re = r * f32.cos theta, im = r * f32.sin theta}

let complex_normalize (a: complex): complex =
  let mag = complex_abs a
  in if mag > 1e-30f32 then complex_scale (1f32 / mag) a else {re = 1f32, im = 0f32}

let rgpu_edge_quality_to_weight (quality: i32): f32 =
  if quality == 0 then 0.25f32
  else if quality == 1 then 1.0f32
  else if quality == 2 then 0.75f32
  else if quality == 3 then 0.1f32
  else if quality == 4 then 0.5f32
  else 0.0f32

let rgpu_edge_quality_to_weight_batch [n] (qualities: [n]i32): [n]f32 =
  map rgpu_edge_quality_to_weight qualities

let rgpu_propagate_quality [ne] (edge_sources: [ne]i64) (edge_targets: [ne]i64) (edge_qualities: [ne]i32) (node_qualities: []i32) (iterations: i64): [ne]i32 =
  let nq_len = length node_qualities
  let propagate_once (qualities: [ne]i32): [ne]i32 =
    map3 (\src tgt q ->
      let src_q = if src >= 0 && src < nq_len then node_qualities[src] else q
      let tgt_q = if tgt >= 0 && tgt < nq_len then node_qualities[tgt] else q
      let any_bad = src_q == 3 || tgt_q == 3 || q == 3
      let min_q = i32.min src_q (i32.min tgt_q q)
      in if any_bad then 3 else min_q
    ) edge_sources edge_targets qualities
  in loop current = edge_qualities for _i < (i64.max 0 iterations) do
    propagate_once current

let rgpu_compute_degree_sequence [num_edges] (num_nodes: i64) (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64): ([]i64, []i64) =
  let safe_src = map (\s -> if s >= 0 && s < num_nodes then s else 0i64) edge_sources
  let safe_tgt = map (\t -> if t >= 0 && t < num_nodes then t else 0i64) edge_targets
  let src_valid = map (\s -> if s >= 0 && s < num_nodes then 1i64 else 0i64) edge_sources
  let tgt_valid = map (\t -> if t >= 0 && t < num_nodes then 1i64 else 0i64) edge_targets
  let out_degrees = reduce_by_index (replicate num_nodes 0i64) (+) 0i64 safe_src src_valid
  let in_degrees = reduce_by_index (replicate num_nodes 0i64) (+) 0i64 safe_tgt tgt_valid
  in (out_degrees, in_degrees)

let rgpu_canonical_form_signature [num_edges] (num_nodes: i64) (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (edge_qualities: [num_edges]i32): u64 =
  let (out_degrees, in_degrees) = rgpu_compute_degree_sequence num_nodes edge_sources edge_targets
  let n_out = length out_degrees
  let n_in = length in_degrees
  let degree_hash = loop h = 0u64 for i < n_out do
    h * 31u64 + u64.i64 out_degrees[i]
  let in_degree_hash = loop h = 0u64 for i < n_in do
    h * 37u64 + u64.i64 in_degrees[i]
  let quality_hash = loop h = 0u64 for i < num_edges do
    h * 41u64 + u64.i32 edge_qualities[i]
  let node_count_hash = u64.i64 num_nodes * 1000003u64
  let edge_count_hash = u64.i64 num_edges * 999983u64
  in degree_hash ^ in_degree_hash ^ quality_hash ^ node_count_hash ^ edge_count_hash

let rgpu_compute_fractal_dimension [num_nodes][num_edges] (node_hashes: [num_nodes]u64) (edge_sources: [num_edges]i64) (_edge_targets: [num_edges]i64): f32 =
  if num_nodes < 2 then 1f32
  else
    let safe_src = map (\s -> if s >= 0 && s < num_nodes then s else 0i64) edge_sources
    let src_valid = map (\s -> if s >= 0 && s < num_nodes then 1u64 else 0u64) edge_sources
    let connectivity = reduce_by_index (replicate num_nodes 0u64) (+) 0u64 safe_src src_valid
    let adjusted_hashes = map2 (\h c -> h ^ (c * 2654435761u64)) node_hashes connectivity
    let box_sizes = [2i64, 4i64, 8i64, 16i64, 32i64]
    let box_counts = map (\box_size ->
      let boxes = map (\h -> h % u64.i64 box_size) adjusted_hashes
      let sorted_boxes = radix_sort_by_key (\b -> b) u64.num_bits u64.get_bit boxes
      in reduce (+) 0i64 (tabulate num_nodes (\i ->
        if i == 0 || sorted_boxes[i] != sorted_boxes[i-1] then 1i64 else 0i64
      ))
    ) box_sizes
    let valid_data = filter (\(_, c) -> c > 0) (zip box_sizes box_counts)
    let n_valid = length valid_data
    in if n_valid < 2 then 1f32 else
      let log_sizes = map (\(sz, _) -> f32.log (f32.i64 sz)) valid_data
      let log_counts = map (\(_, c) -> f32.log (f32.i64 c)) valid_data
      let nf = f32.i64 n_valid
      let sum_x = reduce (+) 0f32 log_sizes
      let sum_y = reduce (+) 0f32 log_counts
      let sum_xy = reduce (+) 0f32 (map2 (*) log_sizes log_counts)
      let sum_x2 = reduce (+) 0f32 (map (\xi -> xi * xi) log_sizes)
      let denominator = nf * sum_x2 - sum_x * sum_x
      in if f32.abs denominator < 1e-10f32 then 1f32 else
        let slope = (nf * sum_xy - sum_x * sum_y) / denominator
        in f32.abs slope

let rgpu_update_edge_weights [n] (current_weights: [n]f32) (feedback: [n]f32) (learning_rate: f32): [n]f32 =
  let safe_lr = f32.max 0f32 (f32.min 1f32 learning_rate)
  in map2 (\w f ->
    let safe_w = if f32.isnan w then 0.5f32 else w
    let safe_f = if f32.isnan f || f32.isinf f then 0f32 else f
    let new_weight = safe_w + safe_lr * safe_f
    in f32.max 0f32 (f32.min 1f32 new_weight)
  ) current_weights feedback

let rgpu_adaptive_weight [n] (base_weights: [n]f32) (temporal_factors: [n]f32) (spatial_factors: [n]f32) (semantic_factors: [n]f32): [n]f32 =
  map4 (\base temp spat sem ->
    let sb = f32.max 0f32 (f32.min 1f32 (if f32.isnan base then 0f32 else base))
    let st = f32.max 0f32 (f32.min 1e4f32 (if f32.isnan temp then 1f32 else temp))
    let ss = f32.max 0f32 (f32.min 1e4f32 (if f32.isnan spat then 1f32 else spat))
    let se = f32.max 0f32 (f32.min 1e4f32 (if f32.isnan sem then 1f32 else sem))
    let adaptive = sb * st * ss * se
    in f32.max 0f32 (f32.min 1f32 (if f32.isnan adaptive || f32.isinf adaptive then 0f32 else adaptive))
  ) base_weights temporal_factors spatial_factors semantic_factors

let rgpu_propagate_weights [num_edges] (edge_weights: [num_edges]f32) (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (source_node: i64) (num_nodes: i64) (iterations: i64) (decay: f32): [num_edges]f32 =
  let safe_decay = f32.max 0f32 (f32.min 1f32 decay)
  let in_bounds (idx: i64): bool = idx >= 0 && idx < num_nodes
  let initial_visited = tabulate num_nodes (\i -> i == source_node)
  let (final_weights, _) = loop (weights, visited) = (copy edge_weights, initial_visited) for _iter < (i64.max 0 iterations) do
    let touched = map2 (\s t ->
      (in_bounds s && visited[s]) || (in_bounds t && visited[t])
    ) edge_sources edge_targets
    let safe_src = map (\s -> if in_bounds s then s else 0i64) edge_sources
    let safe_tgt = map (\t -> if in_bounds t then t else 0i64) edge_targets
    let src_valid = map in_bounds edge_sources
    let tgt_valid = map in_bounds edge_targets
    let new_visited = reduce_by_index (copy visited) (||) false
      (safe_src ++ safe_tgt)
      (map2 (&&) touched src_valid ++ map2 (&&) touched tgt_valid)
    let touched_expanded = map2 (\s t ->
      (in_bounds s && new_visited[s]) || (in_bounds t && new_visited[t])
    ) edge_sources edge_targets
    let new_weights = map2 (\w te -> if te then w * safe_decay else w) weights touched_expanded
    in (new_weights, new_visited)
  in final_weights

let rgpu_xy_route (src_x: i64) (src_y: i64) (dst_x: i64) (dst_y: i64) (grid_width: i64): (i64, []i64) =
  let safe_gw = i64.max 1 grid_width
  let dx = i64.abs (dst_x - src_x)
  let dy = i64.abs (dst_y - src_y)
  let total_hops = dx + dy
  let path_x_dir = if dst_x > src_x then 1i64 else if dst_x < src_x then -1i64 else 0i64
  let path_y_dir = if dst_y > src_y then 1i64 else if dst_y < src_y then -1i64 else 0i64
  let x_steps = tabulate dx (\i ->
    i64.max 0 (src_y * safe_gw + src_x + (i + 1) * path_x_dir)
  )
  let y_steps = tabulate dy (\i ->
    i64.max 0 ((src_y + (i + 1) * path_y_dir) * safe_gw + dst_x)
  )
  in (total_hops, x_steps ++ y_steps)

let rgpu_route_cost (src_x: i64) (src_y: i64) (dst_x: i64) (dst_y: i64) (hop_cost: f32) (congestion_factor: f32): f32 =
  let safe_hop = f32.max 0f32 hop_cost
  let safe_cong = f32.max 0f32 congestion_factor
  let dx = f32.i64 (i64.abs (dst_x - src_x))
  let dy = f32.i64 (i64.abs (dst_y - src_y))
  let manhattan = dx + dy
  let base_cost = manhattan * safe_hop
  let congestion_penalty = safe_cong * manhattan * manhattan
  let total = base_cost + congestion_penalty
  in if f32.isinf total then f32.highest else total

let rgpu_route_cost_batch [n] (src_xs: [n]i64) (src_ys: [n]i64) (dst_xs: [n]i64) (dst_ys: [n]i64) (hop_cost: f32) (congestion_factor: f32): [n]f32 =
  map4 (\sx sy dx dy -> rgpu_route_cost sx sy dx dy hop_cost congestion_factor) src_xs src_ys dst_xs dst_ys

let rgpu_balance_load [num_cores] (core_loads: [num_cores]f32): [num_cores]f32 =
  if num_cores == 0 then core_loads
  else
    let clean = map (\l -> if f32.isnan l then 0f32 else f32.max 0f32 l) core_loads
    let total = reduce (+) 0f32 clean
    let avg = total / f32.i64 num_cores
    let max_dev = 0.1f32 * avg
    in map (\load ->
      let balanced = if load > avg + max_dev then avg + max_dev
                     else if load < avg - max_dev then avg - max_dev
                     else load
      in f32.max 0f32 balanced
    ) clean

let rgpu_compute_core_utilization [num_cores] (cycles_active: [num_cores]i64) (cycles_idle: [num_cores]i64): [num_cores]f32 =
  map2 (\active idle ->
    let total = active + idle
    in if total > 0 then f32.i64 active / f32.i64 total else 0f32
  ) cycles_active cycles_idle

let rgpu_should_gate_core (utilization: f32) (low_threshold: f32) (current_power: f32) (power_budget: f32): bool =
  utilization < low_threshold && current_power > power_budget * 0.5f32

let rgpu_should_gate_core_batch [n] (utilizations: [n]f32) (low_threshold: f32) (current_power: f32) (power_budget: f32): [n]bool =
  map (\u -> rgpu_should_gate_core u low_threshold current_power power_budget) utilizations

let rgpu_power_budget_check [num_cores] (core_powers: [num_cores]f32) (power_budget: f32): (bool, f32, f32) =
  let total_power = reduce (+) 0f32 core_powers
  let headroom = power_budget - total_power
  let within_budget = total_power <= power_budget
  in (within_budget, total_power, headroom)

let rgpu_sparsity_mask [n] (workloads: [n]f32) (threshold: f32): [n]bool =
  map (\w -> w >= threshold) workloads

let rgpu_energy_savings [n] (workloads: [n]f32) (threshold: f32) (idle_power: f32) (active_power: f32): f32 =
  let mask = rgpu_sparsity_mask workloads threshold
  let inactive_count = reduce (+) 0i64 (map (\m -> if m then 0i64 else 1i64) mask)
  let savings_per_core = f32.max 0f32 (active_power - idle_power)
  in f32.i64 inactive_count * savings_per_core

let rgpu_compute_sparsity_ratio [n] (workloads: [n]f32) (threshold: f32): f32 =
  if n == 0 then 0f32
  else
    let mask = rgpu_sparsity_mask workloads threshold
    let inactive_count = reduce (+) 0i64 (map (\m -> if m then 0i64 else 1i64) mask)
    in f32.i64 inactive_count / f32.i64 n

let rgpu_quantum_correlation (state1: complex) (state2: complex): complex =
  complex_mul state1 (complex_conj state2)

let rgpu_quantum_correlation_batch [n] (states1: [n]complex) (states2: [n]complex): [n]complex =
  map2 rgpu_quantum_correlation states1 states2

let rgpu_entangle_states (state1: complex) (state2: complex): complex =
  let sqrt_2_inv = 1f32 / f32.sqrt 2f32
  in complex_scale sqrt_2_inv (complex_add state1 state2)

let rgpu_entangle_states_batch [n] (states1: [n]complex) (states2: [n]complex): [n]complex =
  map2 rgpu_entangle_states states1 states2

let rgpu_measure_probability (state: complex): f32 =
  let prob = complex_abs_sq state
  in f32.max 0f32 (f32.min 1f32 prob)

let rgpu_measure_probability_batch [n] (states: [n]complex): [n]f32 =
  if n == 0 then replicate n 0f32
  else
    let raw = map (\s -> complex_abs_sq s) states
    let total = reduce (+) 0f32 raw
    in if total <= 0f32 || f32.isnan total
       then replicate n (1f32 / f32.i64 n)
       else map (\p -> p / total) raw

let rgpu_hadamard_transform (state: complex): complex =
  let inv_sqrt2 = 1f32 / f32.sqrt 2f32
  in {re = inv_sqrt2 * (state.re + state.im), im = inv_sqrt2 * (state.re - state.im)}

let rgpu_hadamard_transform_batch [n] (states: [n]complex): [n]complex =
  map rgpu_hadamard_transform states

let rgpu_phase_shift (state: complex) (theta: f32): complex =
  let rotation = complex_from_polar 1f32 theta
  in complex_mul state rotation

let rgpu_phase_shift_batch [n] (states: [n]complex) (thetas: [n]f32): [n]complex =
  map2 rgpu_phase_shift states thetas

let rgpu_pauli_x (state: complex): complex =
  {re = state.im, im = state.re}

let rgpu_pauli_y (state: complex): complex =
  {re = -state.im, im = state.re}

let rgpu_pauli_z (state: complex): complex =
  {re = state.re, im = -state.im}

let rgpu_cnot (control: complex) (target: complex): (complex, complex) =
  let phase = f32.atan2 control.im control.re
  let cos_p = f32.cos phase
  let sin_p = f32.sin phase
  let new_target = {re = cos_p * target.re - sin_p * target.im,
                    im = sin_p * target.re + cos_p * target.im}
  in (control, new_target)

let rgpu_fractal_transform (state: complex) (depth: i64): complex =
  loop current = state for i < (i64.max 0 depth) do
    let scale_factor = 1f32 / float_pow 2f32 (i + 1)
    let phase = f32.atan2 current.im current.re
    let rotation = complex_from_polar scale_factor (phase * scale_factor)
    let sum_state = complex_add current rotation
    let norm_factor = 1f32 / f32.sqrt (1f32 + scale_factor * scale_factor)
    in complex_scale norm_factor sum_state

let rgpu_relational_and (state1: complex) (state2: complex): complex =
  complex_normalize (complex_mul state1 state2)

let rgpu_relational_or (state1: complex) (state2: complex): complex =
  let sqrt_2_inv = 1f32 / f32.sqrt 2f32
  in complex_normalize (complex_scale sqrt_2_inv (complex_add state1 state2))

let rgpu_relational_xor (state1: complex) (state2: complex): complex =
  let sqrt_2_inv = 1f32 / f32.sqrt 2f32
  in complex_normalize (complex_scale sqrt_2_inv (complex_sub state1 state2))

let rgpu_partition_nodes (num_nodes: i64) (num_cores: i64): []i64 =
  if num_cores <= 0 then replicate num_nodes 0i64
  else
    let nodes_per_core = num_nodes / num_cores
    let remainder = num_nodes % num_cores
    in tabulate num_nodes (\i ->
      if i < remainder * (nodes_per_core + 1) then
        i / (nodes_per_core + 1)
      else
        let adjusted = i - remainder * (nodes_per_core + 1)
        in remainder + adjusted / (i64.max 1 nodes_per_core)
    )

let rgpu_compute_partition_boundaries (num_nodes: i64) (num_cores: i64): []i64 =
  let safe_cores = i64.max 1 num_cores
  let nodes_per_core = num_nodes / safe_cores
  let remainder = num_nodes % safe_cores
  in tabulate (safe_cores + 1) (\i ->
    if i == 0 then 0i64
    else if i <= remainder then i * (nodes_per_core + 1)
    else remainder * (nodes_per_core + 1) + (i - remainder) * nodes_per_core
  )

let rgpu_distribute_edges [num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (node_partitions: []i64): [num_edges]i64 =
  let np_len = length node_partitions
  in map2 (\src tgt ->
    let src_p = if src >= 0 && src < np_len then node_partitions[src] else 0i64
    let tgt_p = if tgt >= 0 && tgt < np_len then node_partitions[tgt] else 0i64
    in i64.min src_p tgt_p
  ) edge_sources edge_targets

let rgpu_count_cross_partition_edges [num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (node_partitions: []i64): i64 =
  let np_len = length node_partitions
  in reduce (+) 0i64 (map2 (\src tgt ->
    let src_p = if src >= 0 && src < np_len then node_partitions[src] else -1i64
    let tgt_p = if tgt >= 0 && tgt < np_len then node_partitions[tgt] else -1i64
    in if src_p != tgt_p then 1i64 else 0i64
  ) edge_sources edge_targets)

let rgpu_compute_partition_load [num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (node_partitions: []i64) (num_cores: i64): [num_cores]i64 =
  let edge_partitions = rgpu_distribute_edges edge_sources edge_targets node_partitions
  let safe_parts = map (\p -> if p >= 0 && p < num_cores then p else 0i64) edge_partitions
  in reduce_by_index (replicate num_cores 0i64) (+) 0i64 safe_parts (replicate num_edges 1i64)

let rgpu_noc_neighbors (core_id: i64) (grid_width: i64) (grid_height: i64): [4]i64 =
  let safe_gw = i64.max 1 grid_width
  let safe_gh = i64.max 1 grid_height
  let x = core_id % safe_gw
  let y = core_id / safe_gw
  let left = if x > 0 then core_id - 1 else -1i64
  let right = if x < safe_gw - 1 then core_id + 1 else -1i64
  let up = if y > 0 then core_id - safe_gw else -1i64
  let down = if y < safe_gh - 1 then core_id + safe_gw else -1i64
  in [left, right, up, down]

let rgpu_compute_core_position (core_id: i64) (grid_width: i64): (i64, i64) =
  let safe_gw = i64.max 1 grid_width
  in (core_id % safe_gw, core_id / safe_gw)

let rgpu_core_id_from_position (x: i64) (y: i64) (grid_width: i64): i64 =
  let safe_gw = i64.max 1 grid_width
  in y * safe_gw + x

let rgpu_manhattan_distance (src_core: i64) (dst_core: i64) (grid_width: i64): i64 =
  let safe_gw = i64.max 1 grid_width
  let (src_x, src_y) = rgpu_compute_core_position src_core safe_gw
  let (dst_x, dst_y) = rgpu_compute_core_position dst_core safe_gw
  in i64.abs (dst_x - src_x) + i64.abs (dst_y - src_y)

let rgpu_message_latency (src_core: i64) (dst_core: i64) (grid_width: i64) (hop_latency: f32) (base_latency: f32): f32 =
  let hops = rgpu_manhattan_distance src_core dst_core grid_width
  in base_latency + f32.i64 hops * hop_latency

entry rgpu_quality_to_weight (quality: i32): f32 = rgpu_edge_quality_to_weight quality
entry rgpu_quality_to_weight_batch [n] (qualities: [n]i32): [n]f32 = rgpu_edge_quality_to_weight_batch qualities

entry rgpu_propagate_edge_quality [ne] (edge_sources: [ne]i64) (edge_targets: [ne]i64) (edge_qualities: [ne]i32) (node_qualities: []i32) (iterations: i64): [ne]i32 =
  rgpu_propagate_quality edge_sources edge_targets edge_qualities node_qualities iterations

entry rgpu_degree_sequence [num_edges] (num_nodes: i64) (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64): ([]i64, []i64) =
  rgpu_compute_degree_sequence num_nodes edge_sources edge_targets

entry rgpu_canonical_signature [num_edges] (num_nodes: i64) (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (edge_qualities: [num_edges]i32): u64 =
  rgpu_canonical_form_signature num_nodes edge_sources edge_targets edge_qualities

entry rgpu_fractal_dim [num_nodes][num_edges] (node_hashes: [num_nodes]u64) (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64): f32 =
  rgpu_compute_fractal_dimension node_hashes edge_sources edge_targets

entry rgpu_update_weights [n] (weights: [n]f32) (feedback: [n]f32) (learning_rate: f32): [n]f32 =
  rgpu_update_edge_weights weights feedback learning_rate

entry rgpu_adaptive_weights [n] (base_weights: [n]f32) (temporal: [n]f32) (spatial: [n]f32) (semantic: [n]f32): [n]f32 =
  rgpu_adaptive_weight base_weights temporal spatial semantic

entry rgpu_propagate_edge_weights [num_edges] (edge_weights: [num_edges]f32) (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (source_node: i64) (num_nodes: i64) (iterations: i64) (decay: f32): [num_edges]f32 =
  rgpu_propagate_weights edge_weights edge_sources edge_targets source_node num_nodes iterations decay

entry rgpu_compute_xy_route (src_x: i64) (src_y: i64) (dst_x: i64) (dst_y: i64) (grid_width: i64): (i64, []i64) =
  rgpu_xy_route src_x src_y dst_x dst_y grid_width

entry rgpu_compute_route_cost (src_x: i64) (src_y: i64) (dst_x: i64) (dst_y: i64) (hop_cost: f32) (congestion: f32): f32 =
  rgpu_route_cost src_x src_y dst_x dst_y hop_cost congestion

entry rgpu_compute_route_cost_batch [n] (src_xs: [n]i64) (src_ys: [n]i64) (dst_xs: [n]i64) (dst_ys: [n]i64) (hop_cost: f32) (congestion: f32): [n]f32 =
  rgpu_route_cost_batch src_xs src_ys dst_xs dst_ys hop_cost congestion

entry rgpu_load_balance [num_cores] (loads: [num_cores]f32): [num_cores]f32 =
  rgpu_balance_load loads

entry rgpu_core_utilization [n] (active: [n]i64) (idle: [n]i64): [n]f32 =
  rgpu_compute_core_utilization active idle

entry rgpu_gate_core_check (utilization: f32) (threshold: f32) (current_power: f32) (budget: f32): bool =
  rgpu_should_gate_core utilization threshold current_power budget

entry rgpu_gate_core_check_batch [n] (utilizations: [n]f32) (threshold: f32) (current_power: f32) (budget: f32): [n]bool =
  rgpu_should_gate_core_batch utilizations threshold current_power budget

entry rgpu_power_check [n] (core_powers: [n]f32) (budget: f32): (bool, f32, f32) =
  rgpu_power_budget_check core_powers budget

entry rgpu_sparsity [n] (workloads: [n]f32) (threshold: f32): [n]bool =
  rgpu_sparsity_mask workloads threshold

entry rgpu_compute_energy_savings [n] (workloads: [n]f32) (threshold: f32) (idle_power: f32) (active_power: f32): f32 =
  rgpu_energy_savings workloads threshold idle_power active_power

entry rgpu_sparsity_ratio [n] (workloads: [n]f32) (threshold: f32): f32 =
  rgpu_compute_sparsity_ratio workloads threshold

entry rgpu_quantum_corr (re1: f32) (im1: f32) (re2: f32) (im2: f32): (f32, f32) =
  let result = rgpu_quantum_correlation {re=re1, im=im1} {re=re2, im=im2}
  in (result.re, result.im)

entry rgpu_quantum_corr_batch [n] (re1: [n]f32) (im1: [n]f32) (re2: [n]f32) (im2: [n]f32): ([n]f32, [n]f32) =
  let states1 = map2 (\r i -> {re=r, im=i}) re1 im1
  let states2 = map2 (\r i -> {re=r, im=i}) re2 im2
  let results = rgpu_quantum_correlation_batch states1 states2
  in (map (\r -> r.re) results, map (\r -> r.im) results)

entry rgpu_entangle (re1: f32) (im1: f32) (re2: f32) (im2: f32): (f32, f32) =
  let result = rgpu_entangle_states {re=re1, im=im1} {re=re2, im=im2}
  in (result.re, result.im)

entry rgpu_entangle_batch [n] (re1: [n]f32) (im1: [n]f32) (re2: [n]f32) (im2: [n]f32): ([n]f32, [n]f32) =
  let states1 = map2 (\r i -> {re=r, im=i}) re1 im1
  let states2 = map2 (\r i -> {re=r, im=i}) re2 im2
  let results = rgpu_entangle_states_batch states1 states2
  in (map (\r -> r.re) results, map (\r -> r.im) results)

entry rgpu_measure_prob (re: f32) (im: f32): f32 =
  rgpu_measure_probability {re=re, im=im}

entry rgpu_measure_prob_batch [n] (re: [n]f32) (im: [n]f32): [n]f32 =
  rgpu_measure_probability_batch (map2 (\r i -> {re=r, im=i}) re im)

entry rgpu_hadamard (re: f32) (im: f32): (f32, f32) =
  let result = rgpu_hadamard_transform {re=re, im=im}
  in (result.re, result.im)

entry rgpu_hadamard_batch [n] (re: [n]f32) (im: [n]f32): ([n]f32, [n]f32) =
  let results = rgpu_hadamard_transform_batch (map2 (\r i -> {re=r, im=i}) re im)
  in (map (\r -> r.re) results, map (\r -> r.im) results)

entry rgpu_phase (re: f32) (im: f32) (theta: f32): (f32, f32) =
  let result = rgpu_phase_shift {re=re, im=im} theta
  in (result.re, result.im)

entry rgpu_phase_batch [n] (re: [n]f32) (im: [n]f32) (thetas: [n]f32): ([n]f32, [n]f32) =
  let results = rgpu_phase_shift_batch (map2 (\r i -> {re=r, im=i}) re im) thetas
  in (map (\r -> r.re) results, map (\r -> r.im) results)

entry rgpu_partition (num_nodes: i64) (num_cores: i64): []i64 =
  rgpu_partition_nodes num_nodes num_cores

entry rgpu_partition_bounds (num_nodes: i64) (num_cores: i64): []i64 =
  rgpu_compute_partition_boundaries num_nodes num_cores

entry rgpu_edge_distribution [num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (node_partitions: []i64): [num_edges]i64 =
  rgpu_distribute_edges edge_sources edge_targets node_partitions

entry rgpu_cross_partition_edges [num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (node_partitions: []i64): i64 =
  rgpu_count_cross_partition_edges edge_sources edge_targets node_partitions

entry rgpu_partition_loads [num_edges] (edge_sources: [num_edges]i64) (edge_targets: [num_edges]i64) (node_partitions: []i64) (num_cores: i64): [num_cores]i64 =
  rgpu_compute_partition_load edge_sources edge_targets node_partitions num_cores

entry rgpu_core_neighbors (core_id: i64) (grid_width: i64) (grid_height: i64): [4]i64 =
  rgpu_noc_neighbors core_id grid_width grid_height

entry rgpu_core_position (core_id: i64) (grid_width: i64): (i64, i64) =
  rgpu_compute_core_position core_id grid_width

entry rgpu_core_from_pos (x: i64) (y: i64) (grid_width: i64): i64 =
  rgpu_core_id_from_position x y grid_width

entry rgpu_distance (src: i64) (dst: i64) (grid_width: i64): i64 =
  rgpu_manhattan_distance src dst grid_width

entry rgpu_latency (src: i64) (dst: i64) (grid_width: i64) (hop_lat: f32) (base_lat: f32): f32 =
  rgpu_message_latency src dst grid_width hop_lat base_lat

entry rgpu_fractal_xform (re: f32) (im: f32) (depth: i64): (f32, f32) =
  let result = rgpu_fractal_transform {re=re, im=im} depth
  in (result.re, result.im)

entry rgpu_rel_and (re1: f32) (im1: f32) (re2: f32) (im2: f32): (f32, f32) =
  let result = rgpu_relational_and {re=re1, im=im1} {re=re2, im=im2}
  in (result.re, result.im)

entry rgpu_rel_or (re1: f32) (im1: f32) (re2: f32) (im2: f32): (f32, f32) =
  let result = rgpu_relational_or {re=re1, im=im1} {re=re2, im=im2}
  in (result.re, result.im)

entry rgpu_rel_xor (re1: f32) (im1: f32) (re2: f32) (im2: f32): (f32, f32) =
  let result = rgpu_relational_xor {re=re1, im=im1} {re=re2, im=im2}
  in (result.re, result.im)

let rsf_backward_full [n][half] (input: [n][half*2]f16) (grad_output: [n][half*2]f16)
  (weights_s: [half][half+1]f16) (weights_t: [half][half+1]f16)
  (clip_min: f16) (clip_max: f16)
  : ([half][half+1]f16, [half][half+1]f16, [n][half*2]f16) =
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
    let dy1_total = map2 (\dy1_j j ->
      dy1_j f16.+ f16.sum (map (\k -> weights_t[k][j] f16.* dy2[k]) (iota half))
    ) dy1 (iota half)
    let ds = map2 (\j ps ->
      if ps f16.>= clip_min && ps f16.<= clip_max
      then dy1_total[j] f16.* y1[j]
      else f16.i32 0
    ) (iota half) pre_scale
    let dx1 = map2 (f16.*) dy1_total scale
    let dx2_from_ds = map (\k ->
      f16.sum (map (\j -> ds[j] f16.* weights_s[j][k]) (iota half))
    ) (iota half)
    let dx2 = map2 (f16.+) dy2 dx2_from_ds
    in (ds, x2, dy2, y1, dx1 ++ dx2 :> [half*2]f16)
  ) input grad_output
  let ds_all = map (\(a,_,_,_,_) -> a) per_tok
  let x2_all = map (\(_,b,_,_,_) -> b) per_tok
  let dy2_all = map (\(_,_,c,_,_) -> c) per_tok
  let y1_all = map (\(_,_,_,d,_) -> d) per_tok
  let g_in = map (\(_,_,_,_,e) -> e) per_tok
  let ds_t = transpose ds_all
  let x2_t = transpose x2_all
  let dy2_t = transpose dy2_all
  let y1_t = transpose y1_all
  let acc_ws = map2 (\ds_row bias ->
    let inner = map (\x2_column -> f16.sum (map2 (f16.*) ds_row x2_column)) x2_t
    in inner ++ [bias] :> [half+1]f16
  ) ds_t (map f16.sum ds_t)
  let acc_wt = map2 (\dy2_row bias ->
    let inner = map (\y1_column -> f16.sum (map2 (f16.*) dy2_row y1_column)) y1_t
    in inner ++ [bias] :> [half+1]f16
  ) dy2_t (map f16.sum dy2_t)
  in (acc_ws, acc_wt, g_in)

entry scale_matrix_f16 [rows][columns] (values: *[rows][columns]f16) (scale_factor: f16) : *[rows][columns]f16 =
  map (map (\value -> value f16.* scale_factor)) values

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
