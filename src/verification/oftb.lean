variable {F : Type}

def succ_not_zero (n : Nat) (h : Nat.succ n = 0) : False :=
  nomatch h

def zero_not_succ (n : Nat) (h : 0 = Nat.succ n) : False :=
  nomatch h

def succ_inj {n m : Nat} (h : Nat.succ n = Nat.succ m) : n = m :=
  congrArg (fun x => match x with | 0 => 0 | Nat.succ k => k) h

def nat_zero_add : (m : Nat) → 0 + m = m
| 0 => Eq.refl 0
| Nat.succ m => congrArg Nat.succ (nat_zero_add m)

def nat_succ_add (n : Nat) : (m : Nat) → Nat.succ n + m = Nat.succ (n + m)
| 0 => Eq.refl (Nat.succ n)
| Nat.succ m => congrArg Nat.succ (nat_succ_add n m)

def congr_cons {α : Type} {x y : α} {xs ys : List α} (hx : x = y) (hxs : xs = ys) : x :: xs = y :: ys :=
  match hx, hxs with
  | Eq.refl _, Eq.refl _ => Eq.refl _

def congr_append {α : Type} {xs ys us vs : List α} (h1 : xs = ys) (h2 : us = vs) : xs ++ us = ys ++ vs :=
  match h1, h2 with
  | Eq.refl _, Eq.refl _ => Eq.refl _

def congr_add {n1 n2 m1 m2 : Nat} (h1 : n1 = n2) (h2 : m1 = m2) : n1 + m1 = n2 + m2 :=
  match h1, h2 with
  | Eq.refl _, Eq.refl _ => Eq.refl _

def split_at (n : Nat) (L : List F) : List F × List F :=
  match n with
  | 0 => ([], L)
  | Nat.succ k =>
    match L with
    | [] => ([], [])
    | x :: xs =>
      let (l1, l2) := split_at k xs
      (x :: l1, l2)

theorem split_at_lengths_gen : (n m : Nat) → (L : List F) → L.length = n + m →
  (split_at n L).1.length = n ∧ (split_at n L).2.length = m
| 0, m, L, h =>
  ⟨Eq.refl 0, Eq.trans h (nat_zero_add m)⟩
| Nat.succ n, m, [], h =>
  False.elim (zero_not_succ (n + m) (Eq.trans h (nat_succ_add n m)))
| Nat.succ n, m, _ :: xs, h =>
  let h' : Nat.succ xs.length = Nat.succ (n + m) := Eq.trans h (nat_succ_add n m)
  let h_len : xs.length = n + m := succ_inj h'
  let rec_res := split_at_lengths_gen n m xs h_len
  ⟨congrArg Nat.succ rec_res.left, rec_res.right⟩

theorem split_at_reconstruct : (n : Nat) → (L : List F) →
  (split_at n L).1 ++ (split_at n L).2 = L
| 0, L => Eq.refl L
| Nat.succ _, [] => Eq.refl []
| Nat.succ n, x :: xs =>
  let rec_res := split_at_reconstruct n xs
  congrArg (List.cons x) rec_res

theorem split_at_append : (n : Nat) → (L1 L2 : List F) → L1.length = n →
  split_at n (L1 ++ L2) = (L1, L2)
| 0, [], L2, _ => Eq.refl ([], L2)
| 0, _ :: _, _, h => False.elim (succ_not_zero _ h)
| Nat.succ n, x :: xs, L2, h =>
  let h_rec := split_at_append n xs L2 (succ_inj h)
  congrArg (fun p => (x :: p.1, p.2)) h_rec
| Nat.succ _, [], _, h => False.elim (zero_not_succ _ h)

theorem length_zipWith : (f : F → F → F) → (L1 L2 : List F) → L1.length = L2.length →
  (List.zipWith f L1 L2).length = L1.length
| _, [], [], _ => Eq.refl 0
| f, _ :: xs, _ :: ys, h =>
  let h_len : xs.length = ys.length := succ_inj h
  congrArg Nat.succ (length_zipWith f xs ys h_len)
| _, _ :: _, [], h => False.elim (succ_not_zero _ h)
| _, [], _ :: _, h => False.elim (zero_not_succ _ h)

theorem length_append : (L1 L2 : List F) → (L1 ++ L2).length = L1.length + L2.length
| [], L2 => Eq.symm (nat_zero_add L2.length)
| _ :: xs, L2 =>
  Eq.trans (congrArg Nat.succ (length_append xs L2)) (Eq.symm (nat_succ_add xs.length L2.length))

structure ScaleAlgebra (F : Type) where
  add : F → F → F
  sub : F → F → F
  mul : F → F → F
  scale : F
  two : F
  one : F
  distrib_add : ∀ x y z : F, add (mul x z) (mul y z) = mul (add x y) z
  distrib_sub : ∀ x y z : F, sub (mul x z) (mul y z) = mul (sub x y) z
  mul_comm : ∀ x y : F, mul x y = mul y x
  mul_assoc : ∀ x y z : F, mul (mul x y) z = mul x (mul y z)
  mul_one : ∀ x : F, mul x one = x
  add_sub_add : ∀ a b : F, add (sub a b) (add a b) = mul two a
  add_sub_sub : ∀ a b : F, sub (add a b) (sub a b) = mul two b
  sub_add_sub_rev : ∀ u v : F, sub (add u v) (sub v u) = mul two u
  add_add_sub_rev : ∀ u v : F, add (add u v) (sub v u) = mul two v
  scale_property : mul (mul two scale) scale = one

def first_component_proof (sa : ScaleAlgebra F) (a b : F) :
  sa.mul (sa.add (sa.mul (sa.sub a b) sa.scale) (sa.mul (sa.add a b) sa.scale)) sa.scale = a :=
  let step1 : sa.add (sa.mul (sa.sub a b) sa.scale) (sa.mul (sa.add a b) sa.scale) = sa.mul (sa.add (sa.sub a b) (sa.add a b)) sa.scale :=
    sa.distrib_add (sa.sub a b) (sa.add a b) sa.scale
  let step2 : sa.add (sa.sub a b) (sa.add a b) = sa.mul sa.two a :=
    sa.add_sub_add a b
  let step3 : sa.mul (sa.add (sa.sub a b) (sa.add a b)) sa.scale = sa.mul (sa.mul sa.two a) sa.scale :=
    congrArg (fun x => sa.mul x sa.scale) step2
  let step4 : sa.add (sa.mul (sa.sub a b) sa.scale) (sa.mul (sa.add a b) sa.scale) = sa.mul (sa.mul sa.two a) sa.scale :=
    Eq.trans step1 step3
  let step5 : sa.mul (sa.add (sa.mul (sa.sub a b) sa.scale) (sa.mul (sa.add a b) sa.scale)) sa.scale = sa.mul (sa.mul (sa.mul sa.two a) sa.scale) sa.scale :=
    congrArg (fun x => sa.mul x sa.scale) step4
  let s1 : sa.mul (sa.mul (sa.mul sa.two a) sa.scale) sa.scale = sa.mul (sa.mul (sa.mul a sa.two) sa.scale) sa.scale :=
    congrArg (fun x => sa.mul (sa.mul x sa.scale) sa.scale) (sa.mul_comm sa.two a)
  let s2 : sa.mul (sa.mul (sa.mul a sa.two) sa.scale) sa.scale = sa.mul (sa.mul a (sa.mul sa.two sa.scale)) sa.scale :=
    congrArg (fun x => sa.mul x sa.scale) (sa.mul_assoc a sa.two sa.scale)
  let s3 : sa.mul (sa.mul a (sa.mul sa.two sa.scale)) sa.scale = sa.mul a (sa.mul (sa.mul sa.two sa.scale) sa.scale) :=
    sa.mul_assoc a (sa.mul sa.two sa.scale) sa.scale
  let s4 : sa.mul a (sa.mul (sa.mul sa.two sa.scale) sa.scale) = sa.mul a sa.one :=
    congrArg (sa.mul a) sa.scale_property
  let s5 : sa.mul a sa.one = a :=
    sa.mul_one a
  let s_tot : sa.mul (sa.mul (sa.mul sa.two a) sa.scale) sa.scale = a :=
    Eq.trans s1 (Eq.trans s2 (Eq.trans s3 (Eq.trans s4 s5)))
  Eq.trans step5 s_tot

def second_component_proof (sa : ScaleAlgebra F) (a b : F) :
  sa.mul (sa.sub (sa.mul (sa.add a b) sa.scale) (sa.mul (sa.sub a b) sa.scale)) sa.scale = b :=
  let step1 : sa.sub (sa.mul (sa.add a b) sa.scale) (sa.mul (sa.sub a b) sa.scale) = sa.mul (sa.sub (sa.add a b) (sa.sub a b)) sa.scale :=
    sa.distrib_sub (sa.add a b) (sa.sub a b) sa.scale
  let step2 : sa.sub (sa.add a b) (sa.sub a b) = sa.mul sa.two b :=
    sa.add_sub_sub a b
  let step3 : sa.mul (sa.sub (sa.add a b) (sa.sub a b)) sa.scale = sa.mul (sa.mul sa.two b) sa.scale :=
    congrArg (fun x => sa.mul x sa.scale) step2
  let step4 : sa.sub (sa.mul (sa.add a b) sa.scale) (sa.mul (sa.sub a b) sa.scale) = sa.mul (sa.mul sa.two b) sa.scale :=
    Eq.trans step1 step3
  let step5 : sa.mul (sa.sub (sa.mul (sa.add a b) sa.scale) (sa.mul (sa.sub a b) sa.scale)) sa.scale = sa.mul (sa.mul (sa.mul sa.two b) sa.scale) sa.scale :=
    congrArg (fun x => sa.mul x sa.scale) step4
  let s1 : sa.mul (sa.mul (sa.mul sa.two b) sa.scale) sa.scale = sa.mul (sa.mul (sa.mul b sa.two) sa.scale) sa.scale :=
    congrArg (fun x => sa.mul (sa.mul x sa.scale) sa.scale) (sa.mul_comm sa.two b)
  let s2 : sa.mul (sa.mul (sa.mul b sa.two) sa.scale) sa.scale = sa.mul (sa.mul b (sa.mul sa.two sa.scale)) sa.scale :=
    congrArg (fun x => sa.mul x sa.scale) (sa.mul_assoc b sa.two sa.scale)
  let s3 : sa.mul (sa.mul b (sa.mul sa.two sa.scale)) sa.scale = sa.mul b (sa.mul (sa.mul sa.two sa.scale) sa.scale) :=
    sa.mul_assoc b (sa.mul sa.two sa.scale) sa.scale
  let s4 : sa.mul b (sa.mul (sa.mul sa.two sa.scale) sa.scale) = sa.mul b sa.one :=
    congrArg (sa.mul b) sa.scale_property
  let s5 : sa.mul b sa.one = b :=
    sa.mul_one b
  let s_tot : sa.mul (sa.mul (sa.mul sa.two b) sa.scale) sa.scale = b :=
    Eq.trans s1 (Eq.trans s2 (Eq.trans s3 (Eq.trans s4 s5)))
  Eq.trans step5 s_tot

def first_component_proof_rev (sa : ScaleAlgebra F) (u v : F) :
  sa.mul (sa.sub (sa.mul (sa.add u v) sa.scale) (sa.mul (sa.sub v u) sa.scale)) sa.scale = u :=
  let step1 : sa.sub (sa.mul (sa.add u v) sa.scale) (sa.mul (sa.sub v u) sa.scale) = sa.mul (sa.sub (sa.add u v) (sa.sub v u)) sa.scale :=
    sa.distrib_sub (sa.add u v) (sa.sub v u) sa.scale
  let step2 : sa.sub (sa.add u v) (sa.sub v u) = sa.mul sa.two u :=
    sa.sub_add_sub_rev u v
  let step3 : sa.mul (sa.sub (sa.add u v) (sa.sub v u)) sa.scale = sa.mul (sa.mul sa.two u) sa.scale :=
    congrArg (fun x => sa.mul x sa.scale) step2
  let step4 : sa.sub (sa.mul (sa.add u v) sa.scale) (sa.mul (sa.sub v u) sa.scale) = sa.mul (sa.mul sa.two u) sa.scale :=
    Eq.trans step1 step3
  let step5 : sa.mul (sa.sub (sa.mul (sa.add u v) sa.scale) (sa.mul (sa.sub v u) sa.scale)) sa.scale = sa.mul (sa.mul (sa.mul sa.two u) sa.scale) sa.scale :=
    congrArg (fun x => sa.mul x sa.scale) step4
  let s1 : sa.mul (sa.mul (sa.mul sa.two u) sa.scale) sa.scale = sa.mul (sa.mul (sa.mul u sa.two) sa.scale) sa.scale :=
    congrArg (fun x => sa.mul (sa.mul x sa.scale) sa.scale) (sa.mul_comm sa.two u)
  let s2 : sa.mul (sa.mul (sa.mul u sa.two) sa.scale) sa.scale = sa.mul (sa.mul u (sa.mul sa.two sa.scale)) sa.scale :=
    congrArg (fun x => sa.mul x sa.scale) (sa.mul_assoc u sa.two sa.scale)
  let s3 : sa.mul (sa.mul u (sa.mul sa.two sa.scale)) sa.scale = sa.mul u (sa.mul (sa.mul sa.two sa.scale) sa.scale) :=
    sa.mul_assoc u (sa.mul sa.two sa.scale) sa.scale
  let s4 : sa.mul u (sa.mul (sa.mul sa.two sa.scale) sa.scale) = sa.mul u sa.one :=
    congrArg (sa.mul u) sa.scale_property
  let s5 : sa.mul u sa.one = u :=
    sa.mul_one u
  let s_tot : sa.mul (sa.mul (sa.mul sa.two u) sa.scale) sa.scale = u :=
    Eq.trans s1 (Eq.trans s2 (Eq.trans s3 (Eq.trans s4 s5)))
  Eq.trans step5 s_tot

def second_component_proof_rev (sa : ScaleAlgebra F) (u v : F) :
  sa.mul (sa.add (sa.mul (sa.add u v) sa.scale) (sa.mul (sa.sub v u) sa.scale)) sa.scale = v :=
  let step1 : sa.add (sa.mul (sa.add u v) sa.scale) (sa.mul (sa.sub v u) sa.scale) = sa.mul (sa.add (sa.add u v) (sa.sub v u)) sa.scale :=
    sa.distrib_add (sa.add u v) (sa.sub v u) sa.scale
  let step2 : sa.add (sa.add u v) (sa.sub v u) = sa.mul sa.two v :=
    sa.add_add_sub_rev u v
  let step3 : sa.mul (sa.add (sa.add u v) (sa.sub v u)) sa.scale = sa.mul (sa.mul sa.two v) sa.scale :=
    congrArg (fun x => sa.mul x sa.scale) step2
  let step4 : sa.add (sa.mul (sa.add u v) sa.scale) (sa.mul (sa.sub v u) sa.scale) = sa.mul (sa.mul sa.two v) sa.scale :=
    Eq.trans step1 step3
  let step5 : sa.mul (sa.add (sa.mul (sa.add u v) sa.scale) (sa.mul (sa.sub v u) sa.scale)) sa.scale = sa.mul (sa.mul (sa.mul sa.two v) sa.scale) sa.scale :=
    congrArg (fun x => sa.mul x sa.scale) step4
  let s1 : sa.mul (sa.mul (sa.mul sa.two v) sa.scale) sa.scale = sa.mul (sa.mul (sa.mul v sa.two) sa.scale) sa.scale :=
    congrArg (fun x => sa.mul (sa.mul x sa.scale) sa.scale) (sa.mul_comm sa.two v)
  let s2 : sa.mul (sa.mul (sa.mul v sa.two) sa.scale) sa.scale = sa.mul (sa.mul v (sa.mul sa.two sa.scale)) sa.scale :=
    congrArg (fun x => sa.mul x sa.scale) (sa.mul_assoc v sa.two sa.scale)
  let s3 : sa.mul (sa.mul v (sa.mul sa.two sa.scale)) sa.scale = sa.mul v (sa.mul (sa.mul sa.two sa.scale) sa.scale) :=
    sa.mul_assoc v (sa.mul sa.two sa.scale) sa.scale
  let s4 : sa.mul v (sa.mul (sa.mul sa.two sa.scale) sa.scale) = sa.mul v sa.one :=
    congrArg (sa.mul v) sa.scale_property
  let s5 : sa.mul v sa.one = v :=
    sa.mul_one v
  let s_tot : sa.mul (sa.mul (sa.mul sa.two v) sa.scale) sa.scale = v :=
    Eq.trans s1 (Eq.trans s2 (Eq.trans s3 (Eq.trans s4 s5)))
  Eq.trans step5 s_tot

def prove_inverse_pair (sa : ScaleAlgebra F) (a b : F) :
  let u := sa.mul (sa.sub a b) sa.scale
  let v := sa.mul (sa.add a b) sa.scale
  sa.mul (sa.add u v) sa.scale = a ∧ sa.mul (sa.sub v u) sa.scale = b :=
  ⟨first_component_proof sa a b, second_component_proof sa a b⟩

def prove_inverse_pair_rev (sa : ScaleAlgebra F) (u v : F) :
  let a := sa.mul (sa.add u v) sa.scale
  let b := sa.mul (sa.sub v u) sa.scale
  sa.mul (sa.sub a b) sa.scale = u ∧ sa.mul (sa.add a b) sa.scale = v :=
  ⟨first_component_proof_rev sa u v, second_component_proof_rev sa u v⟩

def zipWith_inverse_list (sa : ScaleAlgebra F) :
  (L1 L2 : List F) → L1.length = L2.length →
  List.zipWith (fun x y => sa.mul (sa.add x y) sa.scale)
    (List.zipWith (fun a b => sa.mul (sa.sub a b) sa.scale) L1 L2)
    (List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) L1 L2) = L1 ∧
  List.zipWith (fun x y => sa.mul (sa.sub y x) sa.scale)
    (List.zipWith (fun a b => sa.mul (sa.sub a b) sa.scale) L1 L2)
    (List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) L1 L2) = L2
| [], [], _ => ⟨Eq.refl [], Eq.refl []⟩
| a :: as, b :: bs, h =>
  let h_len : as.length = bs.length := succ_inj h
  let rec_res := zipWith_inverse_list sa as bs h_len
  let pair_res := prove_inverse_pair sa a b
  ⟨congr_cons pair_res.left rec_res.left,
   congr_cons pair_res.right rec_res.right⟩
| _ :: _, [], h => False.elim (succ_not_zero _ h)
| [], _ :: _, h => False.elim (zero_not_succ _ h)

def zipWith_inverse_list_rev (sa : ScaleAlgebra F) :
  (L1 L2 : List F) → L1.length = L2.length →
  List.zipWith (fun x y => sa.mul (sa.sub x y) sa.scale)
    (List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) L1 L2)
    (List.zipWith (fun a b => sa.mul (sa.sub b a) sa.scale) L1 L2) = L1 ∧
  List.zipWith (fun x y => sa.mul (sa.add x y) sa.scale)
    (List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) L1 L2)
    (List.zipWith (fun a b => sa.mul (sa.sub b a) sa.scale) L1 L2) = L2
| [], [], _ => ⟨Eq.refl [], Eq.refl []⟩
| u :: us, v :: vs, h =>
  let h_len : us.length = vs.length := succ_inj h
  let rec_res := zipWith_inverse_list_rev sa us vs h_len
  let pair_res := prove_inverse_pair_rev sa u v
  ⟨congr_cons pair_res.left rec_res.left,
   congr_cons pair_res.right rec_res.right⟩
| _ :: _, [], h => False.elim (succ_not_zero _ h)
| [], _ :: _, h => False.elim (zero_not_succ _ h)

def forwardCore (sa : ScaleAlgebra F) (dim : Nat) (L : List F) : List F :=
  let x1 := (split_at dim L).1
  let x2 := (split_at dim L).2
  let x1_new := List.zipWith (fun a b => sa.mul (sa.sub a b) sa.scale) x1 x2
  let x2_new := List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) x1 x2
  x1_new ++ x2_new

def backwardCore (sa : ScaleAlgebra F) (dim : Nat) (L : List F) : List F :=
  let g1 := (split_at dim L).1
  let g2 := (split_at dim L).2
  let g1_new := List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) g1 g2
  let g2_new := List.zipWith (fun a b => sa.mul (sa.sub b a) sa.scale) g1 g2
  g1_new ++ g2_new

theorem oftb_invertible (sa : ScaleAlgebra F) (dim : Nat) (L : List F) (h : L.length = dim + dim) :
  backwardCore sa dim (forwardCore sa dim L) = L :=
  let x1 := (split_at dim L).1
  let x2 := (split_at dim L).2
  let h_splits : x1.length = dim ∧ x2.length = dim := split_at_lengths_gen dim dim L h
  let x1_new := List.zipWith (fun a b => sa.mul (sa.sub a b) sa.scale) x1 x2
  let x2_new := List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) x1 x2
  let h_x1_new_len : x1_new.length = dim :=
    Eq.trans (length_zipWith (fun a b => sa.mul (sa.sub a b) sa.scale) x1 x2 (Eq.trans h_splits.left (Eq.symm h_splits.right))) h_splits.left
  let h_split_new : split_at dim (x1_new ++ x2_new) = (x1_new, x2_new) :=
    split_at_append dim x1_new x2_new h_x1_new_len
  let h_zip_len : x1.length = x2.length := Eq.trans h_splits.left (Eq.symm h_splits.right)
  let inv_res := zipWith_inverse_list sa x1 x2 h_zip_len
  let h_g1_new : List.zipWith (fun x y => sa.mul (sa.add x y) sa.scale) x1_new x2_new = x1 := inv_res.left
  let h_g2_new : List.zipWith (fun x y => sa.mul (sa.sub y x) sa.scale) x1_new x2_new = x2 := inv_res.right
  let step1 : backwardCore sa dim (forwardCore sa dim L) =
    let (g1, g2) := split_at dim (x1_new ++ x2_new);
    List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) g1 g2 ++
    List.zipWith (fun a b => sa.mul (sa.sub b a) sa.scale) g1 g2 := Eq.refl _
  let step2 : (let (g1, g2) := split_at dim (x1_new ++ x2_new);
    List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) g1 g2 ++
    List.zipWith (fun a b => sa.mul (sa.sub b a) sa.scale) g1 g2) =
    (List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) x1_new x2_new ++
     List.zipWith (fun a b => sa.mul (sa.sub b a) sa.scale) x1_new x2_new) :=
    congrArg (fun p =>
      List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) p.1 p.2 ++
      List.zipWith (fun a b => sa.mul (sa.sub b a) sa.scale) p.1 p.2) h_split_new
  let step3 : (List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) x1_new x2_new ++
     List.zipWith (fun a b => sa.mul (sa.sub b a) sa.scale) x1_new x2_new) =
    x1 ++ x2 :=
    congr_append h_g1_new h_g2_new
  let step4 : x1 ++ x2 = L := split_at_reconstruct dim L
  Eq.trans step1 (Eq.trans step2 (Eq.trans step3 step4))

theorem oftb_invertible_rev (sa : ScaleAlgebra F) (dim : Nat) (L : List F) (h : L.length = dim + dim) :
  forwardCore sa dim (backwardCore sa dim L) = L :=
  let g1 := (split_at dim L).1
  let g2 := (split_at dim L).2
  let h_splits : g1.length = dim ∧ g2.length = dim := split_at_lengths_gen dim dim L h
  let g1_new := List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) g1 g2
  let g2_new := List.zipWith (fun a b => sa.mul (sa.sub b a) sa.scale) g1 g2
  let h_g1_new_len : g1_new.length = dim :=
    Eq.trans (length_zipWith (fun a b => sa.mul (sa.add a b) sa.scale) g1 g2 (Eq.trans h_splits.left (Eq.symm h_splits.right))) h_splits.left
  let h_split_new : split_at dim (g1_new ++ g2_new) = (g1_new, g2_new) :=
    split_at_append dim g1_new g2_new h_g1_new_len
  let h_zip_len : g1.length = g2.length := Eq.trans h_splits.left (Eq.symm h_splits.right)
  let inv_res := zipWith_inverse_list_rev sa g1 g2 h_zip_len
  let h_x1_new : List.zipWith (fun a b => sa.mul (sa.sub a b) sa.scale) g1_new g2_new = g1 := inv_res.left
  let h_x2_new : List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) g1_new g2_new = g2 := inv_res.right
  let step1 : forwardCore sa dim (backwardCore sa dim L) =
    let (x1, x2) := split_at dim (g1_new ++ g2_new);
    List.zipWith (fun a b => sa.mul (sa.sub a b) sa.scale) x1 x2 ++
    List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) x1 x2 := Eq.refl _
  let step2 : (let (x1, x2) := split_at dim (g1_new ++ g2_new);
    List.zipWith (fun a b => sa.mul (sa.sub a b) sa.scale) x1 x2 ++
    List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) x1 x2) =
    (List.zipWith (fun a b => sa.mul (sa.sub a b) sa.scale) g1_new g2_new ++
     List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) g1_new g2_new) :=
    congrArg (fun p =>
      List.zipWith (fun a b => sa.mul (sa.sub a b) sa.scale) p.1 p.2 ++
      List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) p.1 p.2) h_split_new
  let step3 : (List.zipWith (fun a b => sa.mul (sa.sub a b) sa.scale) g1_new g2_new ++
     List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) g1_new g2_new) =
    g1 ++ g2 :=
    congr_append h_x1_new h_x2_new
  let step4 : g1 ++ g2 = L := split_at_reconstruct dim L
  Eq.trans step1 (Eq.trans step2 (Eq.trans step3 step4))

def ValidShape : List Nat → Prop
| [] => False
| [x] => x > 0
| x :: xs => x > 0 ∧ ValidShape xs

def shapeProd : List Nat → Nat
| [] => 1
| x :: xs => x * shapeProd xs

structure Tensor (F : Type) where
  shape : List Nat
  data : List F
  h_valid : ValidShape shape
  h_len : data.length = shapeProd shape

def init_tensor_spec (F : Type) (shape : List Nat) (data : List F) : Prop :=
  ValidShape shape ∧ data.length = shapeProd shape

theorem tensor_init_rejects_invalid_shape (shape : List Nat) (h_invalid : ¬ ValidShape shape) :
  (d : List F) → ¬ init_tensor_spec F shape d :=
  fun _ h_spec => h_invalid h_spec.left

theorem valid_shape_empty_is_false : ¬ ValidShape [] :=
  fun h => h

theorem valid_shape_zero_is_false : (xs : List Nat) → ¬ ValidShape (0 :: xs)
| [] => fun h => Nat.lt_irrefl 0 h
| _ :: _ => fun h => Nat.lt_irrefl 0 h.left

def usize_max : Nat := 18446744073709551615

def ValidDim (dim : Nat) : Prop :=
  dim > 0 ∧ dim ≤ usize_max / 2

def ValidLen (dim : Nat) (n : Nat) : Prop :=
  n = dim + dim

def transform_precondition (dim : Nat) (L : List F) : Prop :=
  ValidDim dim ∧ L.length = dim + dim

theorem oftb_transform_mismatched_size (dim : Nat) (L : List F) (h_mismatch : L.length ≠ dim + dim) :
  ¬ transform_precondition dim L :=
  fun h => h_mismatch h.right

def mixForwardCore (sa : ScaleAlgebra F) (dim : Nat) (L : List F) (_ : transform_precondition dim L) : List F :=
  forwardCore sa dim L

def mixBackwardCore (sa : ScaleAlgebra F) (dim : Nat) (L : List F) (_ : transform_precondition dim L) : List F :=
  backwardCore sa dim L

theorem length_forwardCore (sa : ScaleAlgebra F) (dim : Nat) (L : List F) (h : L.length = dim + dim) :
  (forwardCore sa dim L).length = dim + dim :=
  let x1 := (split_at dim L).1
  let x2 := (split_at dim L).2
  let h_splits : x1.length = dim ∧ x2.length = dim := split_at_lengths_gen dim dim L h
  let x1_new := List.zipWith (fun a b => sa.mul (sa.sub a b) sa.scale) x1 x2
  let x2_new := List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) x1 x2
  let h_x1_new_len : x1_new.length = dim :=
    Eq.trans (length_zipWith (fun a b => sa.mul (sa.sub a b) sa.scale) x1 x2 (Eq.trans h_splits.left (Eq.symm h_splits.right))) h_splits.left
  let h_x2_new_len : x2_new.length = dim :=
    Eq.trans (length_zipWith (fun a b => sa.mul (sa.add a b) sa.scale) x1 x2 (Eq.trans h_splits.left (Eq.symm h_splits.right))) h_splits.left
  let h_append := length_append x1_new x2_new
  let step1 : (x1_new ++ x2_new).length = x1_new.length + x2_new.length := h_append
  let step2 : x1_new.length + x2_new.length = dim + dim :=
    congr_add h_x1_new_len h_x2_new_len
  Eq.trans step1 step2

theorem length_backwardCore (sa : ScaleAlgebra F) (dim : Nat) (L : List F) (h : L.length = dim + dim) :
  (backwardCore sa dim L).length = dim + dim :=
  let g1 := (split_at dim L).1
  let g2 := (split_at dim L).2
  let h_splits : g1.length = dim ∧ g2.length = dim := split_at_lengths_gen dim dim L h
  let g1_new := List.zipWith (fun a b => sa.mul (sa.add a b) sa.scale) g1 g2
  let g2_new := List.zipWith (fun a b => sa.mul (sa.sub b a) sa.scale) g1 g2
  let h_g1_new_len : g1_new.length = dim :=
    Eq.trans (length_zipWith (fun a b => sa.mul (sa.add a b) sa.scale) g1 g2 (Eq.trans h_splits.left (Eq.symm h_splits.right))) h_splits.left
  let h_g2_new_len : g2_new.length = dim :=
    Eq.trans (length_zipWith (fun a b => sa.mul (sa.sub b a) sa.scale) g1 g2 (Eq.trans h_splits.left (Eq.symm h_splits.right))) h_splits.left
  let h_append := length_append g1_new g2_new
  let step1 : (g1_new ++ g2_new).length = g1_new.length + g2_new.length := h_append
  let step2 : g1_new.length + g2_new.length = dim + dim :=
    congr_add h_g1_new_len h_g2_new_len
  Eq.trans step1 step2

theorem mix_round_trip (sa : ScaleAlgebra F) (dim : Nat) (L : List F) (h : transform_precondition dim L) :
  let h_forward_pre : transform_precondition dim (mixForwardCore sa dim L h) :=
    ⟨h.left, length_forwardCore sa dim L h.right⟩
  mixBackwardCore sa dim (mixForwardCore sa dim L h) h_forward_pre = L :=
  oftb_invertible sa dim L h.right

theorem mix_round_trip_rev (sa : ScaleAlgebra F) (dim : Nat) (L : List F) (h : transform_precondition dim L) :
  let h_backward_pre : transform_precondition dim (mixBackwardCore sa dim L h) :=
    ⟨h.left, length_backwardCore sa dim L h.right⟩
  mixForwardCore sa dim (mixBackwardCore sa dim L h) h_backward_pre = L :=
  oftb_invertible_rev sa dim L h.right
