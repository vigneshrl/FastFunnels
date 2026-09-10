#!/bin/bash
# Two domain-randomised pools for the centred-obstacle retrain (v8).
#
#   v8_centred : NO wall obstacles, everything on or within 0.4 m of the
#                centreline, biased LARGE. This is the neck_escalate
#                distribution -- the patch must pick a side and be narrow
#                enough to take it.
#   v8_curric  : the same but the lateral offset sweeps 0 -> 2.6 m across the
#                pool, so the easy side-placed case v7a already solves (87%) is
#                still in the mixture and the hard centred case is reachable by
#                gradient rather than a cliff. A few wall obstacles are kept so
#                the old skill is not forgotten.
#
# Both vary counts, size mix, spacing and station so no single layout can be
# memorised; the trainer samples a fresh variant every episode.
cd /p/cral/vignesh/bigtemp_files/FastFunnels/.claude/worktrees/ring-formation || exit 1
unset DISPLAY
export QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg
PY=/p/cral/vignesh/envs/fastfunnels/bin/python
ROOT=/p/cral/vignesh/bigtemp_files/FastFunnels/maps
N=${1:-300}
JOBS=${2:-14}
mkdir -p $ROOT/train_pool_v8_centred $ROOT/train_pool_v8_curric

gen_centred() {
  s=$1
  nc=$(( 5 + (s * 7) % 16 ))                                        # 5..20 centre blocks
  fl=$(python3 -c "print(round(0.45 + ((${s}*13)%45)/100.0, 2))")   # 0.45..0.89 large
  sp=$(python3 -c "print(round(3.0 + ((${s}*11)%22)/10.0, 1))")     # 3.0..5.1 m apart
  lj=$(python3 -c "print(round(((${s}*17)%41)/100.0, 2))")          # 0.00..0.40 jitter
  /p/cral/vignesh/envs/fastfunnels/bin/python -u presentation_code/make_cluttered_map.py \
      --base on_obs_ext --name ce_$(printf %04d $s) \
      --n-wall 0 --n-centre $nc --frac-large $fl \
      --s-spacing $sp --lateral-jitter $lj --pass-min 2.5 \
      --seed $s --out $ROOT/train_pool_v8_centred > /dev/null 2>&1
}

gen_curric() {
  s=$1
  nw=$(( (s * 5) % 9 ))                                             # 0..8 wall blocks
  nc=$(( 5 + (s * 7) % 16 ))                                        # 5..20 centre blocks
  fl=$(python3 -c "print(round(0.35 + ((${s}*13)%50)/100.0, 2))")   # 0.35..0.84 large
  sp=$(python3 -c "print(round(3.0 + ((${s}*11)%22)/10.0, 1))")
  # offset sweeps the full easy->hard range across the pool
  lj=$(python3 -c "print(round(((${s}*23)%27)/10.0, 2))")           # 0.0..2.6 m jitter
  /p/cral/vignesh/envs/fastfunnels/bin/python -u presentation_code/make_cluttered_map.py \
      --base on_obs_ext --name cu_$(printf %04d $s) \
      --n-wall $nw --n-centre $nc --frac-large $fl \
      --s-spacing $sp --lateral-jitter $lj --pass-min 2.5 \
      --seed $((s + 5000)) --out $ROOT/train_pool_v8_curric > /dev/null 2>&1
}
export -f gen_centred gen_curric
export ROOT

echo "[pool] generating $N centred variants ..."
seq 0 $((N-1)) | xargs -P $JOBS -I{} bash -c 'gen_centred {}'
echo "[pool] v8_centred: $(ls -d $ROOT/train_pool_v8_centred/ce_* 2>/dev/null | wc -l) variants"

echo "[pool] generating $N curriculum variants ..."
seq 0 $((N-1)) | xargs -P $JOBS -I{} bash -c 'gen_curric {}'
echo "[pool] v8_curric:  $(ls -d $ROOT/train_pool_v8_curric/cu_* 2>/dev/null | wc -l) variants"
echo "POOLV8 DONE"
