# GCBF+ narrow-zone baseline

Runs the **pretrained** GCBF+ policy from [MIT-REALM/gcbfplus](https://github.com/MIT-REALM/gcbfplus)
through the same narrow-zone corridor layouts that produced `fig_zone_clearance.png`,
inside a PyRoboSim world built from each map. **Nothing is trained** — the
DubinsCar checkpoint shipped in `gcbfplus/pretrained/DubinsCar/gcbf+` (step 1000)
is loaded and evaluated as-is.

## Running it

```bash
PY=/p/cral/vignesh/envs/gcbf_eval/bin/python

$PY -m gcbf_baseline.run_gcbf_eval \
    --split train:eval_maps_matched:50 \
    --split heldout:eval_maps_heldout:50 \
    --out gcbf_baseline/results/gcbf_zone_clearance.jsonl

$PY -m gcbf_baseline.make_gcbf_figure \
    --gcbf gcbf_baseline/results/gcbf_zone_clearance.jsonl \
    --nominal gcbf_baseline/results/nominal_zone_clearance.jsonl \
    --out fig_zone_clearance_gcbf.png
```

`--policy u_ref` swaps the learned actor for the environment's nominal
goal-tracking PID with no safety filter — the reference point that separates
"the corridor is hard" from "the CBF is conservative".

Single-map pictures, for checking the geometry conversion or for a qualitative
panel:

```bash
$PY -m gcbf_baseline.plot_corridor eval_maps_matched/onv0000            # geometry
$PY -m gcbf_baseline.plot_corridor eval_maps_matched/onv0000 \
    --trace traces/onv0000.npy                                          # a rollout
```

## Results

50 train + 50 held-out maps, 4 agents, `r = 0.29 m`, pretrained DubinsCar
checkpoint at step 1000, no fine-tuning. `results/*.jsonl` holds the per-map rows.

| | narrow-zone clearance | 95% Wilson | per-agent | collisions | stalls |
|---|---|---|---|---|---|
| **GCBF+**, randomised train | 29/50 = 58.0% | [44.2, 70.6] | 88.5% | 0/50 | 21 |
| **GCBF+**, randomised HELD OUT | 33/50 = 66.0% | [52.2, 77.6] | 90.5% | 1/50 | 16 |
| nominal PID, train | 0/50 | [0.0, 7.1] | 0% | 35/50 | 15 |
| nominal PID, HELD OUT | 0/50 | [0.0, 7.1] | 0% | 42/50 | 8 |
| FastFunnels, train | 49/50 = 98% | | | | |
| FastFunnels, HELD OUT | 48/50 = 96% | | | | |

The shape of the baseline's failure is very consistent: of the 38 maps it did
not clear, **34 ended with exactly three of the four agents through** and four
ended with two through. It is not that the formation cannot fit the gap — it is
that one agent, usually the one pushed to the outside of the bend by an
obstacle, stalls in the shoulder at the funnel mouth and never recovers. The
learned actor emits exactly zero action while several metres from its goal.
`results/qual_stalled_hld0002.png` shows a representative case;
`results/qual_cleared_hld0000.png` shows a clean pass.

Cleared runs take a median of 782 steps (23.5 s). The single collision in 100
maps is `hld0001`, the one map whose pinch sits at the very end of the track
(see `exit_clamped` below).

## How a map becomes a GCBF+ problem

| stage | file | what happens |
|---|---|---|
| occupancy → polygon | `corridor_geom.py` | the free component containing the centreline is contour-traced into a shapely polygon; obstacles rasterised clear of the walls come back as holes. Walls are bit-identical across every variant, so the trace is memoised. |
| polygon → world | `pyrobosim_world.py` | one PyRoboSim room whose footprint *is* that polygon. Collision is PyRoboSim's own `check_occupancy` against `world.total_internal_polygon` — the room deflated by the robot radius. |
| polygon → obstacles | `corridor_geom.wall_rects` | every boundary segment near the scored stretch becomes one oriented rectangle, pushed to its occupied side so its inner face sits on the true wall. GCBF+ only understands rectangles. |
| metres → GCBF+ units | `gcbf_corridor_env.py` | `scale = 0.05 / robot_radius_m`, so an agent is exactly the disc the pretrained policy expects and every distance keeps its true ratio to the robot. Time is not rescaled. |

Nothing under `gcbfplus/` is modified.

## Scoring

A map is **cleared** when every agent reaches centreline waypoint
`narrow_wp + 8` without colliding — the same exit test `make_rebuttal_figure.py`
and `record_patch_generalization.py` apply to the funnel policy. The runner also
records per-agent progress, first collision, minimum wall clearance and the
termination reason, so softer readings (per-agent clearance, clearance ignoring
collisions) can be recomputed from the JSONL without re-running.

`--on-collision revert` reproduces the hand-authored PyRoboSim world's behaviour
instead: the offending agent is put back and the run continues. It does not
change the picture — the failure mode is stalling, not crashing.

Two caveats are recorded per map rather than hidden:

- **`exit_clamped`** — a couple of variants put their tightest centreline point
  within a few waypoints of the end of the track, where the centreline runs out
  past the walls. Those are scored on the last waypoint actually inside the
  corridor, so the exit test is a little shorter than `narrow_wp + 8`. The flag
  says which maps those are.
- **Formation fit** — the entrance rank collapses into successive ranks wherever
  the corridor is too tight to hold it abreast, so a map is never failed just
  because four discs do not fit side by side.

## Tuning, and what it revealed

The defaults were tuned on `eval_maps_matched` only, 12 maps per configuration.
Held-out maps were run once, after the configuration was frozen. Clearance on
those 12 train maps:

| configuration | cleared | per-agent | collisions |
|---|---|---|---|
| `--lookahead-wp 1 --spacing 1.2` (**default**) | 10/12 | 96% | 0 |
| `--lookahead-wp 1 --spacing 1.1` | 10/12 | 96% | 0 |
| `--lookahead-wp 1 --spacing 0.9` | 10/12 | 96% | 0 |
| `--lookahead-wp 1 --spacing 1.0` | 9/12 | 94% | 0 |
| `--lookahead-wp 1 --spacing 1.4` | 2/12 | 79% | 0 |
| `--lookahead-wp 1 --spacing 1.6` | 3/12 | 81% | 0 |
| `--lookahead-wp 2 --spacing 1.2` | 0/12 | 73% | 7 |
| `--lookahead-wp 3 --spacing 1.2` | 1/12 | 56% | 0 |
| `--lookahead-wp 6 --spacing 1.2` | 0/12 | 48% | 0 |
| `--policy u_ref --lookahead-wp 6` (no safety filter) | 0/12 | 0% | 12 |

The 12-map tuning sample is optimistic, as small tuning samples are: the chosen
configuration scored 10/12 there and 29/50 on the full train split. Quote the
full-split numbers.

Two things are worth carrying into the write-up:

- **The baseline is very sensitive to how far ahead the reference sits.** Past a
  carrot of roughly 2.5 m, agents cut the corner into the bend's outer wall,
  the CBF halts them, and they never recover — the actor emits exactly zero
  action several metres from its goal. This is a liveness failure, not a safety
  one, and it is worth reporting as such rather than quoting the untuned number.
- **The safety filter is doing real work.** With the same reference and no CBF
  (`--policy u_ref`) the rank reaches the pinch and drives into the wall on
  every map. GCBF+ never collides on any map, in any configuration tried.

## Choices that affect the numbers

- **`--robot-radius`** (default 0.29 m) sets the whole scale. 0.29 m circumscribes
  an f1tenth car; 0.15 m matches the disc radius in the hand-authored PyRoboSim
  script and makes the corridor relatively wider.
- **`--goal-mode carrot`** (default) hands GCBF+ a receding centreline waypoint,
  i.e. a global planner supplies the reference and GCBF+ only has to keep the
  agents safe. This is the *strong* reading of the baseline. `fixed` gives it one
  goal past the zone and makes it find the way itself.
- **`--spacing`** is the lateral pitch of the entrance rank. Wider ranks put the
  outer agents against the walls through the bend and stall them; below about
  `2 x robot_radius + 0.3 m` the rank starts inside GCBF+'s own inter-agent
  warning band and the agents shove each other into the walls. The runner refuses
  spacings that overlap outright.
