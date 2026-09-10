"""Write variants_index.json for the v8 pools (the trainer requires it)."""
import json, pathlib, sys

ROOT = pathlib.Path("/p/cral/vignesh/bigtemp_files/FastFunnels/maps")

for pool_name, prefix in (("train_pool_v8_centred", "ce_"),
                          ("train_pool_v8_curric", "cu_")):
    pool = ROOT / pool_name
    if not pool.exists():
        print(f"{pool_name}: MISSING, skipped")
        continue
    variants = []
    for d in sorted(pool.glob(prefix + "*")):
        if not d.is_dir():
            continue
        name = d.name
        # a variant is only usable if its occupancy grid was actually written
        if not (d / f"{name}_map.pgm").exists():
            continue
        rec = {"name": name, "base_map": "on_obs_ext"}
        idx = pool / f"{name}_index.json"
        if idx.exists():
            try:
                j = json.load(open(idx))
                rec.update(n_obstacles=j.get("n_obstacles"),
                           n_wall=j.get("n_wall"), n_centre=j.get("n_centre"),
                           n_large=j.get("n_large"),
                           gap_width_m=j.get("gap_width_m"))
            except Exception:
                pass
        variants.append(rec)
    with open(pool / "variants_index.json", "w") as f:
        json.dump({"base_maps": ["on_obs_ext"], "variants": variants}, f, indent=1)
    gaps = [v["gap_width_m"] for v in variants if v.get("gap_width_m")]
    obs = [v["n_obstacles"] for v in variants if v.get("n_obstacles")]
    print(f"{pool_name}: {len(variants)} variants"
          + (f", obstacles {min(obs)}-{max(obs)} (mean {sum(obs)/len(obs):.1f})" if obs else "")
          + (f", tightest lane {min(gaps):.2f} m" if gaps else ""))
