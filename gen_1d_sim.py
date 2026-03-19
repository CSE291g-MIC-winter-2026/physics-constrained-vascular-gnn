#!/usr/bin/env python3
"""
gen_1d_sim.py — Convert processed CORO_KD graph → svOneDSolver 1D simulation

Reads:
  - data/processed/{patient}/{patient}_all.pt   (geometry from our GNN graph)
  - data/raw/{patient}/flow-files/inflow_1d.flow (inlet BC)
  - data/raw/{patient}/Simulations/*/rcrt.dat    (RCR outlet BCs)

Writes:
  - {output}/{patient}.in   → svOneDSolver input
  - {output}/results_*.dat  → simulation results (after running solver)

Usage:
  python3 gen_1d_sim.py \
    --patient 0074_H_CORO_KD \
    --raw_dir /mnt/h/CSE291/data/raw \
    --processed_dir /mnt/h/CSE291/data/processed \
    --output /projects/my_runs/0074 \
    --solver /projects/svOneDSolver_build/bin/OneDSolver
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path
from collections import deque

import numpy as np
import torch

# ── Physics constants (CGS) ───────────────────────────────────────────────────
MU    = 0.04          # blood viscosity  [g/(cm·s)]
RHO   = 1.06          # blood density    [g/cm³]
REF_P = 113324.0      # reference press. [dyne/cm²] = 85 mmHg
K1, K2, K3 = 2.0e7, -22.5267, 8.65e5   # Olufsen wall material


# ── Graph helpers ─────────────────────────────────────────────────────────────
def load_graph(pt_path):
    graphs = torch.load(str(pt_path), weights_only=False)
    return graphs[0]   # first snapshot = geometry only


def build_adj(graph):
    """Undirected adjacency + edge props {(lo,hi): (avg_r_cm, length_cm)}."""
    N  = graph.num_nodes
    ei = graph.edge_index.numpy()
    ea = graph.edge_attr.numpy()          # [E, 2]  col0=avg_r  col1=length
    adj   = {i: set() for i in range(N)}
    props = {}
    for e in range(ei.shape[1]):
        s, d = int(ei[0, e]), int(ei[1, e])
        adj[s].add(d); adj[d].add(s)
        key = (min(s, d), max(s, d))
        if key not in props:
            props[key] = (float(ea[e, 0]), float(ea[e, 1]))
    return adj, props


def find_largest_component(N, adj):
    """Return list of node indices in the largest connected component."""
    visited = set()
    components = []
    for start in range(N):
        if start in visited:
            continue
        comp = []
        q = deque([start])
        visited.add(start)
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        components.append(comp)
    return max(components, key=len)


def find_inlet(graph, adj):
    """Inlet = leaf node (degree 1) with highest Z, restricted to largest component."""
    coords  = graph.x[:, :3].numpy()
    largest = set(find_largest_component(graph.num_nodes, adj))
    leaves  = [i for i in largest if len(adj[i]) == 1]
    if not leaves:
        leaves = list(largest)   # fallback: no degree-1 nodes
    return max(leaves, key=lambda n: coords[n, 2])


def bfs_tree(N, adj, props, root):
    """BFS from root → directed edges list [(src, dst, r_cm, L_cm)],
       children dict {node: [child, ...]}.
    """
    visited  = {root}
    queue    = deque([root])
    edges    = []
    children = {i: [] for i in range(N)}
    while queue:
        u = queue.popleft()
        for v in sorted(adj[u]):
            if v not in visited:
                visited.add(v)
                key = (min(u, v), max(u, v))
                r, L = props.get(key, (0.1, 1.0))
                edges.append((u, v, r, L))
                children[u].append(v)
                queue.append(v)
    return edges, children


# ── Boundary condition parsers ────────────────────────────────────────────────
def parse_inflow(flow_file):
    """→ (period_s, ndarray [N,2] columns=[time_s, flow_cm3/s])"""
    data   = np.loadtxt(str(flow_file))
    period = float(data[-1, 0] - data[0, 0])
    return period, data


def parse_rcrt(rcrt_file):
    """Parse SimVascular rcrt.dat → list of (Rp, C, Rd)."""
    text  = Path(rcrt_file).read_text()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    outlets = []
    i = 0
    while i < len(lines):
        # Each RCR block starts with "2" (number of resistance params indicator)
        try:
            val = int(lines[i]); i += 1
        except ValueError:
            i += 1; continue
        if val == 2 and i + 2 < len(lines):
            try:
                Rp = float(lines[i]); i += 1
                C  = float(lines[i]); i += 1
                Rd = float(lines[i]); i += 1
                # skip trailing pressure-time table rows (pairs of floats)
                while i < len(lines):
                    parts = lines[i].split()
                    if len(parts) == 2:
                        try:
                            float(parts[0]); float(parts[1]); i += 1; continue
                        except ValueError:
                            break
                    break
                outlets.append((Rp, C, Rd))
            except (ValueError, IndexError):
                pass
    return outlets


# ── Write svOneDSolver .in file ───────────────────────────────────────────────
def write_input(out_file, model_name, graph,
                directed_edges, children, inlet_node,
                inflow_data, period, rcrt_list, n_cycles=3):
    """
    We use RESISTANCE (Rp+Rd) outlets — simpler than full RCR but avoids
    needing to know the exact RCR datatable format.  Rp+Rd lumped resistance
    still gives physically consistent pulsatile P and v.
    """
    coords = graph.x[:, :3].numpy()
    radii  = graph.x[:, 3].numpy()
    N      = graph.num_nodes

    seg_id_of  = {(s, d): idx for idx, (s, d, r, L) in enumerate(directed_edges)}
    outlet_set = {i for i in range(N) if len(children[i]) == 0}  # leaves
    outlet_set -= {inlet_node}   # inlet is not an outlet

    # ── Assign resistance to each outlet (Rp+Rd from rcrt.dat, by area desc) ──
    outlet_seg_info = []
    for idx, (s, d, r, L) in enumerate(directed_edges):
        if d in outlet_set:
            A_out = np.pi * max(radii[d], 1e-4) ** 2
            outlet_seg_info.append((idx, d, A_out))
    outlet_seg_info.sort(key=lambda x: -x[2])   # largest vessel first

    avg_R = sum(v[0] + v[2] for v in rcrt_list) / len(rcrt_list) if rcrt_list else 10000.0
    rcrt_pad = rcrt_list + [(1000.0, 1e-4, 8000.0)] * max(0, len(outlet_seg_info) - len(rcrt_list))
    seg_R = {}   # seg_idx → total resistance (Rp + Rd)
    for rank, (seg_idx, node, area) in enumerate(outlet_seg_info):
        Rp, C, Rd = rcrt_pad[rank]
        seg_R[seg_idx] = Rp + Rd

    total_t    = n_cycles * period
    dt         = period / 1000.0
    total_steps = int(round(total_t / dt))
    save_every  = max(1, total_steps // 200)   # ~200 saves total

    with open(str(out_file), 'w') as f:

        # ── MODEL ──
        f.write(f"MODEL {model_name}\n\n")

        # ── NODEs ──
        f.write("# === NODES ===\n")
        for i in range(N):
            x, y, z = coords[i]
            f.write(f"NODE {i} {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")

        # ── JOINTs (only at bifurcation / trifurcation nodes) ──
        f.write("# === JOINTS ===\n")
        jnt = 0
        for node in range(N):
            ch = children[node]
            if len(ch) < 2:
                continue
            in_segs  = [seg_id_of[(s, d)] for s, d, r, L in directed_edges if d == node]
            out_segs = [seg_id_of[(node, c)] for c in ch if (node, c) in seg_id_of]
            jn = f"J{jnt}"; jnt += 1
            f.write(f"JOINT {jn} {node} {jn}_IN {jn}_OUT\n")
            f.write(f"JOINTINLET  {jn}_IN  {len(in_segs)}  " + " ".join(map(str, in_segs))  + "\n")
            f.write(f"JOINTOUTLET {jn}_OUT {len(out_segs)} " + " ".join(map(str, out_segs)) + "\n")
        f.write("\n")

        # ── SEGMENTs ──
        f.write("# === SEGMENTS ===\n")
        for idx, (s, d, r, L) in enumerate(directed_edges):
            n_elem = max(5, int(L / 0.2))          # ~1 element per 2 mm
            A_s = max(1e-8, np.pi * radii[s] ** 2)
            A_d = max(1e-8, np.pi * radii[d] ** 2)
            if d in outlet_set:
                bc_type = "RESISTANCE"
                dt_name = f"R_TAB_{idx}"
            else:
                bc_type = "NOBOUND"
                dt_name = "NONE"
            f.write(f"SEGMENT SEG{idx} {idx} {L:.6f} {n_elem} "
                    f"{s} {d} {A_s:.8f} {A_d:.8f} 0.0 "
                    f"MAT1 NONE 0.0 0 0 {bc_type} {dt_name}\n")
        f.write("\n")

        # ── DATATABLEs ──
        f.write("# === DATATABLES ===\n")

        # Inlet flow (repeated for n_cycles to reach steady oscillation)
        f.write("DATATABLE INLETDATA LIST\n")
        t0 = float(inflow_data[0, 0])
        for cyc in range(n_cycles):
            offset = cyc * period
            for row in inflow_data:
                t, q = float(row[0]) - t0 + offset, float(row[1])
                f.write(f"{t:.8e} {q:.8e}\n")
        f.write("ENDDATATABLE\n\n")

        # Resistance tables (constant resistance value)
        for idx, (s, d, r, L) in enumerate(directed_edges):
            if d in outlet_set:
                R = seg_R.get(idx, avg_R)
                f.write(f"DATATABLE R_TAB_{idx} LIST\n")
                f.write(f"0.0 {R:.4f}\n")
                f.write("ENDDATATABLE\n\n")

        # ── MATERIAL ──
        f.write(f"MATERIAL MAT1 OLUFSEN {RHO} {MU} {REF_P:.1f} "
                f"1.0 {K1:.3e} {K2} {K3:.3e}\n\n")

        # ── SOLVEROPTIONS ──
        f.write(f"SOLVEROPTIONS {dt:.8f} {save_every} {total_steps} "
                f"4 INLETDATA FLOW 1.0e-6 1 1\n\n")

        f.write("OUTPUT TEXT\n")

    return str(out_file)


# ── Parse results ─────────────────────────────────────────────────────────────
def parse_results(out_dir, model_name, directed_edges):
    """
    svOneDSolver TEXT output: results_{model}_seg{N}_node{0|1}.dat
    Columns: time  area  flow  pressure  ...
    Returns dict: seg_node_key → ndarray [T, 4] (t, A, Q, P)
    """
    import glob
    pattern = str(Path(out_dir) / f"results_{model_name}_seg*.dat")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No result files matching: {pattern}")

    results = {}
    for fpath in files:
        m = re.search(r'_seg(\d+)_node(\d+)\.dat', Path(fpath).name)
        if not m:
            continue
        seg_id  = int(m.group(1))
        node_pos = int(m.group(2))   # 0 = inlet-side, 1 = outlet-side of segment
        try:
            data = np.loadtxt(fpath, comments='#')
            if data.ndim == 1:
                data = data.reshape(1, -1)
            results[(seg_id, node_pos)] = data
        except Exception as e:
            print(f"  Warning: {fpath}: {e}")
    return results


def results_to_node_arrays(results, directed_edges, N):
    """
    Collapse seg/node results → per-graph-node arrays:
    node_P[n] = mean pressure (dyne/cm²) over last cycle, all time steps
    node_Q[n] = mean |flow| (cm³/s)
    """
    node_P  = {i: [] for i in range(N)}
    node_Q  = {i: [] for i in range(N)}

    for (seg_id, node_pos), data in results.items():
        if seg_id >= len(directed_edges):
            continue
        s, d, r, L = directed_edges[seg_id]
        graph_node = s if node_pos == 0 else d

        t = data[:, 0]
        # Use last cardiac cycle only (most converged)
        t_max = t.max()
        last_cycle_mask = t >= (t_max - t.max() / 3)

        # Columns vary by solver version: try col 3 for pressure, col 2 for flow
        try:
            Q_col = 2 if data.shape[1] > 2 else 1
            P_col = 3 if data.shape[1] > 3 else Q_col
            Q_vals = data[last_cycle_mask, Q_col]
            P_vals = data[last_cycle_mask, P_col]
            node_P[graph_node].extend(P_vals.tolist())
            node_Q[graph_node].extend(Q_vals.tolist())
        except IndexError:
            pass

    # Average
    mean_P = np.zeros(N)
    mean_Q = np.zeros(N)
    for i in range(N):
        if node_P[i]:
            mean_P[i] = np.mean(node_P[i])
        if node_Q[i]:
            mean_Q[i] = np.abs(np.mean(node_Q[i]))
    return mean_P, mean_Q


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate + run 1D simulation for a CORO_KD patient")
    ap.add_argument('--patient',       required=True,  help='e.g. 0074_H_CORO_KD')
    ap.add_argument('--raw_dir',       required=True,  help='path to data/raw/')
    ap.add_argument('--processed_dir', required=True,  help='path to data/processed/')
    ap.add_argument('--output',        required=True,  help='output directory')
    ap.add_argument('--solver',        default='/projects/svOneDSolver_build/bin/OneDSolver')
    ap.add_argument('--n_cycles',      type=int, default=3, help='cardiac cycles to simulate')
    ap.add_argument('--dry_run',       action='store_true', help='write .in file but do not run solver')
    args = ap.parse_args()

    raw_dir  = Path(args.raw_dir)  / args.patient
    proc_pt  = Path(args.processed_dir) / args.patient / f"{args.patient}_all.pt"
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Patient: {args.patient}")

    # 1. Geometry
    g = load_graph(proc_pt)
    print(f"  Graph: {g.num_nodes} nodes, {g.edge_index.shape[1]//2} edges (undirected)")
    adj, props = build_adj(g)

    # Report connected components
    from collections import Counter
    comp_nodes = find_largest_component(g.num_nodes, adj)
    all_comps = []
    visited_c = set()
    for start in range(g.num_nodes):
        if start in visited_c:
            continue
        q2 = deque([start]); visited_c.add(start); c = []
        while q2:
            u = q2.popleft(); c.append(u)
            for v in adj[u]:
                if v not in visited_c: visited_c.add(v); q2.append(v)
        all_comps.append(len(c))
    all_comps.sort(reverse=True)
    print(f"  Components: {len(all_comps)}  sizes={all_comps[:5]}")

    inlet = find_inlet(g, adj)
    print(f"  Inlet node: {inlet}  z={g.x[inlet,2]:.2f} cm  (largest component: {len(comp_nodes)} nodes)")
    d_edges, children = bfs_tree(g.num_nodes, adj, props, inlet)
    outlet_count = sum(1 for i in range(g.num_nodes)
                       if len(children[i]) == 0 and i != inlet)
    print(f"  Directed edges: {len(d_edges)}  |  Outlets: {outlet_count}")

    # 2. Boundary conditions
    inflow_file = raw_dir / 'flow-files' / 'inflow_1d.flow'
    if not inflow_file.exists():
        sys.exit(f"  ERROR: inflow file not found: {inflow_file}")
    period, inflow_data = parse_inflow(inflow_file)
    print(f"  Inflow: period={period:.3f}s  peak={inflow_data[:,1].max():.1f} cm³/s  "
          f"pts={len(inflow_data)}")

    rcrt_files = list((raw_dir / 'Simulations').glob('*/rcrt.dat'))
    rcrt_list  = parse_rcrt(str(rcrt_files[0])) if rcrt_files else []
    print(f"  RCR outlets parsed: {len(rcrt_list)}")

    # 3. Write input
    model_name = args.patient.replace('-', '_')
    in_file    = out_dir / f"{model_name}.in"
    write_input(in_file, model_name, g, d_edges, children, inlet,
                inflow_data, period, rcrt_list, n_cycles=args.n_cycles)
    print(f"  Input written: {in_file}")

    if args.dry_run:
        print("  [dry_run] Skipping solver. Done.")
        return

    # 4. Run solver
    print(f"\n  Running OneDSolver ({args.n_cycles} cycles × {period:.3f}s)...")
    proc = subprocess.run(
        [args.solver, str(in_file)],
        cwd=str(out_dir),
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        combined = (proc.stdout + proc.stderr)
        print("  SOLVER ERROR (last 2000 chars):\n", combined[-2000:])
        sys.exit(1)
    # Show last few lines of solver output
    last_lines = proc.stdout.strip().split('\n')[-5:]
    for line in last_lines:
        print(f"  {line}")

    # 5. Parse results
    try:
        res = parse_results(str(out_dir), model_name, d_edges)
        mean_P, mean_Q = results_to_node_arrays(res, d_edges, g.num_nodes)

        print(f"\n  Results summary:")
        print(f"    Pressure: {mean_P.min()/1333.22:.1f} – {mean_P.max()/1333.22:.1f} mmHg")
        print(f"    Flow:     {mean_Q.min():.2f} – {mean_Q.max():.2f} cm³/s")

        # Compute velocity = Q / A
        radii  = g.x[:, 3].numpy()
        areas  = np.pi * np.maximum(radii, 1e-4) ** 2
        mean_V = mean_Q / areas
        print(f"    Velocity: {mean_V.min():.1f} – {mean_V.max():.1f} cm/s")

        # Save
        np.savez(str(out_dir / 'cfd_labels.npz'),
                 pressure_dyne_cm2=mean_P,
                 flow_cm3_s=mean_Q,
                 velocity_cm_s=mean_V)
        print(f"  Labels saved → {out_dir}/cfd_labels.npz")

    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        print(f"  Check raw solver output in {out_dir}/")

    print('='*60 + '\n')


if __name__ == '__main__':
    main()
