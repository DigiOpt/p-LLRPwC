"""
llrp_pricer_nb.py
=================

Numba-compiled ng-route pricing core for the LLRP column generation, now with
subset-row cut (SRC) support.

Same label-setting algorithm as the pure-Python ``_price_depot`` (ng-memory,
tau/load/rc dominance, cumulative latency accumulated on arrival), compiled to
native code. Labels live in flat preallocated arrays; per-node dominance buckets
are singly-linked lists (``bucket_head`` + ``next_lab``) with lazy deletion.

SRC: each label carries an int64 ``sstate`` holding the parity (mod 2) of visits
to each active cut's customer set. When a visit crosses an even count the cut's
dual is applied to the reduced cost, and dominance uses the corrected rule
(Jepsen et al. 2008): A dominates B only if, in addition to memory/tau/load,
rc_A + sum_{cuts pending in A but not B} (-omega) <= rc_B.

Valid for n <= 64 (customer ng-memory bitmask) and up to 64 active cuts
(cut-parity bitmask).
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _src_pen(diff, omega):
    """Sum of -omega over set bits of ``diff`` (omega <= 0 -> penalty >= 0)."""
    pen = 0.0
    m = 0
    while diff:
        if diff & 1:
            pen -= omega[m]
        diff >>= 1
        m += 1
    return pen


@njit(cache=True, nogil=True)
def _insert(b, rcb, taub, lob, maskb, sstateb, predb,
            rc, tau, load, mask, sstate, node, pred, alive, next_lab,
            bucket_head, cnt, omega):
    """Insert a label at node b if not dominated; drop labels it dominates.
    Dominance includes the SRC penalty correction."""
    j = bucket_head[b]
    while j != -1:
        if alive[j]:
            if (mask[j] & maskb) == mask[j] and tau[j] <= taub + 1e-9 \
                    and load[j] <= lob + 1e-9:
                pen = _src_pen(sstate[j] & ~sstateb, omega)   # A=j pending, B=new not
                if rc[j] + pen <= rcb + 1e-9:
                    return cnt                                # new dominated
        j = next_lab[j]
    j = bucket_head[b]
    while j != -1:
        if alive[j] and (maskb & mask[j]) == maskb and taub <= tau[j] + 1e-9 \
                and lob <= load[j] + 1e-9:
            pen = _src_pen(sstateb & ~sstate[j], omega)
            if rcb + pen <= rc[j] + 1e-9:
                alive[j] = False                              # existing dominated
        j = next_lab[j]
    if cnt >= rc.shape[0]:
        return -1
    idx = cnt
    rc[idx] = rcb
    tau[idx] = taub
    load[idx] = lob
    mask[idx] = maskb
    sstate[idx] = sstateb
    node[idx] = b
    pred[idx] = predb
    alive[idx] = True
    next_lab[idx] = bucket_head[b]
    bucket_head[b] = idx
    return cnt + 1


@njit(cache=True, nogil=True)
def price_core(n, d, p, l_di, c_di, l_id, c_id, l_cc, c_cc, Q, L, F, wC, wP,
               pi, sig_i, rho_i, mu, ng, keep_threshold,
               omega, ccut_start, ccut_idx, fb_cc, fb_out, fb_in,
               rc, tau, load, mask, sstate, node, pred, alive, next_lab,
               bucket_head, out_lab, out_rcf):
    """Label-set ng-routes (with SRC + arc-branching forbids) for one depot.

    fb_cc[u,v]=1 forbids customer arc u->v; fb_out[a]=1 forbids depot->a;
    fb_in[a]=1 forbids a->depot (a cannot be a route's last stop).
    Returns (min_rcf, n_out, count, overflow)."""
    for a in range(n):
        bucket_head[a] = -1
    cnt = 0

    for a in range(n):                           # depot -> a
        if d[a] > Q or fb_out[a]:
            continue
        ta = l_di[a]
        if ta + l_id[a] > L + 1e-9:
            continue
        rc0 = wC * c_di[a] + wP * p[a] * ta - pi[a] - sig_i * d[a]
        m0 = np.int64(1) << a
        ss0 = np.int64(0)
        for k in range(ccut_start[a], ccut_start[a + 1]):   # first C-member: 0->1
            ss0 |= (np.int64(1) << ccut_idx[k])
        cnt = _insert(a, rc0, ta, d[a], m0, ss0, -1, rc, tau, load, mask, sstate,
                      node, pred, alive, next_lab, bucket_head, cnt, omega)
        if cnt == -1:
            return 0.0, 0, 0, 1

    head = 0
    while head < cnt:
        if alive[head]:
            a = node[head]
            rc_ = rc[head]
            ta = tau[head]
            lo = load[head]
            ms = mask[head]
            ss = sstate[head]
            for b in range(n):
                if (ms >> b) & 1:
                    continue
                if fb_cc[a, b]:                  # arc-branching forbid
                    continue
                nlo = lo + d[b]
                if nlo > Q:
                    continue
                nta = ta + l_cc[a, b]
                if nta + l_id[b] > L + 1e-9:
                    continue
                nrc = rc_ + wC * c_cc[a, b] + wP * p[b] * nta - pi[b] - sig_i * d[b]
                nss = ss
                for k in range(ccut_start[b], ccut_start[b + 1]):
                    m = ccut_idx[k]
                    bit = np.int64(1) << m
                    if nss & bit:                # crosses even -> apply dual
                        nrc -= omega[m]
                        nss &= ~bit
                    else:
                        nss |= bit
                nm = (ms & ng[b]) | (np.int64(1) << b)
                cnt = _insert(b, nrc, nta, nlo, nm, nss, head, rc, tau, load, mask,
                              sstate, node, pred, alive, next_lab, bucket_head, cnt, omega)
                if cnt == -1:
                    return 0.0, 0, head, 1
        head += 1

    min_rcf = 0.0
    n_out = 0
    for idx in range(cnt):
        if alive[idx]:
            if fb_in[node[idx]]:                  # a->depot forbidden: can't close here
                continue
            rcf = rc[idx] + wC * (c_id[node[idx]] + F) - rho_i - mu
            if rcf < min_rcf:
                min_rcf = rcf
            if rcf < keep_threshold and n_out < out_lab.shape[0]:
                out_lab[n_out] = idx
                out_rcf[n_out] = rcf
                n_out += 1
    return min_rcf, n_out, cnt, 0


@njit(cache=True, nogil=True)
def _el_insert(head, nxt, alive, RC, LOAD, TRES, VIS, cnt,
               b, rc, load, tres, vis, NODE, PRED, nd, pred):
    """Insert an ELEMENTARY label (rc, load, tres, vis-bitmask) at node ``b``'s
    bucket, with full-elementarity dominance (vis subset). Drops dominated ones.
    Returns new cnt, or -1 on buffer overflow."""
    j = head[b]
    while j != -1:
        if alive[j]:
            # existing j dominates new?  vis[j] subset of vis  AND  rc/load/tres <=
            if (VIS[j] & vis) == VIS[j] and RC[j] <= rc + 1e-9 \
                    and LOAD[j] <= load + 1e-9 and TRES[j] <= tres + 1e-9:
                return cnt
        j = nxt[j]
    j = head[b]
    while j != -1:
        if alive[j] and (vis & VIS[j]) == vis and rc <= RC[j] + 1e-9 \
                and load <= LOAD[j] + 1e-9 and tres <= TRES[j] + 1e-9:
            alive[j] = False
        j = nxt[j]
    if cnt >= RC.shape[0]:
        return -1
    RC[cnt] = rc; LOAD[cnt] = load; TRES[cnt] = tres; VIS[cnt] = vis
    NODE[cnt] = nd; PRED[cnt] = pred
    alive[cnt] = True; nxt[cnt] = head[b]; head[b] = cnt
    return cnt + 1


@njit(cache=True, nogil=True)
def price_cost_bidir(n, d, c_di, l_di, c_id, l_id, c_cc, l_cc, Q, L, F, wC,
                     pi, sig_i, rho_i, mu, fb_cc, fb_out, fb_in, keep_threshold,
                     Frc, Fload, Ftau, Fvis, Fnode, Fpred, Fnext, Falive, Fhead,
                     Brc, Bload, Btime, Bvis, Bnode, Bpred, Bnext, Balive, Bhead,
                     out_f, out_b, out_rc):
    """Bidirectional, fully-elementary pricing for the COST objective
    (reduced cost = sum c - duals; NO latency term). Monotone split on load at
    Q/2: forward paths depot->..->i and backward paths j->..->depot are each
    extended only while their load <= Q/2, then joined over an arc (i,j) (or
    closed directly). This caps path depth and collapses the label explosion
    that forward-only labelling suffers at alpha=0. Exact: every feasible route
    has a valid split (duplicates are harmless -- the master dedupes columns).

    Resources: load (<=Q), arrival/return time (<=L), elementarity (vis bitmask).
    Returns (min_rcf, n_out, overflow)."""
    half = 0.5 * Q
    FEAS = wC * F - rho_i - mu                  # per-route fixed reduced-cost part

    # ---- forward labelling: depot -> ... -> i (buckets by last node) ------ #
    for a in range(n):
        Fhead[a] = -1
    fc = 0
    for a in range(n):                                   # seed depot->a
        if d[a] > Q or fb_out[a] or l_di[a] + l_id[a] > L + 1e-9:
            continue
        rc = wC * c_di[a] - pi[a] - sig_i * d[a]
        fc = _el_insert(Fhead, Fnext, Falive, Frc, Fload, Ftau, Fvis, fc,
                        a, rc, d[a], l_di[a], np.int64(1) << a, Fnode, Fpred, a, -1)
        if fc == -1:
            return 0.0, 0, 1
    h = 0
    while h < fc:
        if Falive[h] and Fload[h] <= half:               # extend only below half-load
            a = Fnode[h]; rca = Frc[h]; ta = Ftau[h]; lo = Fload[h]; vs = Fvis[h]
            for b in range(n):
                if (vs >> b) & 1 or fb_cc[a, b]:
                    continue
                nlo = lo + d[b]
                if nlo > Q:
                    continue
                nta = ta + l_cc[a, b]
                if nta + l_id[b] > L + 1e-9:
                    continue
                rcb = rca + wC * c_cc[a, b] - pi[b] - sig_i * d[b]
                fc = _el_insert(Fhead, Fnext, Falive, Frc, Fload, Ftau, Fvis, fc,
                                b, rcb, nlo, nta, vs | (np.int64(1) << b),
                                Fnode, Fpred, b, h)
                if fc == -1:
                    return 0.0, 0, 1
        h += 1

    # ---- backward labelling: j -> ... -> depot (buckets by first node) ---- #
    for a in range(n):
        Bhead[a] = -1
    bc = 0
    for a in range(n):                                   # seed a->depot
        if d[a] > Q or fb_in[a] or l_di[a] + l_id[a] > L + 1e-9:
            continue
        rc = wC * c_id[a] - pi[a] - sig_i * d[a]
        bc = _el_insert(Bhead, Bnext, Balive, Brc, Bload, Btime, Bvis, bc,
                        a, rc, d[a], l_id[a], np.int64(1) << a, Bnode, Bpred, a, -1)
        if bc == -1:
            return 0.0, 0, 1
    h = 0
    while h < bc:
        if Balive[h] and Bload[h] <= half:               # prepend u -> (first node a)
            a = Bnode[h]; rca = Brc[h]; tb = Btime[h]; lo = Bload[h]; vs = Bvis[h]
            for u in range(n):
                if (vs >> u) & 1 or fb_cc[u, a]:
                    continue
                nlo = lo + d[u]
                if nlo > Q:
                    continue
                ntb = tb + l_cc[u, a]
                if l_di[u] + ntb > L + 1e-9:
                    continue
                rcu = rca + wC * c_cc[u, a] - pi[u] - sig_i * d[u]
                bc = _el_insert(Bhead, Bnext, Balive, Brc, Bload, Btime, Bvis, bc,
                                u, rcu, nlo, ntb, vs | (np.int64(1) << u),
                                Bnode, Bpred, u, h)
                if bc == -1:
                    return 0.0, 0, 1
        h += 1

    # ---- joins ------------------------------------------------------------ #
    min_rcf = 0.0
    n_out = 0
    cap = out_rc.shape[0]

    # (b) forward closed directly: depot->..->i->depot
    for f in range(fc):
        if not Falive[f]:
            continue
        i = Fnode[f]
        if fb_in[i]:
            continue
        rcf = Frc[f] + wC * c_id[i] + FEAS
        if rcf < min_rcf:
            min_rcf = rcf
        if rcf < keep_threshold and n_out < cap:
            out_f[n_out] = f; out_b[n_out] = -1; out_rc[n_out] = rcf; n_out += 1

    # (c) backward opened directly: depot->j->..->depot
    for bb in range(bc):
        if not Balive[bb]:
            continue
        j = Bnode[bb]
        if fb_out[j]:
            continue
        rcf = wC * c_di[j] + Brc[bb] + FEAS
        if rcf < min_rcf:
            min_rcf = rcf
        if rcf < keep_threshold and n_out < cap:
            out_f[n_out] = -1; out_b[n_out] = bb; out_rc[n_out] = rcf; n_out += 1

    # (a) full joins forward f @ i  + arc(i,j) +  backward b @ j
    for f in range(fc):
        if not Falive[f]:
            continue
        i = Fnode[f]; vf = Fvis[f]; lof = Fload[f]; taf = Ftau[f]; rcf0 = Frc[f]
        for j in range(n):
            if i == j or (vf >> j) & 1 or fb_cc[i, j]:
                continue
            bb = Bhead[j]
            while bb != -1:
                if Balive[bb]:
                    vb = Bvis[bb]
                    if (vf & vb) == 0 and lof + Bload[bb] <= Q \
                            and taf + l_cc[i, j] + Btime[bb] <= L + 1e-9:
                        rcf = rcf0 + wC * c_cc[i, j] + Brc[bb] + FEAS
                        if rcf < min_rcf:
                            min_rcf = rcf
                        if rcf < keep_threshold and n_out < cap:
                            out_f[n_out] = f; out_b[n_out] = bb
                            out_rc[n_out] = rcf; n_out += 1
                bb = Bnext[bb]
    return min_rcf, n_out, 0


class NbPricer:
    """Reusable buffers + per-instance constant arrays for the compiled pricer."""

    def __init__(self, inst, ng_masks, cap=2_000_000, out_cap=4000):
        n, Jn = inst.n, inst.J
        self.n = n
        self.Q = float(inst.Q)
        self.L = float(inst.L)
        self.alpha = float(inst.alpha)
        self.d = np.array([inst.demand[Jn[a]] for a in range(n)], dtype=np.float64)
        self.p = np.array([inst.priority[Jn[a]] for a in range(n)], dtype=np.float64)
        self.l_cc = np.array([[inst.l[Jn[a]][Jn[b]] for b in range(n)]
                              for a in range(n)], dtype=np.float64)
        self.c_cc = np.array([[inst.c[Jn[a]][Jn[b]] for b in range(n)]
                              for a in range(n)], dtype=np.float64)
        self.ng = np.array(ng_masks, dtype=np.int64)
        cap = max(cap, 50000)
        self.rc = np.empty(cap, np.float64)
        self.tau = np.empty(cap, np.float64)
        self.load = np.empty(cap, np.float64)
        self.mask = np.empty(cap, np.int64)
        self.sstate = np.empty(cap, np.int64)
        self.node = np.empty(cap, np.int32)
        self.pred = np.empty(cap, np.int32)
        self.alive = np.empty(cap, np.bool_)
        self.next_lab = np.empty(cap, np.int32)
        self.bucket_head = np.empty(n, np.int32)
        self.out_lab = np.empty(out_cap, np.int32)
        self.out_rcf = np.empty(out_cap, np.float64)
        self._zero_cc = np.zeros((n, n), np.int8)   # default: nothing forbidden
        self._zero_v = np.zeros(n, np.int8)
        # --- bidirectional cost-pricer buffers (allocated lazily) ---------- #
        self._bidir_cap = 0
        self._out_cap2 = max(out_cap, 20000)

    def _ensure_bidir(self, cap):
        """(Re)allocate the forward/backward label buffers for the bidirectional
        cost pricer."""
        if self._bidir_cap >= cap:
            return
        self._bidir_cap = cap
        n = self.n
        for pre in ("F", "B"):
            setattr(self, pre + "rc", np.empty(cap, np.float64))
            setattr(self, pre + "load", np.empty(cap, np.float64))
            setattr(self, pre + ("tau" if pre == "F" else "time"), np.empty(cap, np.float64))
            setattr(self, pre + "vis", np.empty(cap, np.int64))
            setattr(self, pre + "node", np.empty(cap, np.int32))
            setattr(self, pre + "pred", np.empty(cap, np.int32))
            setattr(self, pre + "next", np.empty(cap, np.int32))
            setattr(self, pre + "alive", np.empty(cap, np.bool_))
            setattr(self, pre + "head", np.empty(n, np.int32))
        self._out_f = np.empty(self._out_cap2, np.int32)
        self._out_b = np.empty(self._out_cap2, np.int32)
        self._out_rc2 = np.empty(self._out_cap2, np.float64)

    def _cut_csr(self, srcs):
        """Build omega array + customer->cut CSR mapping from active SRCs."""
        n = self.n
        ncut = len(srcs)
        omega = np.array([s[1] for s in srcs], dtype=np.float64) if ncut \
            else np.zeros(0, dtype=np.float64)
        per_cust = [[] for _ in range(n)]
        for m, (Cmask, _) in enumerate(srcs):
            for b in range(n):
                if (Cmask >> b) & 1:
                    per_cust[b].append(m)
        start = np.zeros(n + 1, dtype=np.int32)
        for b in range(n):
            start[b + 1] = start[b] + len(per_cust[b])
        idx = np.empty(int(start[n]), dtype=np.int32)
        for b in range(n):
            for t, m in enumerate(per_cust[b]):
                idx[start[b] + t] = m
        return omega, start, idx

    def price(self, inst, i, pi_cov, sig_i, rho_i, mu, srcs=(), max_cols=40,
              keep_threshold=-1e-6, fb_cc=None, fb_out=None, fb_in=None,
              wC=1.0, wP=None):
        # wP=None -> weighted-sum pricing with wP = inst.alpha (unchanged behaviour).
        # In Tchebycheff/budget modes wC, wP are passed fresh on every CG iteration.
        if wP is None:
            wP = float(inst.alpha)
        n, Jn = self.n, inst.J
        l_di = np.array([inst.l[i][Jn[a]] for a in range(n)], dtype=np.float64)
        c_di = np.array([inst.c[i][Jn[a]] for a in range(n)], dtype=np.float64)
        l_id = np.array([inst.l[Jn[a]][i] for a in range(n)], dtype=np.float64)
        c_id = np.array([inst.c[Jn[a]][i] for a in range(n)], dtype=np.float64)
        pic = np.array(pi_cov, dtype=np.float64)
        omega, ccut_start, ccut_idx = self._cut_csr(srcs)
        fb_cc = self._zero_cc if fb_cc is None else fb_cc
        fb_out = self._zero_v if fb_out is None else fb_out
        fb_in = self._zero_v if fb_in is None else fb_in
        min_rcf, n_out, cnt, overflow = price_core(
            n, self.d, self.p, l_di, c_di, l_id, c_id, self.l_cc, self.c_cc,
            self.Q, self.L, float(inst.F[i]), float(wC), float(wP), pic,
            float(sig_i), float(rho_i), float(mu), self.ng, float(keep_threshold),
            omega, ccut_start, ccut_idx, fb_cc, fb_out, fb_in,
            self.rc, self.tau, self.load, self.mask, self.sstate, self.node,
            self.pred, self.alive, self.next_lab, self.bucket_head,
            self.out_lab, self.out_rcf)
        if overflow:
            return None, None
        order = np.argsort(self.out_rcf[:n_out])
        routes = []
        for k in order[:max_cols]:
            idx = int(self.out_lab[k])
            path = []
            cur = idx
            while cur != -1:
                path.append(int(self.node[cur]))
                cur = int(self.pred[cur])
            path.reverse()
            routes.append((float(self.out_rcf[k]), tuple(path)))
        return routes, float(min_rcf)

    def price_cost(self, inst, i, pi_cov, sig_i, rho_i, mu, max_cols=40,
                   keep_threshold=-1e-6, fb_cc=None, fb_out=None, fb_in=None,
                   cap=1_000_000, wC=1.0):
        """Bidirectional, fully-elementary COST pricing for depot i (no latency,
        no SRC). Returns (routes, min_rcf) like ``price`` -- routes are
        (reduced_cost, (customer_idx, ...))."""
        n, Jn = self.n, inst.J
        self._ensure_bidir(cap)
        l_di = np.array([inst.l[i][Jn[a]] for a in range(n)], dtype=np.float64)
        c_di = np.array([inst.c[i][Jn[a]] for a in range(n)], dtype=np.float64)
        l_id = np.array([inst.l[Jn[a]][i] for a in range(n)], dtype=np.float64)
        c_id = np.array([inst.c[Jn[a]][i] for a in range(n)], dtype=np.float64)
        pic = np.array(pi_cov, dtype=np.float64)
        fb_cc = self._zero_cc if fb_cc is None else fb_cc
        fb_out = self._zero_v if fb_out is None else fb_out
        fb_in = self._zero_v if fb_in is None else fb_in
        min_rcf, n_out, overflow = price_cost_bidir(
            n, self.d, c_di, l_di, c_id, l_id, self.c_cc, self.l_cc,
            self.Q, self.L, float(inst.F[i]), float(wC), pic, float(sig_i), float(rho_i),
            float(mu), fb_cc, fb_out, fb_in, float(keep_threshold),
            self.Frc, self.Fload, self.Ftau, self.Fvis, self.Fnode, self.Fpred,
            self.Fnext, self.Falive, self.Fhead,
            self.Brc, self.Bload, self.Btime, self.Bvis, self.Bnode, self.Bpred,
            self.Bnext, self.Balive, self.Bhead,
            self._out_f, self._out_b, self._out_rc2)
        if overflow:
            return None, None

        def fwd_path(f):                       # depot->..->i  (customer order)
            out = []
            cur = f
            while cur != -1:
                out.append(int(self.Fnode[cur])); cur = int(self.Fpred[cur])
            out.reverse()
            return out

        def bwd_path(b):                       # j->..->depot  (already forward order)
            out = []
            cur = b
            while cur != -1:
                out.append(int(self.Bnode[cur])); cur = int(self.Bpred[cur])
            return out

        order = np.argsort(self._out_rc2[:n_out])
        routes = []
        seen = set()
        for k in order[:max_cols]:
            f = int(self._out_f[k]); b = int(self._out_b[k])
            path = (fwd_path(f) if f != -1 else []) + (bwd_path(b) if b != -1 else [])
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            routes.append((float(self._out_rc2[k]), key))
        return routes, float(min_rcf)
