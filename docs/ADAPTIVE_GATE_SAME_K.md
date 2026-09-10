# Adaptive radial gate same-k topology

This branch keeps `--superlinear-mode=radial-coefficient-gate` default behavior
unchanged: the outer graph uses `--sh-factor-pruning=same-k`, and the ranked
neighbor gate uses `--two-layer-gate-factor-pruning=q-total` unless the user
explicitly requests another supported gate topology.

The optional same-k gate is selected with:

```bash
--superlinear-mode=radial-coefficient-gate \
--two-layer-gate-factor-pruning=same-k
```

The default outer graph remains same-channel.  The gate rank still has the full
learnable `G x K` contraction weights per `l`; same-k only changes the ranked
gate scalar product graph so that no gate scalar contains factors from different
gate channels.

The outer graph can be made cross-channel independently with:

```bash
--superlinear-mode=radial-coefficient-gate \
--sh-factor-pruning=q-total
```

In that mode the model serializes `outer_channel_mode = cross-channel` and the
adaptive radial coefficients still provide the `O` outer primitive channels.
The final outer invariant product graph may then multiply different outer
channels.  This does not change the gate graph; use
`--two-layer-gate-factor-pruning=same-k` separately when same-k gate topology is
desired.

Model files written by this branch include:

```text
two_layer_gate_factor_pruning = q-total|same-k
```

Old files without this key remain readable.  Existing ranked gate files are
interpreted as `q-total`, matching the historical native restriction.  Unranked
gate files with scalar metadata are inferred from their saved scalar indices.
During resume, an explicit `--two-layer-gate-factor-pruning=` option must match
the loaded model; changing the gate graph requires a fresh `init-sh` model.

Server-side contract test:

```bash
python3 dev_test/adaptive_gate_same_k_contract_check.py --binary bin/mlp-sus2
```

The test checks adaptive same-k and q-total gate initialization, the independent
outer q-total/cross-channel mode, serialized scalar counts, adaptive radial
weight dimensions, model roundtrip preservation, and resume mismatch rejection.
