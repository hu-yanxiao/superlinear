# Optional inner energy readout

Set `model.readout.inner: true` or pass `--inner-readout` to add a zero-initialized constant and inner scalar energy head. The default is off. Coefficients are saved as `[b0, bq...]`; older `--two-layer-gate-readout` models retain their Q-only layout.

For the new head, linear initialization jointly solves the outer and inner energy coefficients, including energy, force and virial rows, while keeping center species amplitudes and all other parameters fixed. `--inner-readout-linear-init` requests this solve on a trained checkpoint. The old Q-only option retains its readout-only behavior. The constant contributes no force or virial; dependent constant columns need not have unique fitted coefficients.
