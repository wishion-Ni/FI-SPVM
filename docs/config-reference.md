# Config Reference

## Top Level

Preferred layout uses nested sections:

- `data`
- `kernel`
- `admm`
- `normalization`
- `spectrum_completion`
- `logging`
- `visualization`
- `param_selection`
- `preprocess`
- `priors`

Legacy top-level keys `inputFile`, `noiseWeighted`, and `spectrum_input_type` are still accepted and mapped to `data.*`.

## `data`

- `input_file`: path to the input spectrum file.
- `noise_weighted`: whether to use input weights.
- `input_type`: spectrum input mode, for example `freq` or `period`.

## `kernel`

- `tau_min`, `tau_max`: relaxation time range. Must satisfy `tau_min < tau_max`.
- `num_tau`: number of tau grid points. Must be at least `2`.
- `gamma_min`, `gamma_max`: beta/gamma range. Must satisfy `gamma_min < gamma_max`.
- `num_gamma`: number of gamma grid points. Must be at least `2`.
- `gamma_scale`: `linear` or `log`.

## `admm`

- `lambda1`
- `lambda_tv_tau`
- `lambda_tv_beta`
- `rho`
- `max_iters`
- `tol_primal`
- `tol_dual`
- `group_size_tau`
- `group_size_beta`

Validation requires non-negative regularization values, positive `rho`, and group sizes of at least `1`.

## `spectrum_completion`

- `method`: `none`, `pchip`, or `akima`
- `interpolate`
- `num_points`
- `log_space`
- `weight`

## `visualization`

- `enabled`
- `outputDir`
- `transient_tmax`
- `transient_samples`

`outputDir` must not be empty. `transient_tmax` must be positive. `transient_samples` must be at least `2`.

## `param_selection`

- `enable`
- `num_lambda1`, `lambda1_min`, `lambda1_max`
- `num_lambdat`, `lambdat_min`, `lambdat_max`
- `num_lambdab`, `lambdab_min`, `lambdab_max`
- `outputDir`
- `scan_max_iters`
- `scan_tol`
- `refine_after`

When enabled, each min/max pair must be ordered and each grid count must be at least `1`.

## `preprocess.find_peaks`

- `enable`
- `interp_type`: `akima` or `pchip`
- `smooth_window`
- `peak_prominence`
- `peak_dist_dec`
- `interp_factor`
- `weight_factor`

## `priors`

- `beta_center`
- `beta_sigma`
- `beta_strength`

`beta_sigma` must be positive and `beta_strength` must be non-negative.
