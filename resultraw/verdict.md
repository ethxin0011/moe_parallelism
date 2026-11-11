# Divergence Summary

Seed band (final KL): (0.3499165400862694, 0.3526703715324402, 4)  |  1-ULP band: (0.2830193042755127, 0.2865292727947235, 3)

Numerical-only layouts within seed band at final step: 8/8 (mean KL / seed KL = 0.801).
Per-rank rule runs exceeding seed band vs same-layout baseline: none.
Early divergence slope (log10 param_rel per step): MoE 0.0182 vs dense 0.0231 (ratio 0.79).
Resume branches (final KL): F_b128_dp8_ep4=0.12, F_b128_dp4_ep1=0.128, F_b1024_dp8_ep4=0.00678, F_b1024_dp4_ep1=0.00675, F_b128_ulp1=0.116, F_b1024_ulp1=0.00651
Size base: DP4/EP2 layout KL / seed KL = 0.783
Size m: DP4/EP2 layout KL / seed KL = 0.859
Size base: DP2/EP2 layout KL / seed KL = 0.808
Size m: DP2/EP2 layout KL / seed KL = 0.877
Size l: DP2/EP2 layout KL / seed KL = 0.859
