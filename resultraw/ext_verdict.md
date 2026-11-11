# Router Margin and Expert Granularity Summary

## 1. Router margin vs. routing flips (0 < flip rate < 0.5)
Median AUC that flipped decisions have smaller margins: 0.968162391618853
Flip rate in lowest-10% margins vs rest: (0.15143358658769093, 0.031049310422678538)

                   auc_flip_has_smaller_margin  flip_rate
other                                                    
A_dp2_ep1                               0.9700     0.0008
A_dp4_ep1                               0.9850     0.0008
A_dp8_ep4                               0.9848     0.0006
B_ulp1                                  0.9682     0.0006
C_dp8_ep1_cap                           0.9456     0.0336
C_dp8_ep1_lbl                           0.9498     0.0252
D_dp8_ep4_bglobal                       0.9925     0.0022
D_dp8_ep4_blocal                        0.9002     0.0884
H_base_dp2_ep2                          0.9650     0.0023
I_e32_dp2_ep2                           0.9789     0.0025
I_e64_dp2_ep2                           0.9852     0.0032
I_e8_dp2_ep2                            0.8966     0.0044
X_dp1_thr8                              0.9724     0.0010

## 2. Near-tie fraction (mean of first 3 snapshots)
               near_tie_frac_0.01  near_tie_frac_0.03  near_tie_frac_0.1
run                                                                     
A_dp1_ep1                 0.06574             0.18530            0.48813
D_dp1_ep1                 0.04560             0.12981            0.36800
G_dp1_ep1                 0.06567             0.18516            0.48789
H_l_dp1_ep1               0.07768             0.21288            0.53202
H_m_dp1_ep1               0.06831             0.19125            0.50786
I_e32_dp1_ep1             0.11889             0.31501            0.71209
I_e64_dp1_ep1             0.23265             0.54226            0.91910
I_e8_dp1_ep1              0.02728             0.08095            0.24593

## 3. Expert granularity (fixed total and active parameters, fp16 T4)
 gran  experts  final_layout_kl  final_seed_kl  layout_over_seed  layout_kl_at_32  layout_kl_at_128  step_kl_ge_1e3  step_kl_ge_1e2
dense        0         0.206236       0.354232          0.582207     5.822356e-08          0.030081             128             128
   e8        8         0.274462       0.342938          0.800326     2.897409e-04          0.070104              64              64
  e16       16         0.283565       0.348305          0.814127     4.541142e-05          0.080858              64              64
  e32       32         0.291662       0.341871          0.853135     9.878337e-05          0.082766              64              64
  e64       64         0.272042       0.336567          0.808283     8.816588e-08          0.069927              64              64
