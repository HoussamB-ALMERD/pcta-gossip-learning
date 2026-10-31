| dataset | α | f | attack | method | n | acc | AUROC | byz mass | FPR | TPR |
|---|---|---|---|---|---|---|---|---|---|---|
| cifar10 | 0.1 | 0.2 | calibrated_adaptive | local_krum-ratio0.5 | 3 | 0.376 ± 0.047 | 0.518 ± 0.030 | 0.195 ± 0.028 | – | – |
| cifar10 | 0.1 | 0.2 | calibrated_adaptive | local_krum-ratio1.0 | 3 | 0.384 ± 0.040 | 0.584 ± 0.055 | 0.169 ± 0.037 | – | – |
| cifar10 | 0.1 | 0.2 | calibrated_adaptive | local_krum-ratio2.0 | 3 | 0.374 ± 0.041 | 0.761 ± 0.076 | 0.097 ± 0.041 | – | – |
| cifar10 | 0.1 | 0.2 | calibrated_adaptive | local_krum-ratio5.0 | 3 | 0.375 ± 0.035 | 0.854 ± 0.008 | 0.060 ± 0.011 | – | – |
| cifar10 | 0.1 | 0.2 | calibrated_adaptive | pcta_soft-ratio0.5 | 3 | 0.418 ± 0.031 | 0.496 ± 0.046 | 0.203 ± 0.012 | 0.028 ± 0.000 | 0.030 ± 0.018 |
| cifar10 | 0.1 | 0.2 | calibrated_adaptive | pcta_soft-ratio1.0 | 3 | 0.415 ± 0.045 | 0.501 ± 0.033 | 0.202 ± 0.013 | 0.028 ± 0.002 | 0.032 ± 0.018 |
| cifar10 | 0.1 | 0.2 | calibrated_adaptive | pcta_soft-ratio2.0 | 3 | 0.421 ± 0.034 | 0.533 ± 0.055 | 0.198 ± 0.011 | 0.028 ± 0.001 | 0.032 ± 0.018 |
| cifar10 | 0.1 | 0.2 | calibrated_adaptive | pcta_soft-ratio5.0 | 3 | 0.410 ± 0.044 | 0.728 ± 0.064 | 0.168 ± 0.024 | 0.025 ± 0.001 | 0.086 ± 0.089 |
| cifar10 | 0.1 | 0.2 | sign_flip | local_krum-ww10 | 3 | 0.290 ± 0.033 | 0.855 ± 0.010 | 0.059 ± 0.012 | – | – |
| cifar10 | 0.1 | 0.2 | sign_flip | local_krum-ww3 | 3 | 0.285 ± 0.065 | 0.855 ± 0.009 | 0.059 ± 0.012 | – | – |
| cifar10 | 0.1 | 0.2 | sign_flip | local_krum-ww5 | 3 | 0.293 ± 0.029 | 0.855 ± 0.009 | 0.059 ± 0.012 | – | – |
| cifar10 | 0.1 | 0.2 | sign_flip | pcta_soft-no_probation-ww10 | 3 | 0.376 ± 0.053 | 0.977 ± 0.029 | 0.048 ± 0.039 | 0.015 ± 0.007 | 0.674 ± 0.145 |
| cifar10 | 0.1 | 0.2 | sign_flip | pcta_soft-no_probation-ww3 | 3 | 0.367 ± 0.062 | 0.954 ± 0.054 | 0.085 ± 0.053 | 0.012 ± 0.001 | 0.402 ± 0.227 |
| cifar10 | 0.1 | 0.2 | sign_flip | pcta_soft-no_probation-ww5 | 3 | 0.367 ± 0.057 | 0.970 ± 0.035 | 0.061 ± 0.043 | 0.014 ± 0.004 | 0.570 ± 0.160 |
| cifar10 | 0.1 | 0.2 | sign_flip | pcta_soft-ww10 | 3 | 0.381 ± 0.030 | 0.978 ± 0.028 | 0.035 ± 0.031 | 0.016 ± 0.005 | 0.660 ± 0.110 |
| cifar10 | 0.1 | 0.2 | sign_flip | pcta_soft-ww3 | 3 | 0.376 ± 0.052 | 0.974 ± 0.026 | 0.050 ± 0.033 | 0.013 ± 0.002 | 0.376 ± 0.174 |
| cifar10 | 0.1 | 0.2 | sign_flip | pcta_soft-ww5 | 3 | 0.391 ± 0.064 | 0.977 ± 0.029 | 0.040 ± 0.033 | 0.015 ± 0.006 | 0.562 ± 0.116 |
| fashion_mnist | 0.1 | 0.2 | calibrated_adaptive | local_krum-ratio0.5 | 3 | 0.782 ± 0.022 | 0.513 ± 0.039 | 0.197 ± 0.029 | – | – |
| fashion_mnist | 0.1 | 0.2 | calibrated_adaptive | local_krum-ratio1.0 | 3 | 0.786 ± 0.026 | 0.547 ± 0.052 | 0.184 ± 0.035 | – | – |
| fashion_mnist | 0.1 | 0.2 | calibrated_adaptive | local_krum-ratio2.0 | 3 | 0.790 ± 0.013 | 0.654 ± 0.037 | 0.140 ± 0.026 | – | – |
| fashion_mnist | 0.1 | 0.2 | calibrated_adaptive | local_krum-ratio5.0 | 3 | 0.791 ± 0.025 | 0.824 ± 0.016 | 0.071 ± 0.015 | – | – |
| fashion_mnist | 0.1 | 0.2 | calibrated_adaptive | pcta_soft-ratio0.5 | 3 | 0.797 ± 0.033 | 0.516 ± 0.085 | 0.202 ± 0.019 | 0.009 ± 0.021 | 0.007 ± 0.009 |
| fashion_mnist | 0.1 | 0.2 | calibrated_adaptive | pcta_soft-ratio1.0 | 3 | 0.797 ± 0.028 | 0.518 ± 0.091 | 0.202 ± 0.020 | 0.009 ± 0.020 | 0.008 ± 0.009 |
| fashion_mnist | 0.1 | 0.2 | calibrated_adaptive | pcta_soft-ratio2.0 | 3 | 0.796 ± 0.036 | 0.529 ± 0.098 | 0.201 ± 0.020 | 0.009 ± 0.020 | 0.008 ± 0.009 |
| fashion_mnist | 0.1 | 0.2 | calibrated_adaptive | pcta_soft-ratio5.0 | 3 | 0.794 ± 0.036 | 0.592 ± 0.078 | 0.192 ± 0.021 | 0.008 ± 0.018 | 0.019 ± 0.009 |
| fashion_mnist | 0.1 | 0.2 | sign_flip | local_krum-ww10 | 3 | 0.743 ± 0.031 | 0.851 ± 0.013 | 0.061 ± 0.013 | – | – |
| fashion_mnist | 0.1 | 0.2 | sign_flip | local_krum-ww3 | 3 | 0.730 ± 0.032 | 0.851 ± 0.015 | 0.061 ± 0.014 | – | – |
| fashion_mnist | 0.1 | 0.2 | sign_flip | local_krum-ww5 | 3 | 0.741 ± 0.030 | 0.851 ± 0.013 | 0.061 ± 0.013 | – | – |
| fashion_mnist | 0.1 | 0.2 | sign_flip | pcta_soft-no_probation-ww10 | 3 | 0.753 ± 0.045 | 0.939 ± 0.021 | 0.108 ± 0.013 | 0.004 ± 0.011 | 0.266 ± 0.030 |
| fashion_mnist | 0.1 | 0.2 | sign_flip | pcta_soft-no_probation-ww3 | 3 | 0.732 ± 0.027 | 0.912 ± 0.043 | 0.135 ± 0.008 | 0.002 ± 0.007 | 0.084 ± 0.071 |
| fashion_mnist | 0.1 | 0.2 | sign_flip | pcta_soft-no_probation-ww5 | 3 | 0.744 ± 0.026 | 0.923 ± 0.031 | 0.120 ± 0.016 | 0.003 ± 0.010 | 0.194 ± 0.018 |
| fashion_mnist | 0.1 | 0.2 | sign_flip | pcta_soft-ww10 | 3 | 0.762 ± 0.038 | 0.948 ± 0.017 | 0.090 ± 0.010 | 0.004 ± 0.010 | 0.269 ± 0.047 |
| fashion_mnist | 0.1 | 0.2 | sign_flip | pcta_soft-ww3 | 3 | 0.753 ± 0.055 | 0.949 ± 0.019 | 0.095 ± 0.008 | 0.004 ± 0.010 | 0.071 ± 0.048 |
| fashion_mnist | 0.1 | 0.2 | sign_flip | pcta_soft-ww5 | 3 | 0.757 ± 0.027 | 0.946 ± 0.023 | 0.093 ± 0.013 | 0.004 ± 0.012 | 0.170 ± 0.006 |
