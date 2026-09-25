# Split Policy Report

## Held-out Test Trials

| Trial | Diameter (mm) | Depth (mm) | Regime | Loading Samples |
| --- | ---: | ---: | --- | ---: |
| ecomesh_d10_z1.0_test3 | 10.0 | 1.0 | shallow | 239399 |
| ecomesh_d5_z1.0_test3 | 5.0 | 1.0 | shallow | 239590 |
| ecomesh_d5_z1.5_test9 | 5.0 | 1.5 | deep | 239446 |

## Minimum Regime Sample Policy

- shallow: >= 200,000 samples
- deep: >= 200,000 samples
- current held-out shallow samples: 478,989
- current held-out deep samples: 239,446

## Diameter/Depth Stratified CV Design

- fold_0: ecomesh_d10_z1.0_test1, ecomesh_d5_z1.0_test1, ecomesh_d5_z1.5_test4
- fold_1: ecomesh_d10_z1.0_test2, ecomesh_d5_z1.0_test2, ecomesh_d5_z1.5_test5
- fold_2: ecomesh_d5_z1.5_test1, ecomesh_d5_z1.5_test6
- fold_3: ecomesh_d5_z1.5_test2, ecomesh_d5_z1.5_test7
- fold_4: ecomesh_d5_z1.5_test3, ecomesh_d5_z1.5_test8
