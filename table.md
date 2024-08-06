FNO and LSM are all tested with 4-layer !

| Burgers $\\$ MSE(1e-3) | no channel mixing | p=0 | p=1 | p=2 | p=4 | p=8 | p=16 |
|-|-|-|-|-|-|-|-|
| FNO | 2.225 | 2.342 | 1.130 | 1.147 | 1.163 | 1.059 | 1.071 |
<!-- | FNO (4 layers) | 1.875 | 4.792 | | 1.687 | | | | -->

| DarcyFlow $\\$ MSE(1e-3) | no channel mixing | p=0 | p=1 | p=2 | p=4 | p=8 | p=16 |
|-|-|-|-|-|-|-|-|
| FNO         | 5.073 | 5.889 | 5.689 | 5.726 | 6.024 | 5.641 | |
| T-FNO       | 10.83 | 7.500 | 7.192 | 6.919 |  | | |
| LSM         | / | 4.048 | 3.989 | 3.777 | 4.032 | 3.695 | 3.902 |

TorusLi: T=10
| TorusLi $\\$ MSE | no channel mixing | p=0 | p=1 | p=2 | p=4 | p=8 | p=16 |
|-|-|-|-|-|-|-|-|
| FNO         | 0.1496 | 0.1404 |  | 0.1376 | | | |
| T-FNO       | 0.1448 | 0.1425 |  | 0.1404 | | | |
| LSM         | / |  |  |  |  |  |  |

<!-- | TorusVisForce MSE | no channel mixing | p=0 | p=1 | p=2 | p=4 | p=8 | p=16 |
|-|-|-|-|-|-|-|-|
| LSM         |  |  |  |  |  |  |  | -->
TorusVisForceFew: T=4
| TorusVisForceFew $\\$ MSE(1e-2) | no channel mixing | p=0 | p=1 | p=2 | p=4 | p=8 | p=16 |
|-|-|-|-|-|-|-|-|
| T-FNO       | 2.037 | 2.104 | 1.759 | 1.757 | 1.733 | 1.752 | 1.690 |
| LSM         | / | 8.925 | 8.553 | 8.605 | 8.675 | 8.557 |  |