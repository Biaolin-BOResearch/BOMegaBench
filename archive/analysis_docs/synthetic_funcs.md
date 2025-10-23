# Complete Single-Objective Synthetic Benchmark Functions Catalog

## Total Count: **126 Distinct Single-Objective Functions**

### BBOB Core Suite (24 functions)

| ID | Function Name | Formula/Description | Domain | Global Min | Properties |
|----|---------------|-------------------|--------|------------|------------|
| f1 | **Sphere** | f(x) = ‖z‖² + f_opt | [-5,5]^d | f(0)=0 | Unimodal, separable |
| f2 | **Ellipsoid Separable** | Σᵢ 10^(6(i-1)/(D-1)) zᵢ² | [-5,5]^d | f(0)=0 | Ill-conditioned ~10⁶ |
| f3 | **Rastrigin Separable** | 10(D - Σcos(2πzᵢ)) + ‖z‖² | [-5,5]^d | f(0)=0 | ~10^D local optima |
| f4 | **Skew Rastrigin-Bueche** | Asymmetric Rastrigin | [-5,5]^d | Variable | Asymmetric, ~10^D optima |
| f5 | **Linear Slope** | Purely linear function | [-5,5]^d | Variable | Tests gradient-free methods |
| f6 | **Attractive Sector** | Highly asymmetric, hypercone | [-5,5]^d | Variable | Hypercone structure |
| f7 | **Step-Ellipsoid** | Plateaus + ill-conditioned ellipsoid | [-5,5]^d | Variable | Plateaus |
| f8 | **Rosenbrock Original** | Tri-band structure | [-5,5]^d | Variable | Banana valley |
| f9 | **Rosenbrock Rotated** | Rotated version | [-5,5]^d | Variable | Non-separable valley |
| f10 | **Ellipsoid** | Rotated ellipsoid | [-5,5]^d | Variable | Conditioning ~10⁶ |
| f11 | **Discus** | One short principal axis | [-5,5]^d | Variable | One short axis |
| f12 | **Bent Cigar** | One long principal axis | [-5,5]^d | Variable | One long axis |
| f13 | **Sharp Ridge** | Sharp ridge structure | [-5,5]^d | Variable | Ridge following |
| f14 | **Sum of Different Powers** | Variable sensitivities | [-5,5]^d | Variable | Power variations |
| f15 | **Rastrigin** | Prototypical multimodal | [-5,5]^d | Variable | ~10^D local optima |
| f16 | **Weierstrass** | Fractal-like function | [-0.5,0.5]^d | Variable | Nowhere differentiable |
| f17 | **Schaffer F7 (cond 10)** | Variable amplitude/frequency | [-5,5]^d | Variable | Modulated landscape |
| f18 | **Schaffer F7 (cond 1000)** | Higher conditioning variant | [-5,5]^d | Variable | High conditioning |
| f19 | **Griewank-Rosenbrock F8F2** | Composition function | [-5,5]^d | Variable | Two function combo |
| f20 | **Schwefel** | 2^D prominent minima | [-5,5]^d | Variable | Deceptive global min |
| f21 | **Gallagher 101 peaks** | 101 random peaks | [-5,5]^d | Variable | 101 peaks |
| f22 | **Gallagher 21 peaks** | 21 peaks, conditioning ~1000 | [-5,5]^d | Variable | 21 peaks |
| f23 | **Katsuura** | >10^D global optima | [-5,5]^d | Variable | Pathological repetitive |
| f24 | **Lunacek bi-Rastrigin** | Two superimposed funnels | [-5,5]^d | Variable | Deceptive design |


### Classical Mathematical Functions (40 functions)

| Function Name | Formula/Description | Domain | Global Min | Properties |
|---------------|-------------------|--------|------------|------------|
| **Ackley** | -20·exp(-0.2√(Σxᵢ²/n)) - exp(Σcos(2πxᵢ)/n) + 20 + e | [-32.768,32.768]^d | f(0)=0 | Nearly flat outer region |
| **Griewank** | 1/4000·Σxᵢ² - Πcos(xᵢ/√i) + 1 | [-600,600]^d | f(0)=0 | Variable correlation |
| **Schwefel 1.2** | Σ(Σxⱼ)² | [-100,100]^d | f(0)=0 | Double sum |
| **Schwefel 2.20** | Σ|xᵢ| | [-100,100]^d | f(0)=0 | Absolute values |
| **Schwefel 2.21** | max|xᵢ| | [-100,100]^d | f(0)=0 | MaxMod |
| **Schwefel 2.22** | Σ|xᵢ| + Π|xᵢ| | [-10,10]^d | f(0)=0 | Sum + product |
| **Schwefel 2.23** | Σxᵢ¹⁰ | [-10,10]^d | f(0)=0 | High powers |
| **Schwefel 2.26** | 418.9829n - Σxᵢsin(√|xᵢ|) | [-500,500]^d | Variable | Classic deceptive |
| **Styblinski-Tang** | ½Σ(xᵢ⁴ - 16xᵢ² + 5xᵢ) | [-5,5]^d | f=-39.166·d | Separable multimodal |
| **Dixon-Price** | (x₁-1)² + Σi(2xᵢ² - xᵢ₋₁)² | [-10,10]^d | Variable | Adjacent interaction |
| **Levy** | sin²(πw₁) + Σ(wᵢ-1)²[1+10sin²(πwᵢ+1)] + (wₙ-1)²[1+sin²(2πwₙ)] | [-10,10]^d | f(1,...,1)=0 | Complex multimodal |
| **Levy N.13** | sin²(3πx₁) + Σ(xᵢ-1)²[1+sin²(3πxᵢ+1)] + (xₙ-1)²[1+sin²(2πxₙ)] | [-10,10]^d | f(1,...,1)=0 | Levy variant |
| **Michalewicz** | -Σsin(xᵢ)sin²ᵐ(ixᵢ²/π) | [0,π]^d | Variable | Parameter m=10 |
| **Zakharov** | Σxᵢ² + (Σ0.5ixᵢ)² + (Σ0.5ixᵢ)⁴ | [-5,10]^d | f(0)=0 | Unimodal |
| **Salomon** | 1 - cos(2π‖x‖) + 0.1‖x‖ | [-100,100]^d | f(0)=0 | Multimodal |
| **Alpine 1** | Σ|xᵢsin(xᵢ) + 0.1xᵢ| | [-10,10]^d | f(0)=0 | Absolute values |
| **Alpine 2** | Π√xᵢsin(xᵢ) | [0,10]^d | f=2.808^d | Product form |
| **Schaffer F1** | 0.5 + (sin²(√(x₁²+x₂²)) - 0.5)/(1 + 0.001(x₁²+x₂²))² | [-100,100]² | f(0,0)=0 | 2D trigonometric |
| **Schaffer F2** | 0.5 + (sin²(x₁²-x₂²) - 0.5)/(1 + 0.001(x₁²+x₂²))² | [-100,100]² | f(0,0)=0 | 2D variant |
| **Schaffer F3** | 0.5 + (sin²(cos(|x₁²-x₂²|)) - 0.5)/(1 + 0.001(x₁²+x₂²))² | [-100,100]² | Variable | Complex 2D |
| **Schaffer F4** | 0.5 + (cos²(sin(|x₁²-x₂²|)) - 0.5)/(1 + 0.001(x₁²+x₂²))² | [-100,100]² | Variable | Complex 2D |
| **Schaffer F5** | 0.5 + (sin²(√(x₁²+x₂²)) - 0.5)/(1 + 0.001(x₁²+x₂²))² | [-100,100]² | Variable | 2D variant |
| **Schaffer F6** | 0.5 + (sin²(√(x₁²+x₂²)) - 0.5)/(1 + 0.001(x₁²+x₂²))² | [-100,100]² | Variable | Standard 2D |
| **Schaffer F7** | (√(x₁²+x₂²))^0.25 [sin²(50(√(x₁²+x₂²))^0.1) + 1] | [-100,100]² | f(0,0)=0 | Modulated |
| **Easom** | -cos(x₁)cos(x₂)exp(-(x₁-π)²-(x₂-π)²) | [-100,100]² | f(π,π)=-1 | Sharp minimum |
| **Cross-in-tray** | -0.0001[|sin(x₁)sin(x₂)exp(|100-√(x₁²+x₂²)/π|)| + 1]^0.1 | [-10,10]² | f=±2.06261 | 4 global minima |
| **Eggholder** | -(x₂+47)sin(√|x₂+x₁/2+47|) - x₁sin(√|x₁-(x₂+47)|) | [-512,512]² | f=-959.6407 | Complex landscape |
| **Holder Table** | -|sin(x₁)cos(x₂)exp(|1-√(x₁²+x₂²)/π|)| | [-10,10]² | f=-19.2085 | 4 symmetric minima |
| **Drop-wave** | -(1+cos(12√(x₁²+x₂²)))/(0.5(x₁²+x₂²)+2) | [-5.12,5.12]² | f(0,0)=-1 | Oscillatory |
| **Shubert** | [Σi·cos((i+1)x₁+i)][Σi·cos((i+1)x₂+i)] | [-10,10]² | f=-186.7309 | 18 global minima |
| **Powell** | Σ[(x₄ᵢ₋₃+10x₄ᵢ₋₂)² + 5(x₄ᵢ₋₁-x₄ᵢ)² + (x₄ᵢ₋₂-2x₄ᵢ₋₁)⁴ + 10(x₄ᵢ₋₃-x₄ᵢ)⁴] | [-4,5]^d | f(0)=0 | Groups of 4 |
| **Trid** | Σ(xᵢ-1)² - Σxᵢxᵢ₋₁ | Variable | Variable | Special structure |
| **Booth** | (x₁+2x₂-7)² + (2x₁+x₂-5)² | [-10,10]² | f(1,3)=0 | Quadratic |
| **Matyas** | 0.26(x₁²+x₂²) - 0.48x₁x₂ | [-10,10]² | f(0,0)=0 | Simple quadratic |
| **McCormick** | sin(x₁+x₂) + (x₁-x₂)² - 1.5x₁ + 2.5x₂ + 1 | Variable | f=-1.9133 | Trigonometric |
| **Six-Hump Camel** | (4-2.1x₁²+x₁⁴/3)x₁² + x₁x₂ + (-4+4x₂²)x₂² | Variable | f=-1.0316 | 6 local minima |
| **Goldstein-Price** | [1+(x₁+x₂+1)²(19-14x₁+3x₁²-14x₂+6x₁x₂+3x₂²)]×[30+(2x₁-3x₂)²(18-32x₁+12x₁²+48x₂-36x₁x₂+27x₂²)] | [-2,2]² | f(0,-1)=3 | Multimodal |
| **Beale** | (1.5-x₁+x₁x₂)² + (2.25-x₁+x₁x₂²)² + (2.625-x₁+x₁x₂³)² | [-4.5,4.5]² | f(3,0.5)=0 | Sharp peaks |
| **Branin** | (x₂-bx₁²+cx₁-r)² + s(1-t)cos(x₁) + s | [-5,10]×[0,15] | f=0.397887 | 3 global minima |
| **Hartmann 3D** | -Σαᵢexp(-Σ A_{ij}(x_j-P_{ij})²) | [0,1]³ | f=-3.86278 | 4 local minima |
| **Hartmann 6D** | -Σαᵢexp(-Σ A_{ij}(x_j-P_{ij})²) | [0,1]⁶ | f=-3.32237 | 6 local minima |

### Additional BoTorch-Specific Functions (12 functions)

| Function Name | Description | Domain | Global Min | Properties |
|---------------|-------------|--------|------------|------------|
| **Hartmann 4D** | 4D Hartmann variant | [0,1]⁴ | Variable | Intermediate complexity |
| **Shekel (m=5)** | 5 peaks Shekel | [0,10]⁴ | Variable | 5 local minima |
| **Shekel (m=7)** | 7 peaks Shekel | [0,10]⁴ | Variable | 7 local minima |
| **Shekel (m=10)** | 10 peaks Shekel | [0,10]⁴ | Variable | 10 local minina |
| **Sum Squares** | Σi·xᵢ² | [-10,10]^d | f(0)=0 | Weighted separable |
| **Forrester** | (6x-2)²sin(12x-4) | [0,1] | Variable | 1D multi-fidelity |
| **Perm** | Σ[Σ(j^k+β)(xⱼ^k-(1/j)^k)]² | Variable | f(1,1/2,...,1/d)=0 | Permutation structure |
| **Rotated Hyper-Ellipsoid** | Σ(Σxⱼ)² | [-65.536,65.536]^d | f(0)=0 | Rotated ellipsoid |
| **Sphere BoTorch** | Σxᵢ² | Variable | f(0)=0 | BoTorch implementation |
| **Three-Hump Camel** | 2x₁² - 1.05x₁⁴ + x₁⁶/6 + x₁x₂ + x₂² | [-5,5]² | f(0,0)=0 | 3 local minima |
| **Cosines** | -Σcos(5πxᵢ) | [0,1]^d | f=d | Trigonometric |
| **Exponential** | -exp(-0.5Σxᵢ²) | [-1,1]^d | f(0)=-1 | Exponential form |

## Summary Statistics

- **BBOB Core Suite**: 24 functions (most rigorous standard)
- **BBOB-largescale**: 24 functions (high-dimensional O(n) variants, 20-640D)
- **BBOB-mixint**: 24 functions (mixed-integer variants, 80% discrete)
- **Classical Mathematical**: 42 functions (widely used benchmarks)
- **BoTorch Additional**: 12 functions (modern implementations)

**Total: 126 distinct single-objective synthetic benchmark functions**

### Key BBOB Extensions:
- **bbob-largescale**: Uses permuted block-diagonal transformations instead of full orthogonal matrices, enabling efficient evaluation up to 640 dimensions with O(n) instead of O(n²) complexity
- **bbob-mixint**: Discretizes 80% of variables with 5 different arities (2,4,8,16,∞), testing mixed-integer optimization capabilities

These functions span complexity from simple unimodal (Sphere) to highly complex multimodal landscapes (Katsuura with >10^d optima), plus specialized variants for high-dimensional and mixed-integer optimization scenarios.