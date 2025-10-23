# Olympus Integration for BOMegaBench

This document describes the integration of Olympus benchmarks into BOMegaBench for testing Bayesian optimization algorithm performance.

## Overview

[Olympus](https://github.com/aspuru-guzik-group/olympus) is a benchmarking framework for optimization algorithms that provides:
- **Test Surfaces**: Synthetic optimization functions with various properties
- **Real-World Datasets**: Experimental data from chemistry and materials science

This integration brings unique Olympus benchmarks into BOMegaBench, avoiding duplicates with existing functions and focusing on valuable additions.

## What's Integrated

### 1. Olympus Surfaces (21+ unique functions)

Olympus surfaces that are NOT duplicated in BOMegaBench:

#### Categorical Variable Versions
Functions adapted for categorical (discrete choice) variables:
- `cat_ackley` - Categorical Ackley function
- `cat_camel` - Categorical Camel function
- `cat_dejong` - Categorical Dejong function
- `cat_michalewicz` - Categorical Michalewicz function
- `cat_slope` - Categorical Slope function

#### Discrete Versions
Functions with discrete (ordinal) variables:
- `discrete_ackley` - Discrete Ackley function
- `discrete_double_well` - Discrete double-well function
- `discrete_michalewicz` - Discrete Michalewicz function

#### Mountain/Terrain Functions
Special 2D visualization functions shaped like mountains:
- `denali` - Denali mountain surface
- `everest` - Everest mountain surface
- `k2` - K2 mountain surface
- `kilimanjaro` - Kilimanjaro mountain surface
- `matterhorn` - Matterhorn mountain surface
- `mont_blanc` - Mont Blanc mountain surface

#### Special Functions
Unique optimization landscapes:
- `ackley_path` - Ackley path function
- `gaussian_mixture` - Gaussian mixture surface
- `hyper_ellipsoid` - Hyper-ellipsoid function
- `linear_funnel` - Linear funnel function
- `narrow_funnel` - Narrow funnel function

#### Multi-Objective Functions
(Note: May require additional BOMegaBench support for multi-objective optimization)
- `mult_fonseca` - Fonseca multi-objective function
- `mult_viennet` - Viennet multi-objective function
- `mult_zdt1` - ZDT1 multi-objective function
- `mult_zdt2` - ZDT2 multi-objective function
- `mult_zdt3` - ZDT3 multi-objective function

### 2. Olympus Datasets (43 real-world problems)

Real experimental datasets organized by domain:

#### Chemical Reactions (14 datasets)
- Buchwald-Hartwig reactions: `buchwald_a`, `buchwald_b`, `buchwald_c`, `buchwald_d`, `buchwald_e`
- Suzuki-Miyaura reactions: `suzuki`, `suzuki_edbo`, `suzuki_i`, `suzuki_ii`, `suzuki_iii`, `suzuki_iv`
- Others: `benzylation`, `alkox`, `snar`

#### Materials Science (8 datasets)
- `perovskites` - Perovskite materials
- `fullerenes` - Fullerene synthesis
- `dye_lasers` - Organic dye lasers
- `redoxmers` - Redox-active molecules
- `colors_bob`, `colors_n9` - Color mixing
- `thin_film` - Thin film deposition
- `crossed_barrel` - Crossed barrel optimization

#### Photovoltaics (4 datasets)
- `photo_pce10`, `photo_wf3` - Photovoltaic optimization
- `p3ht` - P3HT polymer
- `mmli_opv` - Organic photovoltaics

#### Nanoparticles (3 datasets)
- `agnp` - Silver nanoparticles
- `lnp3` - Lipid nanoparticles
- `autoam` - Automated manufacturing

#### Electrochemistry (5 datasets)
- `electrochem` - General electrochemistry
- `oer_plate_3496`, `oer_plate_3851`, `oer_plate_3860`, `oer_plate_4098` - Oxygen evolution reaction plates

#### Liquids (7 datasets)
Solvent and liquid properties:
- `liquid_ace_100`, `liquid_dce`, `liquid_hep_100`, `liquid_thf_100`, `liquid_thf_500`, `liquid_toluene`, `liquid_water`

#### Other (2 datasets)
- `hplc` - HPLC optimization
- `vapdiff_crystal` - Vapor diffusion crystallization

## Usage

### Basic Usage

```python
from bomegabench.functions import (
    create_olympus_surfaces_suite,
    create_olympus_datasets_suite,
    OlympusDenaliFunction,
    OlympusSuzukiFunction,
)

# Create full suites
surfaces_suite = create_olympus_surfaces_suite()
datasets_suite = create_olympus_datasets_suite()

# Or use individual functions
denali = OlympusDenaliFunction()
suzuki = OlympusSuzukiFunction()

# Evaluate
import torch
X = torch.rand(10, denali.dim)
Y = denali(X)
```

### Category-Specific Suites

For datasets, you can create category-specific suites:

```python
from bomegabench.functions import (
    create_olympus_chemistry_suite,
    create_olympus_materials_suite,
    create_olympus_photovoltaics_suite,
)

# Only chemistry datasets
chemistry = create_olympus_chemistry_suite()

# Only materials datasets
materials = create_olympus_materials_suite()

# Only photovoltaics datasets
photovoltaics = create_olympus_photovoltaics_suite()
```

### Advanced: Custom Olympus Function

```python
from bomegabench.functions import OlympusSurfaceWrapper, OlympusDatasetWrapper

# Create custom surface
custom_surface = OlympusSurfaceWrapper(
    surface_name="CatAckley",
    dim=5,
    num_opts=21,  # For categorical surfaces
)

# Create custom dataset
custom_dataset = OlympusDatasetWrapper(
    dataset_name="buchwald_b",
    use_train_set=False,  # Use test set
)
```

## Testing

Run the integration test:

```bash
python examples/test_olympus_integration.py
```

This will:
1. Test loading of Olympus surfaces
2. Test loading of Olympus datasets
3. Verify function evaluation
4. List all available benchmarks

## Implementation Details

### Architecture

The integration uses wrapper classes to adapt Olympus interfaces to BOMegaBench:

- `OlympusSurfaceWrapper`: Wraps Olympus surfaces as `BenchmarkFunction`
- `OlympusDatasetWrapper`: Wraps Olympus datasets as `BenchmarkFunction`

Both wrappers:
- Convert Olympus parameter spaces to BOMegaBench bounds
- Handle PyTorch tensor conversion
- Support both continuous and categorical variables
- Provide metadata about the benchmark

### Dataset Evaluation

Olympus datasets use trained emulators to predict objective values:

1. **Primary method**: Use pre-trained Bayesian Neural Network emulators
2. **Fallback method**: Nearest neighbor interpolation from experimental data

This ensures datasets work even if emulator training hasn't been completed.

### Dependencies

The integration requires:
- `olympus` package (available in `olympus/` directory)
- Standard BOMegaBench dependencies (PyTorch, NumPy)

Optional (for full dataset support):
- Emulator models (may need separate download/training)

## Why These Benchmarks?

### Avoiding Duplicates

We carefully excluded Olympus surfaces already in BOMegaBench:
- Standard versions of: Ackley, Branin, Levy, Michalewicz, Rastrigin, Rosenbrock, Schwefel, Zakharov, Styblinski-Tang

### Unique Value

The integrated benchmarks provide:

1. **Categorical/Discrete variants**: Test BO algorithms on non-continuous spaces
2. **Mountain terrains**: Interesting 2D visualization benchmarks
3. **Real-world problems**: 43 datasets from actual experiments
4. **Domain diversity**: Chemistry, materials, energy, manufacturing

These are particularly valuable for:
- Testing mixed-variable BO algorithms
- Benchmarking on realistic experimental optimization problems
- Comparing algorithm performance across diverse domains

## Examples

See `examples/test_olympus_integration.py` for complete examples of:
- Creating and using Olympus surfaces
- Creating and using Olympus datasets
- Evaluating functions
- Listing available benchmarks

## References

- Olympus GitHub: https://github.com/aspuru-guzik-group/olympus
- Olympus Paper: Häse et al., "Olympus: A benchmarking framework for noisy optimization and experiment planning" (2020)
- BOMegaBench Documentation: See main README.md

## Future Work

Potential enhancements:
- Full multi-objective optimization support
- Emulator pre-training for all datasets
- Additional Olympus features (noise models, constraints)
- Integration with Olympus planners for baseline comparisons

## Troubleshooting

### "Olympus surfaces not available"

Make sure the `olympus/` directory is present and contains the Olympus source code.

### "Could not load emulator"

Some datasets may not have pre-trained emulators. The wrapper automatically falls back to nearest neighbor prediction from the experimental data.

### Import errors

If you see circular import errors from Olympus, this is expected and handled. The integration uses lazy loading to avoid these issues.

## Contributing

To add more Olympus benchmarks:

1. Check if the benchmark duplicates existing BOMegaBench functions
2. Add to `OLYMPUS_UNIQUE_SURFACES` or `OLYMPUS_DATASETS` in the respective module
3. Create a convenience class if commonly used
4. Update this documentation
5. Add tests in `examples/test_olympus_integration.py`
