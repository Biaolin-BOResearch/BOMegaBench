"""
Benchmark function implementations organized by suites.
"""

# Import synthetic functions suite
from .synthetic_functions import create_synthetic_suite
SyntheticSuite = create_synthetic_suite()

# Import LassoBench suites (with optional dependency)
try:
    from .lasso_bench import (
        LassoBenchSyntheticSuite,
        LassoBenchRealSuite
    )
    LASSO_BENCH_AVAILABLE = True
except ImportError:
    LassoBenchSyntheticSuite = None
    LassoBenchRealSuite = None
    LASSO_BENCH_AVAILABLE = False

# Import HPO benchmarks (with optional dependency)
try:
    from .hpo_benchmarks import HPOBenchmarksSuite
    HPO_AVAILABLE = True
except ImportError:
    HPOBenchmarksSuite = None
    HPO_AVAILABLE = False

# Import HPOBench benchmarks (with optional dependency)
try:
    from .hpobench_benchmarks import (HPOBenchMLSuite, HPOBenchODSuite,
                                      HPOBenchNASSuite, HPOBenchRLSuite,
                                      HPOBenchSurrogatesSuite)
    HPOBENCH_AVAILABLE = True
except ImportError:
    HPOBenchMLSuite = None
    HPOBenchODSuite = None
    HPOBenchNASSuite = None
    HPOBenchRLSuite = None
    HPOBenchSurrogatesSuite = None
    HPOBENCH_AVAILABLE = False

# Import Database Tuning benchmarks
try:
    from .database import (
        DatabaseTuningSuite,
        DatabaseTuningFunction,
        create_database_tuning_suite
    )
    DATABASE_TUNING_AVAILABLE = True
except ImportError:
    DatabaseTuningSuite = None
    DatabaseTuningFunction = None
    create_database_tuning_suite = None
    DATABASE_TUNING_AVAILABLE = False

# Molecular Optimization benchmarks - REMOVED
# Reason: MolOpt uses SMILES string inputs, which are not supported in current testing scope
MOL_OPT_AVAILABLE = False

# Import Olympus Surfaces (with optional dependency)
try:
    from .olympus_surfaces import (
        create_olympus_surfaces_suite,
        OlympusSurfaceWrapper,
        OlympusDenaliFunction,
        OlympusEverestFunction,
        OlympusCatAckleyFunction,
        OlympusDiscreteAckleyFunction,
        OlympusGaussianMixtureFunction,
        OlympusLinearFunnelFunction,
    )
    OLYMPUS_SURFACES_AVAILABLE = True
except ImportError as e:
    print(f"Olympus surfaces not available: {e}")
    create_olympus_surfaces_suite = None
    OlympusSurfaceWrapper = None
    OlympusDenaliFunction = None
    OlympusEverestFunction = None
    OlympusCatAckleyFunction = None
    OlympusDiscreteAckleyFunction = None
    OlympusGaussianMixtureFunction = None
    OlympusLinearFunnelFunction = None
    OLYMPUS_SURFACES_AVAILABLE = False

# Import Olympus Datasets (with optional dependency)
try:
    from .olympus_datasets import (
        create_olympus_datasets_suite,
        create_olympus_chemistry_suite,
        create_olympus_materials_suite,
        create_olympus_photovoltaics_suite,
        OlympusDatasetWrapper,
        OlympusBuchwaldAFunction,
        OlympusSuzukiFunction,
        OlympusPerovskitesFunction,
        OlympusDyeLasersFunction,
    )
    OLYMPUS_DATASETS_AVAILABLE = True
except ImportError as e:
    print(f"Olympus datasets not available: {e}")
    create_olympus_datasets_suite = None
    create_olympus_chemistry_suite = None
    create_olympus_materials_suite = None
    create_olympus_photovoltaics_suite = None
    OlympusDatasetWrapper = None
    OlympusBuchwaldAFunction = None
    OlympusSuzukiFunction = None
    OlympusPerovskitesFunction = None
    OlympusDyeLasersFunction = None
    OLYMPUS_DATASETS_AVAILABLE = False

# Import MuJoCo Control Tasks (with optional dependency)
try:
    from .mujoco_control import (
        create_mujoco_control_suite,
        MuJoCoControlWrapper,
        HalfCheetahLinearFunction,
        HopperLinearFunction,
        Walker2dLinearFunction,
        AntLinearFunction,
        HumanoidLinearFunction,
    )
    MUJOCO_AVAILABLE = True
except ImportError as e:
    print(f"MuJoCo not available: {e}")
    create_mujoco_control_suite = None
    MuJoCoControlWrapper = None
    HalfCheetahLinearFunction = None
    HopperLinearFunction = None
    Walker2dLinearFunction = None
    AntLinearFunction = None
    HumanoidLinearFunction = None
    MUJOCO_AVAILABLE = False

# Import Design-Bench Tasks (non-overlapping only, with optional dependency)
try:
    from .design_bench_tasks import (
        create_design_bench_suite,
        DesignBenchWrapper,
        SuperconductorRFFunction,
        GFPTransformerFunction,
        TFBind8ExactFunction,
        UTRTransformerFunction,
    )
    DESIGN_BENCH_AVAILABLE = True
except ImportError as e:
    print(f"Design-Bench not available: {e}")
    create_design_bench_suite = None
    DesignBenchWrapper = None
    SuperconductorRFFunction = None
    GFPTransformerFunction = None
    TFBind8ExactFunction = None
    UTRTransformerFunction = None
    DESIGN_BENCH_AVAILABLE = False

# Import Robosuite Manipulation Tasks (with optional dependency)
try:
    from .robosuite_manipulation import (
        create_robosuite_manipulation_suite,
        RobosuiteManipulationWrapper,
        LiftLinearFunction,
        DoorLinearFunction,
        StackLinearFunction,
        NutAssemblyLinearFunction,
        PickPlaceLinearFunction,
        MANIPULATION_ENVS,
    )
    ROBOSUITE_AVAILABLE = True
except ImportError as e:
    print(f"Robosuite not available: {e}")
    create_robosuite_manipulation_suite = None
    RobosuiteManipulationWrapper = None
    LiftLinearFunction = None
    DoorLinearFunction = None
    StackLinearFunction = None
    NutAssemblyLinearFunction = None
    PickPlaceLinearFunction = None
    MANIPULATION_ENVS = None
    ROBOSUITE_AVAILABLE = False

# Import HumanoidBench Tasks (with optional dependency)
try:
    from .humanoid_bench_tasks import (
        create_humanoid_bench_suite,
        HumanoidBenchWrapper,
        H1HandWalkFunction,
        H1HandPushFunction,
        H1HandCabinetFunction,
        H1HandDoorFunction,
        G1WalkFunction,
        G1PushFunction,
        LOCOMOTION_TASKS,
        MANIPULATION_TASKS as HUMANOID_MANIPULATION_TASKS,
        ALL_TASKS as HUMANOID_ALL_TASKS,
        AVAILABLE_ROBOTS,
    )
    HUMANOID_BENCH_AVAILABLE = True
except ImportError as e:
    print(f"HumanoidBench not available: {e}")
    create_humanoid_bench_suite = None
    HumanoidBenchWrapper = None
    H1HandWalkFunction = None
    H1HandPushFunction = None
    H1HandCabinetFunction = None
    H1HandDoorFunction = None
    G1WalkFunction = None
    G1PushFunction = None
    LOCOMOTION_TASKS = None
    HUMANOID_MANIPULATION_TASKS = None
    HUMANOID_ALL_TASKS = None
    AVAILABLE_ROBOTS = None
    HUMANOID_BENCH_AVAILABLE = False

from .registry import (
    get_function,
    list_functions,
    list_suites,
    get_suite,
    get_function_summary,
    get_multimodal_functions,
    get_unimodal_functions,
    get_functions_by_property,
)

__all__ = [
    "SyntheticSuite",
    "get_function",
    "list_functions",
    "list_suites",
    "get_suite",
    "get_function_summary",
    "get_multimodal_functions",
    "get_unimodal_functions",
    "get_functions_by_property",
]

# Add LassoBench suites to exports if available
if LASSO_BENCH_AVAILABLE:
    __all__.extend([
        "LassoBenchSyntheticSuite",
        "LassoBenchRealSuite"
    ])

# Add HPO benchmarks to exports if available
if HPO_AVAILABLE:
    __all__.extend([
        "HPOBenchmarksSuite"
    ])

# Add HPOBench benchmarks to exports if available
if HPOBENCH_AVAILABLE:
    __all__.extend([
        "HPOBenchMLSuite",
        "HPOBenchODSuite",
        "HPOBenchNASSuite",
        "HPOBenchRLSuite",
        "HPOBenchSurrogatesSuite"
    ])

# Add Database Tuning benchmarks to exports if available
if DATABASE_TUNING_AVAILABLE:
    __all__.extend([
        "DatabaseTuningSuite",
        "DatabaseTuningFunction",
        "create_database_tuning_suite"
    ])

# Molecular Optimization benchmarks - REMOVED (not exported)

# Add Olympus Surfaces to exports if available
if OLYMPUS_SURFACES_AVAILABLE:
    __all__.extend([
        "create_olympus_surfaces_suite",
        "OlympusSurfaceWrapper",
        "OlympusDenaliFunction",
        "OlympusEverestFunction",
        "OlympusCatAckleyFunction",
        "OlympusDiscreteAckleyFunction",
        "OlympusGaussianMixtureFunction",
        "OlympusLinearFunnelFunction",
    ])

# Add Olympus Datasets to exports if available
if OLYMPUS_DATASETS_AVAILABLE:
    __all__.extend([
        "create_olympus_datasets_suite",
        "create_olympus_chemistry_suite",
        "create_olympus_materials_suite",
        "create_olympus_photovoltaics_suite",
        "OlympusDatasetWrapper",
        "OlympusBuchwaldAFunction",
        "OlympusSuzukiFunction",
        "OlympusPerovskitesFunction",
        "OlympusDyeLasersFunction",
    ])

# Add MuJoCo Control Tasks to exports if available
if MUJOCO_AVAILABLE:
    __all__.extend([
        "create_mujoco_control_suite",
        "MuJoCoControlWrapper",
        "HalfCheetahLinearFunction",
        "HopperLinearFunction",
        "Walker2dLinearFunction",
        "AntLinearFunction",
        "HumanoidLinearFunction",
    ])

# Add Design-Bench Tasks to exports if available
if DESIGN_BENCH_AVAILABLE:
    __all__.extend([
        "create_design_bench_suite",
        "DesignBenchWrapper",
        "SuperconductorRFFunction",
        "GFPTransformerFunction",
        "TFBind8ExactFunction",
        "UTRTransformerFunction",
    ])

# Add Robosuite Manipulation Tasks to exports if available
if ROBOSUITE_AVAILABLE:
    __all__.extend([
        "create_robosuite_manipulation_suite",
        "RobosuiteManipulationWrapper",
        "LiftLinearFunction",
        "DoorLinearFunction",
        "StackLinearFunction",
        "NutAssemblyLinearFunction",
        "PickPlaceLinearFunction",
        "MANIPULATION_ENVS",
    ])

# Add HumanoidBench Tasks to exports if available
if HUMANOID_BENCH_AVAILABLE:
    __all__.extend([
        "create_humanoid_bench_suite",
        "HumanoidBenchWrapper",
        "H1HandWalkFunction",
        "H1HandPushFunction",
        "H1HandCabinetFunction",
        "H1HandDoorFunction",
        "G1WalkFunction",
        "G1PushFunction",
        "LOCOMOTION_TASKS",
        "HUMANOID_MANIPULATION_TASKS",
        "HUMANOID_ALL_TASKS",
        "AVAILABLE_ROBOTS",
    ])