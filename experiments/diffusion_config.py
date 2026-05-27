"""
Diffusion Study Configuration System
Defines configuration spaces for systematic parameter studies in HyperGRAND.
Tests impact of integration schemes, diffusion depth, and attention mechanisms.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
import itertools


@dataclass
class DiffusionStudyConfig:
    """Configuration for a single diffusion study run"""
    
    # Study metadata
    study_dimension: str  # 'integration_scheme', 'diffusion_depth', 'attention_mechanism'
    config_variant: str   # Specific value being tested
    
    # Model architecture
    hidden_dim: int = 32
    num_layers: int = 3
    alpha: float = 0.1
    dropout: float = 0.1
    integration_scheme: str = 'explicit'
    
    # Attention mechanism
    use_attention: bool = True
    attention_type: str = 'full'  # 'full', 'none', 'simplified'
    
    # Training parameters
    epochs: int = 200
    patience: int = 50
    lr: float = 0.01
    weight_decay: float = 1e-5
    
    # Early termination
    enable_early_termination: bool = True
    baseline_multiplier: float = 2.0
    termination_check_epoch: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'study_dimension': self.study_dimension,
            'config_variant': self.config_variant,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'integration_scheme': self.integration_scheme,
            'use_attention': self.use_attention,
            'attention_type': self.attention_type,
            'epochs': self.epochs,
            'patience': self.patience,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'DiffusionStudyConfig':
        """Create config from dictionary"""
        return cls(**config)


class DiffusionStudySpace:
    """Defines the configuration space for diffusion studies"""
    
    # Integration scheme study configurations
    INTEGRATION_SCHEMES = {
        'explicit': {
            'integration_scheme': 'explicit',
            'description': 'Forward Euler (explicit) integration - fast but may be unstable',
            'expected_performance': 'Best for clustering (fast convergence)',
        },
        'implicit': {
            'integration_scheme': 'implicit',
            'description': 'Backward Euler (implicit) integration - stable but slower',
            'expected_performance': 'Best for classification (stability over long horizons)',
        },
        'adaptive': {
            'integration_scheme': 'adaptive',
            'description': 'RK45-style adaptive stepping - dynamic step size',
            'expected_performance': 'Balanced performance across tasks',
        },
    }
    
    # Diffusion depth (number of layers) study configurations
    DIFFUSION_DEPTHS = {
        '1_layer': {
            'num_layers': 1,
            'description': 'Single diffusion step - minimal time discretization',
            'expected_performance': 'Under-smoothing, may preserve local structure',
        },
        '2_layers': {
            'num_layers': 2,
            'description': 'Two diffusion steps - light smoothing',
            'expected_performance': 'Good for small hypergraphs',
        },
        '3_layers': {
            'num_layers': 3,
            'description': 'Three diffusion steps - moderate smoothing (baseline)',
            'expected_performance': 'Balanced performance',
        },
        '4_layers': {
            'num_layers': 4,
            'description': 'Four diffusion steps - strong smoothing',
            'expected_performance': 'Good for noisy data',
        },
        '5_layers': {
            'num_layers': 5,
            'description': 'Five diffusion steps - maximum smoothing',
            'expected_performance': 'Over-smoothing may occur',
        },
    }
    
    # Attention mechanism study configurations
    ATTENTION_MECHANISMS = {
        'full_attention': {
            'use_attention': True,
            'attention_type': 'full',
            'description': 'Full learned attention matrix G with W_K and W_Q',
            'expected_performance': 'Best overall - learned diffusion tensor',
        },
        'no_attention': {
            'use_attention': False,
            'attention_type': 'none',
            'description': 'Identity diffusion tensor (no attention)',
            'expected_performance': 'Simple baseline - uniform diffusion',
        },
        'simplified_attention': {
            'use_attention': True,
            'attention_type': 'simplified',
            'description': 'Dot product attention without learnable projections',
            'expected_performance': 'Middle ground - some adaptation',
        },
    }
    
    @classmethod
    def get_baseline_config(cls, task_type: str = 'classification') -> DiffusionStudyConfig:
        """Get baseline configuration for a task type"""
        baseline_configs = {
            'classification': DiffusionStudyConfig(
                study_dimension='baseline',
                config_variant='baseline_classification',
                hidden_dim=64,
                num_layers=3,
                alpha=0.05,
                dropout=0.3,
                integration_scheme='implicit',
                lr=0.01,
                weight_decay=5e-4,
            ),
            'clustering': DiffusionStudyConfig(
                study_dimension='baseline',
                config_variant='baseline_clustering',
                hidden_dim=32,
                num_layers=2,
                alpha=0.1,
                dropout=0.1,
                integration_scheme='explicit',
                lr=0.001,
                weight_decay=1e-5,
            ),
            'partitioning': DiffusionStudyConfig(
                study_dimension='baseline',
                config_variant='baseline_partitioning',
                hidden_dim=48,
                num_layers=2,
                alpha=0.08,
                dropout=0.2,
                integration_scheme='adaptive',
                lr=0.005,
                weight_decay=1e-4,
            ),
        }
        return baseline_configs.get(task_type, baseline_configs['classification'])
    
    @classmethod
    def generate_integration_scheme_configs(cls, baseline: DiffusionStudyConfig) -> List[DiffusionStudyConfig]:
        """Generate configs for integration scheme study"""
        configs = []
        for variant_name, variant_config in cls.INTEGRATION_SCHEMES.items():
            config = DiffusionStudyConfig(
                study_dimension='integration_scheme',
                config_variant=variant_name,
                hidden_dim=baseline.hidden_dim,
                num_layers=baseline.num_layers,
                alpha=baseline.alpha,
                dropout=baseline.dropout,
                integration_scheme=variant_config['integration_scheme'],
                use_attention=baseline.use_attention,
                attention_type=baseline.attention_type,
                epochs=baseline.epochs,
                patience=baseline.patience,
                lr=baseline.lr,
                weight_decay=baseline.weight_decay,
            )
            configs.append(config)
        return configs
    
    @classmethod
    def generate_diffusion_depth_configs(cls, baseline: DiffusionStudyConfig) -> List[DiffusionStudyConfig]:
        """Generate configs for diffusion depth study"""
        configs = []
        for variant_name, variant_config in cls.DIFFUSION_DEPTHS.items():
            config = DiffusionStudyConfig(
                study_dimension='diffusion_depth',
                config_variant=variant_name,
                hidden_dim=baseline.hidden_dim,
                num_layers=variant_config['num_layers'],
                alpha=baseline.alpha,
                dropout=baseline.dropout,
                integration_scheme=baseline.integration_scheme,
                use_attention=baseline.use_attention,
                attention_type=baseline.attention_type,
                epochs=baseline.epochs,
                patience=baseline.patience,
                lr=baseline.lr,
                weight_decay=baseline.weight_decay,
            )
            configs.append(config)
        return configs
    
    @classmethod
    def generate_attention_mechanism_configs(cls, baseline: DiffusionStudyConfig) -> List[DiffusionStudyConfig]:
        """Generate configs for attention mechanism study"""
        configs = []
        for variant_name, variant_config in cls.ATTENTION_MECHANISMS.items():
            config = DiffusionStudyConfig(
                study_dimension='attention_mechanism',
                config_variant=variant_name,
                hidden_dim=baseline.hidden_dim,
                num_layers=baseline.num_layers,
                alpha=baseline.alpha,
                dropout=baseline.dropout,
                integration_scheme=baseline.integration_scheme,
                use_attention=variant_config['use_attention'],
                attention_type=variant_config['attention_type'],
                epochs=baseline.epochs,
                patience=baseline.patience,
                lr=baseline.lr,
                weight_decay=baseline.weight_decay,
            )
            configs.append(config)
        return configs
    
    @classmethod
    def generate_all_study_configs(
        cls, 
        task_type: str = 'classification',
        study_dimensions: List[str] = None
    ) -> Dict[str, List[DiffusionStudyConfig]]:
        """
        Generate all study configurations
        
        Args:
            task_type: Task type for baseline config
            study_dimensions: List of dimensions to study. If None, studies all dimensions.
                            Options: 'integration_scheme', 'diffusion_depth', 'attention_mechanism'
        
        Returns:
            Dictionary mapping study dimension to list of configs
        """
        if study_dimensions is None:
            study_dimensions = ['integration_scheme', 'diffusion_depth', 'attention_mechanism']
        
        baseline = cls.get_baseline_config(task_type)
        all_configs = {'baseline': [baseline]}
        
        if 'integration_scheme' in study_dimensions:
            all_configs['integration_scheme'] = cls.generate_integration_scheme_configs(baseline)
        
        if 'diffusion_depth' in study_dimensions:
            all_configs['diffusion_depth'] = cls.generate_diffusion_depth_configs(baseline)
        
        if 'attention_mechanism' in study_dimensions:
            all_configs['attention_mechanism'] = cls.generate_attention_mechanism_configs(baseline)
        
        return all_configs
    
    @classmethod
    def get_fast_mode_configs(cls, task_type: str = 'classification') -> Dict[str, List[DiffusionStudyConfig]]:
        """
        Get reduced configuration set for fast hypothesis testing
        Tests only critical variants: explicit vs implicit, 1 vs 3 vs 5 layers
        """
        baseline = cls.get_baseline_config(task_type)
        
        # Integration scheme: only explicit vs implicit (skip adaptive)
        integration_configs = [
            DiffusionStudyConfig(
                study_dimension='integration_scheme',
                config_variant='explicit',
                integration_scheme='explicit',
                **{k: v for k, v in baseline.to_dict().items() 
                   if k not in ['study_dimension', 'config_variant', 'integration_scheme']}
            ),
            DiffusionStudyConfig(
                study_dimension='integration_scheme',
                config_variant='implicit',
                integration_scheme='implicit',
                **{k: v for k, v in baseline.to_dict().items() 
                   if k not in ['study_dimension', 'config_variant', 'integration_scheme']}
            ),
        ]
        
        # Diffusion depth: only 1, 3, 5 layers
        depth_configs = [
            DiffusionStudyConfig(
                study_dimension='diffusion_depth',
                config_variant='1_layer',
                num_layers=1,
                **{k: v for k, v in baseline.to_dict().items() 
                   if k not in ['study_dimension', 'config_variant', 'num_layers']}
            ),
            DiffusionStudyConfig(
                study_dimension='diffusion_depth',
                config_variant='3_layers',
                num_layers=3,
                **{k: v for k, v in baseline.to_dict().items() 
                   if k not in ['study_dimension', 'config_variant', 'num_layers']}
            ),
            DiffusionStudyConfig(
                study_dimension='diffusion_depth',
                config_variant='5_layers',
                num_layers=5,
                **{k: v for k, v in baseline.to_dict().items() 
                   if k not in ['study_dimension', 'config_variant', 'num_layers']}
            ),
        ]
        
        return {
            'baseline': [baseline],
            'integration_scheme': integration_configs,
            'diffusion_depth': depth_configs,
        }


def get_study_execution_plan(
    datasets: List[str],
    task_types: Dict[str, str],
    study_configs: Dict[str, List[DiffusionStudyConfig]],
    num_seeds: int = 5
) -> Dict[str, Any]:
    """
    Generate execution plan for diffusion studies
    
    Args:
        datasets: List of dataset names
        task_types: Mapping of dataset name to task type
        study_configs: Dict of study dimension to configs
        num_seeds: Number of random seeds per configuration
    
    Returns:
        Execution plan with total runs, estimated time, etc.
    """
    total_runs = 0
    study_breakdown = {}
    
    for study_dim, configs in study_configs.items():
        num_configs = len(configs)
        runs_per_study = len(datasets) * num_configs * num_seeds
        total_runs += runs_per_study
        study_breakdown[study_dim] = {
            'num_configs': num_configs,
            'total_runs': runs_per_study,
        }
    
    # Estimate time (rough: 5 min/run average)
    estimated_minutes = total_runs * 5
    estimated_hours = estimated_minutes / 60
    
    return {
        'total_runs': total_runs,
        'num_datasets': len(datasets),
        'num_seeds': num_seeds,
        'study_breakdown': study_breakdown,
        'estimated_minutes': estimated_minutes,
        'estimated_hours': estimated_hours,
    }
