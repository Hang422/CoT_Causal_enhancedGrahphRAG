from typing import Dict, Optional, Any, List
import yaml
import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class DatabaseConfig:
    """Configuration for a single Neo4j database"""
    name: str
    description: str
    max_connections: int


@dataclass
class Neo4jConfig:
    """Neo4j configuration"""
    uri: str
    username: str
    password: str
    databases: Dict[str, DatabaseConfig]


@dataclass
class OpenAIConfig:
    """OpenAI configuration"""
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    retry_attempts: int
    timeout: int


@dataclass
class EntityConfig:
    """Entity processing configuration"""
    threshold: float
    batch_size: int
    cache_size: int


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str
    file: str
    format: str


@dataclass
class Config:
    """Main configuration class"""
    neo4j: Neo4jConfig
    openai: OpenAIConfig
    entity_processing: EntityConfig
    logging: LoggingConfig

    @classmethod
    def load(cls, env: Optional[str] = None) -> 'Config':
        """Load configuration based on environment

        Args:
            env: Environment name (development, production, testing)
                If None, uses ENVIRONMENT env variable or defaults to development

        Returns:
            Config object
        """
        if env is None:
            env = os.getenv('ENVIRONMENT', 'development')

        config_path = Path(__file__).parent.parent.parent / 'config' / f'{env}.yaml'

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # Replace environment variables in config
        config_data = cls._replace_env_vars(config_data)

        # Create database configs
        db_configs = {
            name: DatabaseConfig(**db_config)
            for name, db_config in config_data['neo4j']['databases'].items()
        }

        return cls(
            neo4j=Neo4jConfig(
                uri=config_data['neo4j']['uri'],
                username=config_data['neo4j']['username'],
                password=config_data['neo4j']['password'],
                databases=db_configs
            ),
            openai=OpenAIConfig(**config_data['openai']),
            entity_processing=EntityConfig(**config_data['entity_processing']),
            logging=LoggingConfig(**config_data['logging'])
        )

    @staticmethod
    def _replace_env_vars(config: dict) -> str | dict[Any, dict] | list[dict] | dict:
        """Recursively replace environment variables in config"""
        if isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        elif isinstance(config, dict):
            return {k: Config._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [Config._replace_env_vars(v) for v in config]
        return config