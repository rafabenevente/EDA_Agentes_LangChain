"""
Configurações centralizadas para o projeto EDA Agentes LangChain.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configurações do projeto carregadas do arquivo .env"""
    
    # Google AI Configuration
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    google_model: str = Field(default="gemini-2.0-flash-lite", env="GOOGLE_MODEL")
    google_temperature: float = Field(default=0.7, env="GOOGLE_TEMPERATURE")
    google_max_tokens: int = Field(default=8192, env="GOOGLE_MAX_TOKENS")
    
    # Streamlit Configuration
    streamlit_server_port: int = Field(default=8501, env="STREAMLIT_SERVER_PORT")
    streamlit_server_address: str = Field(default="localhost", env="STREAMLIT_SERVER_ADDRESS")
    streamlit_theme_primary_color: str = Field(default="#FF6B6B", env="STREAMLIT_THEME_PRIMARY_COLOR")
    
    # Data Configuration
    max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    allowed_extensions: str = Field(default="csv", env="ALLOWED_EXTENSIONS")
    upload_folder: str = Field(default="data/uploads", env="UPLOAD_FOLDER")
    cache_folder: str = Field(default="data/cache", env="CACHE_FOLDER")
    
    # Agent Configuration
    agent_max_iterations: int = Field(default=20, env="AGENT_MAX_ITERATIONS")
    max_agent_iterations: int = Field(default=20, env="AGENT_MAX_ITERATIONS")  # Alias para compatibilidade
    agent_memory_window_size: int = Field(default=10, env="AGENT_MEMORY_WINDOW_SIZE")
    agent_temperature: float = Field(default=0.7, env="AGENT_TEMPERATURE")
    
    # Cache Configuration
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    streamlit_cache_ttl: int = Field(default=1800, env="STREAMLIT_CACHE_TTL")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/eda_agent.log", env="LOG_FILE")
    enable_verbose_logging: bool = Field(default=False, env="ENABLE_VERBOSE_LOGGING")
    
    # Performance Configuration
    max_rows_preview: int = Field(default=1000, env="MAX_ROWS_PREVIEW")
    max_correlation_matrix_size: int = Field(default=50, env="MAX_CORRELATION_MATRIX_SIZE")
    visualization_dpi: int = Field(default=150, env="VISUALIZATION_DPI")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def project_root(self) -> Path:
        """Retorna o diretório raiz do projeto"""
        return Path(__file__).parent.parent

    @property
    def upload_folder_path(self) -> Path:
        """Retorna o caminho absoluto para a pasta de uploads"""
        return self.project_root / self.upload_folder

    @property
    def cache_folder_path(self) -> Path:
        """Retorna o caminho absoluto para a pasta de cache"""
        return self.project_root / self.cache_folder

    @property
    def log_file_path(self) -> Path:
        """Retorna o caminho absoluto para o arquivo de log"""
        return self.project_root / self.log_file

    def ensure_directories(self) -> None:
        """Garante que os diretórios necessários existam"""
        self.upload_folder_path.mkdir(parents=True, exist_ok=True)
        self.cache_folder_path.mkdir(parents=True, exist_ok=True)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)


# Instância global das configurações
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Retorna uma instância singleton das configurações"""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings


def load_settings() -> Settings:
    """Alias para get_settings() para compatibilidade"""
    return get_settings()


# Configurações para diferentes ambientes
class DevelopmentSettings(Settings):
    """Configurações específicas para desenvolvimento"""
    log_level: str = "DEBUG"
    enable_verbose_logging: bool = True
    enable_cache: bool = False


class ProductionSettings(Settings):
    """Configurações específicas para produção"""
    log_level: str = "WARNING"
    enable_verbose_logging: bool = False
    enable_cache: bool = True


def get_settings_for_environment(env: str = None) -> Settings:
    """Retorna configurações baseadas no ambiente especificado"""
    if env is None:
        env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "development":
        return DevelopmentSettings()
    else:
        return Settings()


# Instância global das configurações
settings = get_settings()