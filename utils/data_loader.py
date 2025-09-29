"""
Carregador de dados CSV com validação e detecção automática de encoding.
"""

import io
import logging
import pandas as pd
import chardet
from pathlib import Path
from typing import Optional, Union, Dict, Any
from config.settings import get_settings

logger = logging.getLogger(__name__)


class DataLoader:
    """Classe responsável por carregar e validar arquivos CSV"""
    
    def __init__(self):
        self.settings = get_settings()
        self.max_file_size = self.settings.max_file_size_mb * 1024 * 1024  # Convert to bytes
    
    def detect_encoding(self, file_content: bytes) -> str:
        """Detecta automaticamente o encoding do arquivo"""
        try:
            result = chardet.detect(file_content)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            logger.info(f"Encoding detectado: {encoding} (confiança: {confidence:.2f})")
            
            # Se a confiança for muito baixa, usar utf-8 como padrão
            if confidence < 0.7:
                logger.warning(f"Baixa confiança na detecção de encoding. Usando utf-8 como padrão.")
                return 'utf-8'
            
            return encoding
        except Exception as e:
            logger.error(f"Erro na detecção de encoding: {e}")
            return 'utf-8'
    
    def validate_file_size(self, file_size: int) -> bool:
        """Valida se o arquivo não excede o tamanho máximo permitido"""
        if file_size > self.max_file_size:
            raise ValueError(
                f"Arquivo muito grande: {file_size / (1024*1024):.2f}MB. "
                f"Tamanho máximo permitido: {self.settings.max_file_size_mb}MB"
            )
        return True
    
    def validate_file_extension(self, filename: str) -> bool:
        """Valida se a extensão do arquivo é permitida"""
        allowed_extensions = self.settings.allowed_extensions.split(',')
        file_extension = Path(filename).suffix.lower().lstrip('.')
        
        if file_extension not in allowed_extensions:
            raise ValueError(
                f"Extensão de arquivo não permitida: .{file_extension}. "
                f"Extensões permitidas: {', '.join([f'.{ext}' for ext in allowed_extensions])}"
            )
        return True
    
    def load_csv(self, file_upload) -> Optional[pd.DataFrame]:
        """
        Carrega um arquivo CSV enviado via Streamlit
        
        Args:
            file_upload: Objeto de arquivo do Streamlit
            
        Returns:
            DataFrame do pandas ou None se houver erro
        """
        try:
            # Validar nome do arquivo
            if hasattr(file_upload, 'name'):
                self.validate_file_extension(file_upload.name)
            
            # Ler conteúdo do arquivo
            if isinstance(file_upload, str):
                # Se for um caminho de arquivo
                with open(file_upload, 'rb') as f:
                    file_content = f.read()
                file_size = len(file_content)
            else:
                # Se for um objeto de arquivo
                file_content = file_upload.read()
                file_size = len(file_content)
            
            # Validar tamanho
            self.validate_file_size(file_size)
            
            # Detectar encoding
            encoding = self.detect_encoding(file_content)
            
            # Tentar carregar o CSV
            try:
                # Primeiro, tenta com o encoding detectado
                df = pd.read_csv(
                    io.StringIO(file_content.decode(encoding)),
                    encoding=encoding
                )
            except UnicodeDecodeError:
                # Se falhar, tenta com utf-8
                logger.warning("Falha no encoding detectado, tentando utf-8")
                df = pd.read_csv(
                    io.StringIO(file_content.decode('utf-8', errors='ignore')),
                    encoding='utf-8'
                )
            except Exception as e:
                # Última tentativa com latin-1
                logger.warning(f"Falha com utf-8, tentando latin-1: {e}")
                df = pd.read_csv(
                    io.StringIO(file_content.decode('latin-1')),
                    encoding='latin-1'
                )
            
            # Validações básicas do DataFrame
            if df.empty:
                raise ValueError("O arquivo CSV está vazio")
            
            if len(df.columns) == 0:
                raise ValueError("O arquivo CSV não possui colunas válidas")
            
            # Log informações do dataset
            logger.info(f"CSV carregado com sucesso: {len(df)} linhas, {len(df.columns)} colunas")
            logger.info(f"Colunas: {list(df.columns)}")
            logger.info(f"Tipos de dados: {df.dtypes.to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar CSV: {e}")
            raise e
    
    def load_csv_from_path(self, file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Carrega um arquivo CSV a partir de um caminho no sistema de arquivos
        
        Args:
            file_path: Caminho para o arquivo CSV
            
        Returns:
            DataFrame do pandas ou None se houver erro
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
            
            # Validar extensão
            self.validate_file_extension(file_path.name)
            
            # Validar tamanho
            file_size = file_path.stat().st_size
            self.validate_file_size(file_size)
            
            # Ler arquivo e detectar encoding
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            encoding = self.detect_encoding(file_content)
            
            # Carregar CSV
            df = pd.read_csv(file_path, encoding=encoding)
            
            logger.info(f"CSV carregado de {file_path}: {len(df)} linhas, {len(df.columns)} colunas")
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar CSV de {file_path}: {e}")
            raise e
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Retorna informações resumidas sobre o dataset
        
        Args:
            df: DataFrame do pandas
            
        Returns:
            Dicionário com informações do dataset
        """
        try:
            info = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'numeric_columns': list(df.select_dtypes(include=['number']).columns),
                'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
                'datetime_columns': list(df.select_dtypes(include=['datetime']).columns),
                'has_duplicates': df.duplicated().any(),
                'total_duplicates': df.duplicated().sum()
            }
            
            # Adicionar estatísticas básicas para colunas numéricas
            if info['numeric_columns']:
                info['numeric_stats'] = df[info['numeric_columns']].describe().to_dict()
            
            return info
            
        except Exception as e:
            logger.error(f"Erro ao obter informações do dataset: {e}")
            return {}
    
    def save_uploaded_file(self, file_upload, filename: Optional[str] = None) -> Path:
        """
        Salva um arquivo enviado na pasta de uploads
        
        Args:
            file_upload: Objeto de arquivo do Streamlit
            filename: Nome personalizado para o arquivo (opcional)
            
        Returns:
            Caminho para o arquivo salvo
        """
        try:
            if filename is None:
                filename = file_upload.name
            
            file_path = self.settings.upload_folder_path / filename
            
            # Garantir que a pasta existe
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Salvar arquivo
            with open(file_path, 'wb') as f:
                f.write(file_upload.read())
            
            logger.info(f"Arquivo salvo em: {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Erro ao salvar arquivo: {e}")
            raise e