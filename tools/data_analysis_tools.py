"""
Ferramentas customizadas para análise de dados CSV
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from langchain.tools import tool
from utils.visualization_helpers import VisualizationHelpers

logger = logging.getLogger(__name__)
viz_helper = VisualizationHelpers()

def get_current_dataframe() -> Optional[pd.DataFrame]:
    """Retorna o dataframe atual (importado do agente principal)"""
    try:
        from agents.eda_agent import get_current_dataframe as get_df
        return get_df()
    except ImportError:
        # Fallback para desenvolvimento/teste
        return None


@tool
def describe_dataset() -> Dict[str, Any]:
    """
    Fornece uma descrição estatística completa do dataset atual.
    
    Returns:
        Dicionário com informações estatísticas do dataset
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        result = {
            "shape": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "duplicates": {
                "has_duplicates": df.duplicated().any(),
                "total_duplicates": df.duplicated().sum(),
                "duplicate_percentage": df.duplicated().sum() / len(df) * 100
            }
        }
        
        # Estatísticas para colunas numéricas
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            result["numeric_summary"] = df[numeric_columns].describe().to_dict()
        
        # Informações sobre colunas categóricas
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            categorical_info = {}
            for col in categorical_columns:
                categorical_info[col] = {
                    "unique_values": df[col].nunique(),
                    "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    "frequency_of_most_frequent": df[col].value_counts().iloc[0] if not df[col].empty else 0
                }
            result["categorical_info"] = categorical_info
        
        logger.info("Dataset descrito com sucesso")
        return result
        
    except Exception as e:
        error_msg = f"Erro ao descrever dataset: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def analyze_data_types() -> Dict[str, List[str]]:
    """
    Analisa e categoriza os tipos de dados das colunas do dataset.
    
    Returns:
        Dicionário categorizando as colunas por tipo de dado
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        result = {
            "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime": df.select_dtypes(include=['datetime64']).columns.tolist(),
            "boolean": df.select_dtypes(include=['bool']).columns.tolist()
        }
        
        # Tentar identificar colunas que podem ser categóricas mas estão como object
        for col in result["categorical"]:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.05:  # Menos de 5% de valores únicos
                result.setdefault("likely_categorical", []).append(col)
        
        # Tentar identificar colunas que podem ser numéricas mas estão como object
        potential_numeric = []
        for col in result["categorical"]:
            sample_values = df[col].dropna().head(100)
            try:
                pd.to_numeric(sample_values)
                potential_numeric.append(col)
            except (ValueError, TypeError):
                pass
        
        if potential_numeric:
            result["potential_numeric"] = potential_numeric
        
        logger.info("Tipos de dados analisados com sucesso")
        return result
        
    except Exception as e:
        error_msg = f"Erro ao analisar tipos de dados: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def get_column_info(column_name: str) -> Dict[str, Any]:
    """
    Obtém informações detalhadas sobre uma coluna específica.
    
    Args:
        column_name: Nome da coluna a ser analisada
        
    Returns:
        Dicionário com informações detalhadas da coluna
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        if column_name not in df.columns:
            return {"error": f"Coluna '{column_name}' não encontrada no dataset"}
        
        col = df[column_name]
        
        result = {
            "column_name": column_name,
            "data_type": str(col.dtype),
            "total_values": len(col),
            "missing_values": col.isnull().sum(),
            "missing_percentage": col.isnull().sum() / len(col) * 100,
            "unique_values": col.nunique(),
            "unique_percentage": col.nunique() / len(col) * 100
        }
        
        # Se for numérica
        if pd.api.types.is_numeric_dtype(col):
            result.update({
                "min": col.min(),
                "max": col.max(),
                "mean": col.mean(),
                "median": col.median(),
                "std": col.std(),
                "variance": col.var(),
                "quartiles": {
                    "Q1": col.quantile(0.25),
                    "Q3": col.quantile(0.75),
                    "IQR": col.quantile(0.75) - col.quantile(0.25)
                },
                "skewness": col.skew(),
                "kurtosis": col.kurtosis()
            })
            
            # Identificar outliers usando IQR
            Q1 = col.quantile(0.25)
            Q3 = col.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col[(col < lower_bound) | (col > upper_bound)]
            
            result["outliers"] = {
                "count": len(outliers),
                "percentage": len(outliers) / len(col) * 100,
                "values": outliers.tolist()[:10]  # Primeiros 10 outliers
            }
        
        # Se for categórica
        elif pd.api.types.is_object_dtype(col) or pd.api.types.is_categorical_dtype(col):
            value_counts = col.value_counts()
            result.update({
                "most_frequent": value_counts.index[0] if not value_counts.empty else None,
                "frequency_of_most_frequent": value_counts.iloc[0] if not value_counts.empty else 0,
                "least_frequent": value_counts.index[-1] if not value_counts.empty else None,
                "frequency_of_least_frequent": value_counts.iloc[-1] if not value_counts.empty else 0,
                "top_categories": value_counts.head(10).to_dict()
            })
        
        # Se for datetime
        elif pd.api.types.is_datetime64_any_dtype(col):
            result.update({
                "min_date": col.min(),
                "max_date": col.max(),
                "date_range_days": (col.max() - col.min()).days
            })
        
        logger.info(f"Informações da coluna '{column_name}' obtidas com sucesso")
        return result
        
    except Exception as e:
        error_msg = f"Erro ao obter informações da coluna '{column_name}': {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def get_data_quality_report() -> Dict[str, Any]:
    """
    Gera um relatório de qualidade dos dados.
    
    Returns:
        Relatório completo de qualidade dos dados
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        report = {
            "dataset_overview": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            "missing_data": {},
            "duplicates": {
                "total_duplicates": df.duplicated().sum(),
                "duplicate_percentage": df.duplicated().sum() / len(df) * 100
            },
            "data_types": {
                "numeric": len(df.select_dtypes(include=[np.number]).columns),
                "categorical": len(df.select_dtypes(include=['object', 'category']).columns),
                "datetime": len(df.select_dtypes(include=['datetime64']).columns),
                "boolean": len(df.select_dtypes(include=['bool']).columns)
            },
            "quality_issues": []
        }
        
        # Análise de dados faltantes
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df) * 100)
        
        for col in df.columns:
            if missing_data[col] > 0:
                report["missing_data"][col] = {
                    "count": int(missing_data[col]),
                    "percentage": float(missing_percentage[col])
                }
                
                # Identificar problemas críticos
                if missing_percentage[col] > 50:
                    report["quality_issues"].append({
                        "type": "high_missing_data",
                        "column": col,
                        "description": f"Coluna '{col}' tem mais de 50% de dados faltantes",
                        "severity": "high"
                    })
                elif missing_percentage[col] > 20:
                    report["quality_issues"].append({
                        "type": "moderate_missing_data",
                        "column": col,
                        "description": f"Coluna '{col}' tem mais de 20% de dados faltantes",
                        "severity": "medium"
                    })
        
        # Identificar colunas com pouca variabilidade
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].std() == 0:
                report["quality_issues"].append({
                    "type": "no_variance",
                    "column": col,
                    "description": f"Coluna '{col}' tem variância zero (valores constantes)",
                    "severity": "medium"
                })
        
        # Identificar colunas categóricas com muitas categorias
        for col in df.select_dtypes(include=['object', 'category']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95:
                report["quality_issues"].append({
                    "type": "high_cardinality",
                    "column": col,
                    "description": f"Coluna '{col}' tem cardinalidade muito alta (95%+ valores únicos)",
                    "severity": "low"
                })
        
        # Calcular score de qualidade
        quality_score = 100
        for issue in report["quality_issues"]:
            if issue["severity"] == "high":
                quality_score -= 20
            elif issue["severity"] == "medium":
                quality_score -= 10
            elif issue["severity"] == "low":
                quality_score -= 5
        
        report["quality_score"] = max(0, quality_score)
        
        logger.info("Relatório de qualidade de dados gerado com sucesso")
        return report
        
    except Exception as e:
        error_msg = f"Erro ao gerar relatório de qualidade: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def compare_columns(column1: str, column2: str) -> Dict[str, Any]:
    """
    Compara duas colunas do dataset.
    
    Args:
        column1: Nome da primeira coluna
        column2: Nome da segunda coluna
        
    Returns:
        Dicionário com comparação entre as colunas
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        if column1 not in df.columns:
            return {"error": f"Coluna '{column1}' não encontrada no dataset"}
        
        if column2 not in df.columns:
            return {"error": f"Coluna '{column2}' não encontrada no dataset"}
        
        col1 = df[column1]
        col2 = df[column2]
        
        comparison = {
            "column1": {
                "name": column1,
                "type": str(col1.dtype),
                "unique_values": col1.nunique(),
                "missing_values": col1.isnull().sum()
            },
            "column2": {
                "name": column2,
                "type": str(col2.dtype),
                "unique_values": col2.nunique(),
                "missing_values": col2.isnull().sum()
            }
        }
        
        # Se ambas forem numéricas, calcular correlação
        if pd.api.types.is_numeric_dtype(col1) and pd.api.types.is_numeric_dtype(col2):
            correlation = col1.corr(col2)
            comparison["correlation"] = {
                "pearson": correlation,
                "interpretation": "forte" if abs(correlation) > 0.7 else 
                               "moderada" if abs(correlation) > 0.3 else "fraca"
            }
            
            comparison["column1"].update({
                "mean": col1.mean(),
                "std": col1.std(),
                "min": col1.min(),
                "max": col1.max()
            })
            
            comparison["column2"].update({
                "mean": col2.mean(),
                "std": col2.std(),
                "min": col2.min(),
                "max": col2.max()
            })
        
        # Se ambas forem categóricas, comparar categorias
        elif (pd.api.types.is_object_dtype(col1) and pd.api.types.is_object_dtype(col2)):
            common_values = set(col1.dropna().unique()) & set(col2.dropna().unique())
            comparison["common_categories"] = {
                "count": len(common_values),
                "values": list(common_values)[:10]  # Primeiras 10
            }
        
        logger.info(f"Comparação entre '{column1}' e '{column2}' realizada com sucesso")
        return comparison
        
    except Exception as e:
        error_msg = f"Erro ao comparar colunas '{column1}' e '{column2}': {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def get_data_analysis_tools() -> List:
    """Retorna lista de todas as ferramentas de análise de dados"""
    return [
        describe_dataset,
        analyze_data_types,
        get_column_info,
        get_data_quality_report,
        compare_columns
    ]