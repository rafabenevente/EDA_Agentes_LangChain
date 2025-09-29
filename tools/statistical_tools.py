"""
Ferramentas para análises estatísticas avançadas
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from langchain.tools import tool
def get_current_dataframe():
    """Retorna o dataframe atual (importado do agente principal)"""
    try:
        from agents.eda_agent import get_current_dataframe as get_df
        return get_df()
    except ImportError:
        return None

logger = logging.getLogger(__name__)


@tool
def calculate_correlation_analysis(columns: str = None, method: str = "pearson") -> Dict[str, Any]:
    """
    Realiza análise detalhada de correlações entre variáveis.
    
    Args:
        columns: Lista de colunas separadas por vírgula (opcional)
        method: Método de correlação ('pearson', 'spearman', 'kendall')
        
    Returns:
        Análise detalhada de correlações
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        # Selecionar colunas numéricas
        if columns:
            column_list = [col.strip() for col in columns.split(',')]
            missing_cols = [col for col in column_list if col not in df.columns]
            if missing_cols:
                return {"error": f"Colunas não encontradas: {missing_cols}"}
            
            non_numeric = [col for col in column_list if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                return {"error": f"Colunas não numéricas: {non_numeric}"}
        else:
            column_list = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(column_list) < 2:
            return {"error": "Necessário pelo menos 2 colunas numéricas"}
        
        # Calcular matriz de correlação
        corr_matrix = df[column_list].corr(method=method)
        
        # Analisar correlações
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                # Classificar força da correlação
                abs_corr = abs(corr_value)
                if abs_corr >= 0.8:
                    strength = "muito forte"
                elif abs_corr >= 0.6:
                    strength = "forte"
                elif abs_corr >= 0.4:
                    strength = "moderada"
                elif abs_corr >= 0.2:
                    strength = "fraca"
                else:
                    strength = "muito fraca"
                
                # Determinar direção
                direction = "positiva" if corr_value > 0 else "negativa"
                
                correlations.append({
                    "variable1": var1,
                    "variable2": var2,
                    "correlation": float(corr_value),
                    "strength": strength,
                    "direction": direction,
                    "abs_correlation": float(abs_corr)
                })
        
        # Ordenar por correlação absoluta
        correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)
        
        # Estatísticas resumo
        corr_values = [c["correlation"] for c in correlations]
        
        result = {
            "method": method,
            "total_variables": len(column_list),
            "variables_analyzed": column_list,
            "total_correlations": len(correlations),
            "strongest_correlations": correlations[:10],  # Top 10
            "weakest_correlations": correlations[-5:],   # Bottom 5
            "correlation_statistics": {
                "mean_correlation": float(np.mean(np.abs(corr_values))),
                "max_correlation": float(max(corr_values, key=abs)),
                "min_correlation": float(min(corr_values, key=abs)),
                "strong_correlations_count": sum(1 for c in correlations if c["abs_correlation"] >= 0.6),
                "weak_correlations_count": sum(1 for c in correlations if c["abs_correlation"] < 0.2)
            },
            "success": True
        }
        
        logger.info(f"Análise de correlação realizada para {len(column_list)} variáveis")
        return result
        
    except Exception as e:
        error_msg = f"Erro na análise de correlação: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def perform_normality_tests(columns: str = None) -> Dict[str, Any]:
    """
    Realiza testes de normalidade nas variáveis numéricas.
    
    Args:
        columns: Lista de colunas separadas por vírgula (opcional)
        
    Returns:
        Resultados dos testes de normalidade
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        # Selecionar colunas numéricas
        if columns:
            column_list = [col.strip() for col in columns.split(',')]
            missing_cols = [col for col in column_list if col not in df.columns]
            if missing_cols:
                return {"error": f"Colunas não encontradas: {missing_cols}"}
            
            non_numeric = [col for col in column_list if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                return {"error": f"Colunas não numéricas: {non_numeric}"}
        else:
            column_list = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(column_list) == 0:
            return {"error": "Nenhuma coluna numérica encontrada"}
        
        results = {}
        
        for col in column_list:
            col_data = df[col].dropna()
            
            if len(col_data) < 8:  # Mínimo para testes
                results[col] = {"error": "Dados insuficientes para teste"}
                continue
            
            # Teste de Shapiro-Wilk (para amostras pequenas < 5000)
            if len(col_data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(col_data)
                shapiro_normal = shapiro_p > 0.05
            else:
                shapiro_stat, shapiro_p, shapiro_normal = None, None, None
            
            # Teste de D'Agostino (para amostras maiores)
            if len(col_data) >= 20:
                dagostino_stat, dagostino_p = stats.normaltest(col_data)
                dagostino_normal = dagostino_p > 0.05
            else:
                dagostino_stat, dagostino_p, dagostino_normal = None, None, None
            
            # Teste de Kolmogorov-Smirnov
            ks_stat, ks_p = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
            ks_normal = ks_p > 0.05
            
            # Calcular assimetria e curtose
            skewness = float(stats.skew(col_data))
            kurtosis = float(stats.kurtosis(col_data))
            
            # Interpretação da assimetria
            if abs(skewness) < 0.5:
                skew_interpretation = "simétrica"
            elif abs(skewness) < 1:
                skew_interpretation = "moderadamente assimétrica"
            else:
                skew_interpretation = "altamente assimétrica"
            
            # Interpretação da curtose
            if abs(kurtosis) < 0.5:
                kurt_interpretation = "mesocúrtica (normal)"
            elif kurtosis > 0.5:
                kurt_interpretation = "leptocúrtica (mais pontiaguda)"
            else:
                kurt_interpretation = "platicúrtica (mais achatada)"
            
            results[col] = {
                "sample_size": len(col_data),
                "shapiro_test": {
                    "statistic": float(shapiro_stat) if shapiro_stat else None,
                    "p_value": float(shapiro_p) if shapiro_p else None,
                    "is_normal": shapiro_normal
                } if shapiro_stat else None,
                "dagostino_test": {
                    "statistic": float(dagostino_stat) if dagostino_stat else None,
                    "p_value": float(dagostino_p) if dagostino_p else None,
                    "is_normal": dagostino_normal
                } if dagostino_stat else None,
                "ks_test": {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_p),
                    "is_normal": ks_normal
                },
                "descriptive_stats": {
                    "skewness": skewness,
                    "skewness_interpretation": skew_interpretation,
                    "kurtosis": kurtosis,
                    "kurtosis_interpretation": kurt_interpretation
                },
                "conclusion": "normal" if (shapiro_normal if shapiro_normal is not None else 
                                         dagostino_normal if dagostino_normal is not None else ks_normal) 
                             else "não normal"
            }
        
        # Resumo geral
        total_vars = len(results)
        normal_vars = sum(1 for r in results.values() if isinstance(r, dict) and r.get("conclusion") == "normal")
        
        summary = {
            "total_variables": total_vars,
            "normal_variables": normal_vars,
            "non_normal_variables": total_vars - normal_vars,
            "percentage_normal": float(normal_vars / total_vars * 100) if total_vars > 0 else 0
        }
        
        result = {
            "variables_tested": column_list,
            "results": results,
            "summary": summary,
            "success": True
        }
        
        logger.info(f"Testes de normalidade realizados para {len(column_list)} variáveis")
        return result
        
    except Exception as e:
        error_msg = f"Erro nos testes de normalidade: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def calculate_descriptive_statistics(columns: str = None) -> Dict[str, Any]:
    """
    Calcula estatísticas descritivas detalhadas.
    
    Args:
        columns: Lista de colunas separadas por vírgula (opcional)
        
    Returns:
        Estatísticas descritivas detalhadas
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        # Selecionar colunas numéricas
        if columns:
            column_list = [col.strip() for col in columns.split(',')]
            missing_cols = [col for col in column_list if col not in df.columns]
            if missing_cols:
                return {"error": f"Colunas não encontradas: {missing_cols}"}
            
            non_numeric = [col for col in column_list if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                return {"error": f"Colunas não numéricas: {non_numeric}"}
        else:
            column_list = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(column_list) == 0:
            return {"error": "Nenhuma coluna numérica encontrada"}
        
        results = {}
        
        for col in column_list:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                results[col] = {"error": "Nenhum dado válido"}
                continue
            
            # Estatísticas básicas
            basic_stats = {
                "count": len(col_data),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "mode": float(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                "std": float(col_data.std()),
                "variance": float(col_data.var()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "range": float(col_data.max() - col_data.min())
            }
            
            # Quartis e percentis
            percentiles = {
                "Q1": float(col_data.quantile(0.25)),
                "Q2": float(col_data.quantile(0.50)),  # Mediana
                "Q3": float(col_data.quantile(0.75)),
                "IQR": float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                "P5": float(col_data.quantile(0.05)),
                "P95": float(col_data.quantile(0.95)),
                "P99": float(col_data.quantile(0.99))
            }
            
            # Medidas de forma
            shape_measures = {
                "skewness": float(stats.skew(col_data)),
                "kurtosis": float(stats.kurtosis(col_data))
            }
            
            # Medidas de posição relativa
            position_measures = {
                "coefficient_of_variation": float(basic_stats["std"] / basic_stats["mean"] * 100) if basic_stats["mean"] != 0 else 0,
                "mean_absolute_deviation": float(np.mean(np.abs(col_data - col_data.mean()))),
                "median_absolute_deviation": float(np.median(np.abs(col_data - col_data.median())))
            }
            
            # Análise de outliers (usando IQR)
            Q1 = percentiles["Q1"]
            Q3 = percentiles["Q3"]
            IQR = percentiles["IQR"]
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            outlier_analysis = {
                "outlier_count": len(outliers),
                "outlier_percentage": float(len(outliers) / len(col_data) * 100),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "outlier_values": outliers.tolist()[:10]  # Primeiros 10
            }
            
            results[col] = {
                "basic_statistics": basic_stats,
                "percentiles": percentiles,
                "shape_measures": shape_measures,
                "position_measures": position_measures,
                "outlier_analysis": outlier_analysis
            }
        
        # Análise comparativa entre variáveis
        if len(column_list) > 1:
            comparison = {
                "highest_mean": max(column_list, key=lambda x: results[x]["basic_statistics"]["mean"]),
                "highest_std": max(column_list, key=lambda x: results[x]["basic_statistics"]["std"]),
                "most_skewed": max(column_list, key=lambda x: abs(results[x]["shape_measures"]["skewness"])),
                "most_outliers": max(column_list, key=lambda x: results[x]["outlier_analysis"]["outlier_percentage"])
            }
        else:
            comparison = None
        
        result = {
            "variables_analyzed": column_list,
            "detailed_statistics": results,
            "comparative_analysis": comparison,
            "success": True
        }
        
        logger.info(f"Estatísticas descritivas calculadas para {len(column_list)} variáveis")
        return result
        
    except Exception as e:
        error_msg = f"Erro no cálculo de estatísticas descritivas: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def perform_pca_analysis(n_components: int = None, columns: str = None) -> Dict[str, Any]:
    """
    Realiza Análise de Componentes Principais (PCA).
    
    Args:
        n_components: Número de componentes principais (opcional)
        columns: Lista de colunas separadas por vírgula (opcional)
        
    Returns:
        Resultados da análise PCA
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        # Selecionar colunas numéricas
        if columns:
            column_list = [col.strip() for col in columns.split(',')]
            missing_cols = [col for col in column_list if col not in df.columns]
            if missing_cols:
                return {"error": f"Colunas não encontradas: {missing_cols}"}
            
            non_numeric = [col for col in column_list if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                return {"error": f"Colunas não numéricas: {non_numeric}"}
        else:
            column_list = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(column_list) < 2:
            return {"error": "Necessário pelo menos 2 colunas numéricas para PCA"}
        
        # Preparar dados
        data = df[column_list].dropna()
        
        if len(data) == 0:
            return {"error": "Nenhum dado válido após remoção de valores nulos"}
        
        # Padronizar dados
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Determinar número de componentes
        if n_components is None:
            n_components = min(len(column_list), len(data))
        else:
            n_components = min(n_components, len(column_list), len(data))
        
        # Aplicar PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(data_scaled)
        
        # Analisar resultados
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Componentes principais
        component_info = []
        for i in range(n_components):
            component_info.append({
                "component": f"PC{i+1}",
                "explained_variance": float(explained_variance[i]),
                "cumulative_variance": float(cumulative_variance[i]),
                "loadings": {col: float(pca.components_[i][j]) 
                           for j, col in enumerate(column_list)}
            })
        
        # Identificar variáveis mais importantes para cada componente
        important_vars = []
        for i in range(min(3, n_components)):  # Top 3 componentes
            loadings = np.abs(pca.components_[i])
            sorted_indices = np.argsort(loadings)[::-1]
            important_vars.append({
                "component": f"PC{i+1}",
                "most_important_variables": [
                    {
                        "variable": column_list[idx],
                        "loading": float(pca.components_[i][idx]),
                        "abs_loading": float(loadings[idx])
                    }
                    for idx in sorted_indices[:3]
                ]
            })
        
        # Sugestão de número ideal de componentes (90% da variância)
        ideal_components = int(np.where(cumulative_variance >= 0.90)[0][0] + 1) if any(cumulative_variance >= 0.90) else n_components
        
        result = {
            "original_variables": column_list,
            "n_components": n_components,
            "sample_size": len(data),
            "total_variance_explained": float(explained_variance.sum()),
            "component_details": component_info,
            "important_variables_by_component": important_vars,
            "suggested_components_90_variance": ideal_components,
            "variance_summary": {
                "first_component": float(explained_variance[0]),
                "first_two_components": float(explained_variance[:2].sum()) if n_components >= 2 else None,
                "first_three_components": float(explained_variance[:3].sum()) if n_components >= 3 else None
            },
            "success": True
        }
        
        logger.info(f"Análise PCA realizada com {n_components} componentes")
        return result
        
    except Exception as e:
        error_msg = f"Erro na análise PCA: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def get_statistical_tools() -> List:
    """Retorna lista de todas as ferramentas estatísticas"""
    return [
        calculate_correlation_analysis,
        perform_normality_tests,
        calculate_descriptive_statistics,
        perform_pca_analysis
    ]