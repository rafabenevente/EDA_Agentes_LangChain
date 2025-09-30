"""
Ferramentas para detecção e análise de outliers/anomalias
"""

import logging
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
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
def detect_outliers_iqr(factor: float = 1.5, columns: str = None,) -> Dict[str, Any]:
    """
    Detecta outliers usando o método IQR (Interquartile Range).
    
    Args:
        columns: Lista de colunas separadas por vírgula (opcional)
        factor: Fator multiplicativo para definir outliers (padrão: 1.5)
        
    Returns:
        Análise de outliers usando método IQR
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
        total_outliers = 0
        outlier_indices = set()
        
        for col in column_list:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                results[col] = {"error": "Nenhum dado válido"}
                continue
            
            # Calcular quartis
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Definir limites
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Identificar outliers
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers = df[col][outliers_mask]
            
            # Análise dos outliers
            outlier_stats = {
                "count": len(outliers),
                "percentage": float(len(outliers) / len(col_data) * 100),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "Q1": float(Q1),
                "Q3": float(Q3),
                "IQR": float(IQR),
                "outlier_values": {
                    "below_lower": outliers[outliers < lower_bound].tolist()[:5],
                    "above_upper": outliers[outliers > upper_bound].tolist()[:5],
                    "extreme_low": float(outliers.min()) if len(outliers) > 0 else None,
                    "extreme_high": float(outliers.max()) if len(outliers) > 0 else None
                },
                "outlier_indices": outliers.index.tolist()
            }
            
            results[col] = outlier_stats
            total_outliers += len(outliers)
            outlier_indices.update(outliers.index.tolist())
        
        # Análise geral
        rows_with_outliers = len(outlier_indices)
        
        # Linhas com múltiplos outliers
        outlier_counts_per_row = {}
        for col in column_list:
            if "outlier_indices" in results[col]:
                for idx in results[col]["outlier_indices"]:
                    outlier_counts_per_row[idx] = outlier_counts_per_row.get(idx, 0) + 1
        
        multi_outlier_rows = {k: v for k, v in outlier_counts_per_row.items() if v > 1}
        
        summary = {
            "method": "IQR",
            "factor_used": factor,
            "total_variables": len(column_list),
            "total_outliers": total_outliers,
            "rows_with_outliers": rows_with_outliers,
            "percentage_rows_with_outliers": float(rows_with_outliers / len(df) * 100),
            "rows_with_multiple_outliers": len(multi_outlier_rows),
            "most_problematic_rows": sorted(multi_outlier_rows.items(), 
                                          key=lambda x: x[1], reverse=True)[:5]
        }
        
        result = {
            "variables_analyzed": column_list,
            "outlier_analysis": results,
            "summary": summary,
            "success": True
        }
        
        logger.info(f"Detecção de outliers IQR realizada para {len(column_list)} variáveis")
        return result
        
    except Exception as e:
        error_msg = f"Erro na detecção de outliers IQR: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def detect_outliers_zscore(columns: str = None, threshold: float = 3.0) -> Dict[str, Any]:
    """
    Detecta outliers usando o Z-Score.
    
    Args:
        columns: Lista de colunas separadas por vírgula (opcional)
        threshold: Limiar do Z-Score para considerar outlier (padrão: 3.0)
        
    Returns:
        Análise de outliers usando Z-Score
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
        total_outliers = 0
        outlier_indices = set()
        
        for col in column_list:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                results[col] = {"error": "Nenhum dado válido"}
                continue
            
            # Calcular Z-Score
            mean = col_data.mean()
            std = col_data.std()
            
            if std == 0:
                results[col] = {"error": "Desvio padrão zero - não é possível calcular Z-Score"}
                continue
            
            z_scores = np.abs((df[col] - mean) / std)
            outliers_mask = z_scores > threshold
            outliers = df[col][outliers_mask]
            outlier_z_scores = z_scores[outliers_mask]
            
            # Análise dos outliers
            outlier_stats = {
                "count": len(outliers),
                "percentage": float(len(outliers) / len(col_data) * 100),
                "threshold_used": threshold,
                "mean": float(mean),
                "std": float(std),
                "outlier_details": [
                    {
                        "index": int(idx),
                        "value": float(val),
                        "z_score": float(z_scores[idx])
                    }
                    for idx, val in outliers.head(10).items()
                ],
                "max_z_score": float(outlier_z_scores.max()) if len(outlier_z_scores) > 0 else None,
                "min_outlier_value": float(outliers.min()) if len(outliers) > 0 else None,
                "max_outlier_value": float(outliers.max()) if len(outliers) > 0 else None,
                "outlier_indices": outliers.index.tolist()
            }
            
            results[col] = outlier_stats
            total_outliers += len(outliers)
            outlier_indices.update(outliers.index.tolist())
        
        # Análise geral
        rows_with_outliers = len(outlier_indices)
        
        # Linhas com múltiplos outliers
        outlier_counts_per_row = {}
        for col in column_list:
            if "outlier_indices" in results[col]:
                for idx in results[col]["outlier_indices"]:
                    outlier_counts_per_row[idx] = outlier_counts_per_row.get(idx, 0) + 1
        
        multi_outlier_rows = {k: v for k, v in outlier_counts_per_row.items() if v > 1}
        
        summary = {
            "method": "Z-Score",
            "threshold_used": threshold,
            "total_variables": len(column_list),
            "total_outliers": total_outliers,
            "rows_with_outliers": rows_with_outliers,
            "percentage_rows_with_outliers": float(rows_with_outliers / len(df) * 100),
            "rows_with_multiple_outliers": len(multi_outlier_rows),
            "most_problematic_rows": sorted(multi_outlier_rows.items(), 
                                          key=lambda x: x[1], reverse=True)[:5]
        }
        
        result = {
            "variables_analyzed": column_list,
            "outlier_analysis": results,
            "summary": summary,
            "success": True
        }
        
        logger.info(f"Detecção de outliers Z-Score realizada para {len(column_list)} variáveis")
        return result
        
    except Exception as e:
        error_msg = f"Erro na detecção de outliers Z-Score: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def detect_outliers_isolation_forest(columns: str = None, contamination: float = 0.1, 
                                    random_state: int = 42) -> Dict[str, Any]:
    """
    Detecta outliers usando Isolation Forest.
    
    Args:
        columns: Lista de colunas separadas por vírgula (opcional)
        contamination: Proporção esperada de outliers (padrão: 0.1)
        random_state: Semente aleatória para reproducibilidade
        
    Returns:
        Análise de outliers usando Isolation Forest
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
        
        # Preparar dados
        data = df[column_list].dropna()
        
        if len(data) == 0:
            return {"error": "Nenhum dado válido após remoção de valores nulos"}
        
        if len(data) < 10:
            return {"error": "Dados insuficientes para Isolation Forest (mínimo: 10 amostras)"}
        
        # Padronizar dados
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Aplicar Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        
        outlier_labels = iso_forest.fit_predict(data_scaled)
        outlier_scores = iso_forest.decision_function(data_scaled)
        
        # Identificar outliers (label -1)
        outlier_mask = outlier_labels == -1
        outlier_indices = data.index[outlier_mask].tolist()
        outlier_data = data[outlier_mask]
        
        # Análise por variável
        variable_analysis = {}
        for col in column_list:
            outlier_values = outlier_data[col]
            
            variable_analysis[col] = {
                "outliers_in_variable": len(outlier_values),
                "outlier_values": outlier_values.tolist()[:10],
                "min_outlier": float(outlier_values.min()) if len(outlier_values) > 0 else None,
                "max_outlier": float(outlier_values.max()) if len(outlier_values) > 0 else None,
                "mean_outlier": float(outlier_values.mean()) if len(outlier_values) > 0 else None
            }
        
        # Análise dos scores
        outlier_details = []
        for i, (idx, score) in enumerate(zip(outlier_indices, outlier_scores[outlier_mask])):
            if i < 20:  # Limitar a 20 para performance
                outlier_details.append({
                    "index": int(idx),
                    "anomaly_score": float(score),
                    "values": {col: float(data.loc[idx, col]) for col in column_list}
                })
        
        # Ordenar por score (mais anômalos primeiro)
        outlier_details.sort(key=lambda x: x["anomaly_score"])
        
        result = {
            "method": "Isolation Forest",
            "parameters": {
                "contamination": contamination,
                "random_state": random_state,
                "variables_used": column_list
            },
            "results": {
                "total_samples": len(data),
                "outliers_detected": len(outlier_indices),
                "outlier_percentage": float(len(outlier_indices) / len(data) * 100),
                "outlier_indices": outlier_indices,
                "most_anomalous": outlier_details[:10],
                "variable_analysis": variable_analysis
            },
            "model_info": {
                "n_estimators": 100,
                "contamination_used": contamination,
                "feature_count": len(column_list)
            },
            "success": True
        }
        
        logger.info(f"Detecção com Isolation Forest: {len(outlier_indices)} outliers encontrados")
        return result
        
    except Exception as e:
        error_msg = f"Erro na detecção com Isolation Forest: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def detect_outliers_lof(columns: str = None, n_neighbors: int = 20, 
                       contamination: float = 0.1) -> Dict[str, Any]:
    """
    Detecta outliers usando Local Outlier Factor (LOF).
    
    Args:
        columns: Lista de colunas separadas por vírgula (opcional)
        n_neighbors: Número de vizinhos para o algoritmo LOF
        contamination: Proporção esperada de outliers
        
    Returns:
        Análise de outliers usando LOF
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
        
        # Preparar dados
        data = df[column_list].dropna()
        
        if len(data) == 0:
            return {"error": "Nenhum dado válido após remoção de valores nulos"}
        
        if len(data) <= n_neighbors:
            return {"error": f"Dados insuficientes para LOF (necessário > {n_neighbors} amostras)"}
        
        # Padronizar dados
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Aplicar LOF
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination
        )
        
        outlier_labels = lof.fit_predict(data_scaled)
        lof_scores = lof.negative_outlier_factor_
        
        # Identificar outliers (label -1)
        outlier_mask = outlier_labels == -1
        outlier_indices = data.index[outlier_mask].tolist()
        outlier_data = data[outlier_mask]
        outlier_lof_scores = lof_scores[outlier_mask]
        
        # Análise detalhada dos outliers
        outlier_details = []
        for i, (idx, score) in enumerate(zip(outlier_indices, outlier_lof_scores)):
            if i < 20:  # Limitar a 20
                outlier_details.append({
                    "index": int(idx),
                    "lof_score": float(-score),  # Converter para positivo (mais fácil interpretação)
                    "values": {col: float(data.loc[idx, col]) for col in column_list}
                })
        
        # Ordenar por LOF score (mais anômalos primeiro)
        outlier_details.sort(key=lambda x: x["lof_score"], reverse=True)
        
        # Análise por variável
        variable_analysis = {}
        for col in column_list:
            outlier_values = outlier_data[col]
            normal_values = data[~outlier_mask][col]
            
            variable_analysis[col] = {
                "outliers_count": len(outlier_values),
                "outlier_mean": float(outlier_values.mean()) if len(outlier_values) > 0 else None,
                "normal_mean": float(normal_values.mean()) if len(normal_values) > 0 else None,
                "difference_from_normal": float(outlier_values.mean() - normal_values.mean()) if len(outlier_values) > 0 and len(normal_values) > 0 else None,
                "outlier_std": float(outlier_values.std()) if len(outlier_values) > 0 else None,
                "normal_std": float(normal_values.std()) if len(normal_values) > 0 else None
            }
        
        result = {
            "method": "Local Outlier Factor (LOF)",
            "parameters": {
                "n_neighbors": n_neighbors,
                "contamination": contamination,
                "variables_used": column_list
            },
            "results": {
                "total_samples": len(data),
                "outliers_detected": len(outlier_indices),
                "outlier_percentage": float(len(outlier_indices) / len(data) * 100),
                "outlier_indices": outlier_indices,
                "most_anomalous": outlier_details[:10],
                "variable_analysis": variable_analysis,
                "lof_score_stats": {
                    "min_lof_score": float(-lof_scores.max()),  # Mais anômalo
                    "max_lof_score": float(-lof_scores.min()),  # Menos anômalo
                    "mean_lof_score": float(-lof_scores.mean())
                }
            },
            "success": True
        }
        
        logger.info(f"Detecção com LOF: {len(outlier_indices)} outliers encontrados")
        return result
        
    except Exception as e:
        error_msg = f"Erro na detecção com LOF: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def compare_outlier_methods(columns: str|None = None) -> Dict[str, Any]:
    """
    Compara diferentes métodos de detecção de outliers.
    
    Args:
        columns: Lista de colunas separadas por vírgula (opcional)
        
    Returns:
        Comparação entre métodos de detecção de outliers
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        # Aplicar cada método
        methods = {
            "IQR": detect_outliers_iqr,
            "Z-Score": detect_outliers_zscore,
            "Isolation Forest": detect_outliers_isolation_forest
        }
        
        results = {}
        all_outliers = {}
        
        for method_name, method_func in methods.items():
            try:
                if method_name == "IQR":
                    result = method_func.func(columns, 1.5)
                elif method_name == "Z-Score":
                    result = method_func.func(columns, 3.0)
                else:  # Isolation Forest
                    result = method_func.func(columns, 0.1, 42)
                
                if result.get("success"):
                    results[method_name] = result
                    
                    # Coletar índices de outliers para comparação
                    if method_name in ["IQR", "Z-Score"]:
                        outlier_indices = set()
                        for col_result in result["outlier_analysis"].values():
                            if "outlier_indices" in col_result:
                                outlier_indices.update(col_result["outlier_indices"])
                    else:  # Isolation Forest
                        outlier_indices = set(result["results"]["outlier_indices"])
                    
                    all_outliers[method_name] = outlier_indices
                else:
                    results[method_name] = {"error": result.get("error", "Erro desconhecido")}
                    
            except Exception as e:
                results[method_name] = {"error": str(e)}
        
        # Análise comparativa
        comparison = {}
        
        if len(all_outliers) >= 2:
            # Outliers comuns entre métodos
            methods_list = list(all_outliers.keys())
            
            for i in range(len(methods_list)):
                for j in range(i+1, len(methods_list)):
                    method1, method2 = methods_list[i], methods_list[j]
                    
                    common = all_outliers[method1] & all_outliers[method2]
                    union = all_outliers[method1] | all_outliers[method2]
                    
                    jaccard = len(common) / len(union) if len(union) > 0 else 0
                    
                    comparison[f"{method1}_vs_{method2}"] = {
                        "common_outliers": len(common),
                        "jaccard_similarity": float(jaccard),
                        "method1_only": len(all_outliers[method1] - all_outliers[method2]),
                        "method2_only": len(all_outliers[method2] - all_outliers[method1]),
                        "common_indices": list(common)[:10]  # Primeiros 10
                    }
            
            # Consenso entre métodos
            if len(all_outliers) >= 3:
                consensus_outliers = set.intersection(*all_outliers.values())
                any_method_outliers = set.union(*all_outliers.values())
                
                comparison["consensus"] = {
                    "outliers_all_methods": len(consensus_outliers),
                    "outliers_any_method": len(any_method_outliers),
                    "consensus_indices": list(consensus_outliers)[:10],
                    "consensus_percentage": float(len(consensus_outliers) / len(any_method_outliers) * 100) if len(any_method_outliers) > 0 else 0
                }
        
        # Resumo por método
        method_summary = {}
        for method_name, outlier_set in all_outliers.items():
            method_summary[method_name] = {
                "outliers_detected": len(outlier_set),
                "percentage": float(len(outlier_set) / len(df) * 100)
            }
        
        result = {
            "comparison_results": results,
            "method_summary": method_summary,
            "comparative_analysis": comparison,
            "recommendations": {
                "most_conservative": min(method_summary.items(), key=lambda x: x[1]["outliers_detected"])[0] if method_summary else None,
                "most_aggressive": max(method_summary.items(), key=lambda x: x[1]["outliers_detected"])[0] if method_summary else None,
                "suggested_approach": "Use consenso entre métodos para maior confiabilidade" if len(all_outliers) >= 2 else "Execute pelo menos dois métodos diferentes"
            },
            "success": True
        }
        
        logger.info(f"Comparação de métodos realizada: {len(results)} métodos testados")
        return result
        
    except Exception as e:
        error_msg = f"Erro na comparação de métodos: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def get_outlier_detection_tools() -> List:
    """Retorna lista de todas as ferramentas de detecção de outliers"""
    return [
        detect_outliers_iqr,
        detect_outliers_zscore,
        detect_outliers_isolation_forest,
        detect_outliers_lof,
        compare_outlier_methods
    ]