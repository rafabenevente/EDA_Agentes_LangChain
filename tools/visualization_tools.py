"""
Ferramentas para criação de visualizações
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from langchain.tools import tool
from utils.visualization_helpers import VisualizationHelpers
def get_current_dataframe():
    """Retorna o dataframe atual (importado do agente principal)"""
    try:
        from agents.eda_agent import get_current_dataframe as get_df
        return get_df()
    except ImportError:
        return None

logger = logging.getLogger(__name__)
viz_helper = VisualizationHelpers()

# Lista para armazenar visualizações criadas
_created_visualizations: List[Dict[str, Any]] = []


def get_created_visualizations() -> List[Dict[str, Any]]:
    """Retorna lista de visualizações criadas"""
    return _created_visualizations


def clear_visualizations() -> None:
    """Limpa lista de visualizações"""
    global _created_visualizations
    _created_visualizations = []


def add_visualization(viz_type: str, figure, title: str, description: str = "") -> None:
    """Adiciona uma visualização à lista"""
    _created_visualizations.append({
        "type": viz_type,
        "figure": figure,
        "title": title,
        "description": description
    })


@tool
def create_histogram(column_name: str, bins: int = 30, title: str = None) -> Dict[str, Any]:
    """
    Cria um histograma para uma coluna numérica.
    
    Args:
        column_name: Nome da coluna numérica
        bins: Número de bins para o histograma
        title: Título personalizado (opcional)
        
    Returns:
        Informações sobre o histograma criado
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        if column_name not in df.columns:
            return {"error": f"Coluna '{column_name}' não encontrada"}
        
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            return {"error": f"Coluna '{column_name}' não é numérica"}
        
        # Criar histograma
        fig = viz_helper.create_histogram(df, column_name, bins, title)
        
        # Adicionar à lista de visualizações
        chart_title = title or f"Histograma de {column_name}"
        add_visualization("histogram", fig, chart_title, 
                         f"Distribuição da variável {column_name}")
        
        # Calcular estatísticas para o resultado
        col_data = df[column_name].dropna()
        result = {
            "chart_type": "histogram",
            "column": column_name,
            "title": chart_title,
            "statistics": {
                "count": len(col_data),
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "bins": bins
            },
            "success": True
        }
        
        logger.info(f"Histograma criado para coluna '{column_name}'")
        return result
        
    except Exception as e:
        error_msg = f"Erro ao criar histograma para '{column_name}': {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def create_box_plot(column_name: str, by_column: str = None, title: str = None) -> Dict[str, Any]:
    """
    Cria um box plot para uma coluna numérica.
    
    Args:
        column_name: Nome da coluna numérica
        by_column: Coluna para agrupar (opcional)
        title: Título personalizado (opcional)
        
    Returns:
        Informações sobre o box plot criado
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        if column_name not in df.columns:
            return {"error": f"Coluna '{column_name}' não encontrada"}
        
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            return {"error": f"Coluna '{column_name}' não é numérica"}
        
        if by_column and by_column not in df.columns:
            return {"error": f"Coluna de agrupamento '{by_column}' não encontrada"}
        
        # Criar box plot
        fig = viz_helper.create_box_plot(df, column_name, by_column, title)
        
        # Adicionar à lista de visualizações
        chart_title = title or f"Box Plot de {column_name}"
        if by_column:
            chart_title += f" por {by_column}"
        
        add_visualization("box_plot", fig, chart_title,
                         f"Box plot mostrando distribuição e outliers de {column_name}")
        
        # Calcular estatísticas
        col_data = df[column_name].dropna()
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        
        result = {
            "chart_type": "box_plot",
            "column": column_name,
            "grouped_by": by_column,
            "title": chart_title,
            "statistics": {
                "count": len(col_data),
                "median": float(col_data.median()),
                "Q1": float(Q1),
                "Q3": float(Q3),
                "IQR": float(IQR),
                "outliers_count": len(col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)])
            },
            "success": True
        }
        
        logger.info(f"Box plot criado para coluna '{column_name}'")
        return result
        
    except Exception as e:
        error_msg = f"Erro ao criar box plot para '{column_name}': {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def create_scatter_plot(x_column: str, y_column: str, color_column: str = None, 
                       size_column: str = None, title: str = None) -> Dict[str, Any]:
    """
    Cria um gráfico de dispersão entre duas variáveis.
    
    Args:
        x_column: Nome da coluna para eixo X
        y_column: Nome da coluna para eixo Y
        color_column: Coluna para colorir pontos (opcional)
        size_column: Coluna para definir tamanho dos pontos (opcional)
        title: Título personalizado (opcional)
        
    Returns:
        Informações sobre o scatter plot criado
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        if x_column not in df.columns:
            return {"error": f"Coluna X '{x_column}' não encontrada"}
        
        if y_column not in df.columns:
            return {"error": f"Coluna Y '{y_column}' não encontrada"}
        
        if color_column and color_column not in df.columns:
            return {"error": f"Coluna de cor '{color_column}' não encontrada"}
        
        if size_column and size_column not in df.columns:
            return {"error": f"Coluna de tamanho '{size_column}' não encontrada"}
        
        # Criar scatter plot
        fig = viz_helper.create_scatter_plot(df, x_column, y_column, color_column, size_column, title)
        
        # Adicionar à lista de visualizações
        chart_title = title or f"Dispersão: {x_column} vs {y_column}"
        add_visualization("scatter_plot", fig, chart_title,
                         f"Relação entre {x_column} e {y_column}")
        
        # Calcular correlação se ambas forem numéricas
        correlation = None
        if (pd.api.types.is_numeric_dtype(df[x_column]) and 
            pd.api.types.is_numeric_dtype(df[y_column])):
            correlation = df[x_column].corr(df[y_column])
        
        result = {
            "chart_type": "scatter_plot",
            "x_column": x_column,
            "y_column": y_column,
            "color_column": color_column,
            "size_column": size_column,
            "title": chart_title,
            "correlation": float(correlation) if correlation is not None else None,
            "success": True
        }
        
        logger.info(f"Scatter plot criado: '{x_column}' vs '{y_column}'")
        return result
        
    except Exception as e:
        error_msg = f"Erro ao criar scatter plot: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def create_correlation_matrix(columns: str = None, title: str = None) -> Dict[str, Any]:
    """
    Cria uma matriz de correlação para variáveis numéricas.
    
    Args:
        columns: Lista de colunas separadas por vírgula (opcional, usa todas numéricas se não especificado)
        title: Título personalizado (opcional)
        
    Returns:
        Informações sobre a matriz de correlação criada
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        # Selecionar colunas numéricas
        if columns:
            column_list = [col.strip() for col in columns.split(',')]
            # Verificar se todas existem
            missing_cols = [col for col in column_list if col not in df.columns]
            if missing_cols:
                return {"error": f"Colunas não encontradas: {missing_cols}"}
            
            # Verificar se são numéricas
            non_numeric = [col for col in column_list if not pd.api.types.is_numeric_dtype(df[col])]
            if non_numeric:
                return {"error": f"Colunas não numéricas: {non_numeric}"}
        else:
            column_list = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(column_list) < 2:
            return {"error": "Necessário pelo menos 2 colunas numéricas para correlação"}
        
        # Criar matriz de correlação
        fig = viz_helper.create_correlation_heatmap(df, column_list, title)
        
        # Adicionar à lista de visualizações
        chart_title = title or "Matriz de Correlação"
        add_visualization("correlation_matrix", fig, chart_title,
                         f"Correlações entre {len(column_list)} variáveis numéricas")
        
        # Calcular estatísticas da correlação
        corr_matrix = df[column_list].corr()
        
        # Encontrar correlações mais fortes (excluindo diagonal)
        corr_values = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_values.append({
                    "var1": corr_matrix.columns[i],
                    "var2": corr_matrix.columns[j],
                    "correlation": corr_matrix.iloc[i, j]
                })
        
        # Ordenar por correlação absoluta
        corr_values.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        result = {
            "chart_type": "correlation_matrix",
            "columns_analyzed": column_list,
            "title": chart_title,
            "strongest_correlations": corr_values[:5],  # Top 5
            "total_correlations": len(corr_values),
            "success": True
        }
        
        logger.info(f"Matriz de correlação criada com {len(column_list)} variáveis")
        return result
        
    except Exception as e:
        error_msg = f"Erro ao criar matriz de correlação: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def create_bar_chart(column_name: str, orientation: str = "v", max_categories: int = 20,
                    title: str = None) -> Dict[str, Any]:
    """
    Cria um gráfico de barras para uma variável categórica.
    
    Args:
        column_name: Nome da coluna categórica
        orientation: Orientação do gráfico ('v' para vertical, 'h' para horizontal)
        max_categories: Número máximo de categorias a mostrar
        title: Título personalizado (opcional)
        
    Returns:
        Informações sobre o gráfico de barras criado
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        if column_name not in df.columns:
            return {"error": f"Coluna '{column_name}' não encontrada"}
        
        # Criar gráfico de barras
        fig = viz_helper.create_bar_chart(df, column_name, None, title, orientation)
        
        # Adicionar à lista de visualizações
        chart_title = title or f"Distribuição de {column_name}"
        add_visualization("bar_chart", fig, chart_title,
                         f"Frequência das categorias em {column_name}")
        
        # Calcular estatísticas
        value_counts = df[column_name].value_counts().head(max_categories)
        
        result = {
            "chart_type": "bar_chart",
            "column": column_name,
            "orientation": orientation,
            "title": chart_title,
            "statistics": {
                "unique_categories": df[column_name].nunique(),
                "most_frequent": value_counts.index[0] if not value_counts.empty else None,
                "frequency_of_most_frequent": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                "categories_shown": min(max_categories, len(value_counts))
            },
            "success": True
        }
        
        logger.info(f"Gráfico de barras criado para coluna '{column_name}'")
        return result
        
    except Exception as e:
        error_msg = f"Erro ao criar gráfico de barras para '{column_name}': {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def create_pie_chart(column_name: str, max_categories: int = 10, title: str = None) -> Dict[str, Any]:
    """
    Cria um gráfico de pizza para uma variável categórica.
    
    Args:
        column_name: Nome da coluna categórica
        max_categories: Número máximo de categorias a mostrar
        title: Título personalizado (opcional)
        
    Returns:
        Informações sobre o gráfico de pizza criado
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        if column_name not in df.columns:
            return {"error": f"Coluna '{column_name}' não encontrada"}
        
        # Verificar se tem muitas categorias
        unique_values = df[column_name].nunique()
        if unique_values > max_categories:
            logger.warning(f"Coluna '{column_name}' tem {unique_values} categorias. Mostrando apenas as top {max_categories}")
        
        # Criar gráfico de pizza
        fig = viz_helper.create_pie_chart(df, column_name, title, max_categories)
        
        # Adicionar à lista de visualizações
        chart_title = title or f"Distribuição de {column_name}"
        add_visualization("pie_chart", fig, chart_title,
                         f"Proporções das categorias em {column_name}")
        
        # Calcular estatísticas
        value_counts = df[column_name].value_counts().head(max_categories)
        total = value_counts.sum()
        
        result = {
            "chart_type": "pie_chart",
            "column": column_name,
            "title": chart_title,
            "statistics": {
                "total_categories": unique_values,
                "categories_shown": len(value_counts),
                "largest_slice": {
                    "category": value_counts.index[0],
                    "count": int(value_counts.iloc[0]),
                    "percentage": float(value_counts.iloc[0] / total * 100)
                },
                "total_represented": int(total),
                "total_dataset": len(df)
            },
            "success": True
        }
        
        logger.info(f"Gráfico de pizza criado para coluna '{column_name}'")
        return result
        
    except Exception as e:
        error_msg = f"Erro ao criar gráfico de pizza para '{column_name}': {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@tool
def create_dashboard_summary() -> Dict[str, Any]:
    """
    Cria um dashboard resumo com as principais visualizações do dataset.
    
    Returns:
        Informações sobre o dashboard criado
    """
    try:
        df = get_current_dataframe()
        if df is None:
            return {"error": "Nenhum dataset foi carregado"}
        
        # Criar múltiplas visualizações
        figures = viz_helper.create_summary_dashboard(df)
        
        # Adicionar cada figura à lista de visualizações
        for i, fig in enumerate(figures):
            add_visualization(
                "dashboard_item", 
                fig, 
                f"Dashboard Item {i+1}",
                "Visualização resumo do dataset"
            )
        
        # Informações sobre o que foi criado
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        result = {
            "chart_type": "dashboard",
            "title": "Dashboard Resumo",
            "visualizations_created": len(figures),
            "dataset_summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols)
            },
            "success": True
        }
        
        logger.info(f"Dashboard resumo criado com {len(figures)} visualizações")
        return result
        
    except Exception as e:
        error_msg = f"Erro ao criar dashboard resumo: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def get_visualization_tools() -> List:
    """Retorna lista de todas as ferramentas de visualização"""
    return [
        create_histogram,
        create_box_plot,
        create_scatter_plot,
        create_correlation_matrix,
        create_bar_chart,
        create_pie_chart,
        create_dashboard_summary
    ]