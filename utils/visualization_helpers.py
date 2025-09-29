"""
Helpers para criação e formatação de visualizações
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from config.settings import get_settings

logger = logging.getLogger(__name__)


class VisualizationHelpers:
    """Classe com funções auxiliares para criação de visualizações"""
    
    def __init__(self):
        self.settings = get_settings()
        self.color_palette = px.colors.qualitative.Set1
        self.theme = {
            'background_color': '#ffffff',
            'grid_color': '#f0f0f0',
            'text_color': '#2c3e50',
            'primary_color': '#3498db'
        }
    
    def setup_plotly_theme(self) -> Dict[str, Any]:
        """Configura tema padrão para gráficos Plotly"""
        return {
            'layout': {
                'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': self.theme['text_color']},
                'plot_bgcolor': self.theme['background_color'],
                'paper_bgcolor': self.theme['background_color'],
                'xaxis': {'gridcolor': self.theme['grid_color']},
                'yaxis': {'gridcolor': self.theme['grid_color']},
                'colorway': self.color_palette
            }
        }
    
    def create_histogram(self, df: pd.DataFrame, column: str, bins: int = 30, 
                        title: str = None, color: str = None) -> go.Figure:
        """Cria um histograma para uma coluna numérica"""
        try:
            if title is None:
                title = f"Distribuição de {column}"
            
            fig = px.histogram(
                df, 
                x=column, 
                nbins=bins,
                title=title,
                color_discrete_sequence=[color or self.color_palette[0]]
            )
            
            fig.update_layout(**self.setup_plotly_theme()['layout'])
            fig.update_layout(
                xaxis_title=column,
                yaxis_title="Frequência",
                showlegend=False
            )
            
            # Adicionar estatísticas no hover
            fig.update_traces(
                hovertemplate=f"<b>{column}</b>: %{{x}}<br>" +
                             "<b>Frequência</b>: %{y}<br>" +
                             "<extra></extra>"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro ao criar histograma para {column}: {e}")
            return self.create_error_figure(f"Erro ao criar histograma: {e}")
    
    def create_box_plot(self, df: pd.DataFrame, column: str, by_column: str = None,
                       title: str = None) -> go.Figure:
        """Cria um box plot para uma coluna numérica"""
        try:
            if title is None:
                if by_column:
                    title = f"Box Plot de {column} por {by_column}"
                else:
                    title = f"Box Plot de {column}"
            
            if by_column and by_column in df.columns:
                fig = px.box(df, x=by_column, y=column, title=title)
            else:
                fig = px.box(df, y=column, title=title)
            
            fig.update_layout(**self.setup_plotly_theme()['layout'])
            fig.update_layout(
                yaxis_title=column,
                xaxis_title=by_column if by_column else ""
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro ao criar box plot para {column}: {e}")
            return self.create_error_figure(f"Erro ao criar box plot: {e}")
    
    def create_scatter_plot(self, df: pd.DataFrame, x_column: str, y_column: str,
                           color_column: str = None, size_column: str = None,
                           title: str = None) -> go.Figure:
        """Cria um gráfico de dispersão"""
        try:
            if title is None:
                title = f"Dispersão: {x_column} vs {y_column}"
            
            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                color=color_column,
                size=size_column,
                title=title,
                hover_data=df.columns.tolist()
            )
            
            fig.update_layout(**self.setup_plotly_theme()['layout'])
            fig.update_layout(
                xaxis_title=x_column,
                yaxis_title=y_column
            )
            
            # Adicionar linha de tendência se ambas colunas forem numéricas
            if df[x_column].dtype in ['int64', 'float64'] and df[y_column].dtype in ['int64', 'float64']:
                fig.add_scatter(
                    x=df[x_column],
                    y=df[y_column].rolling(window=min(10, len(df))).mean(),
                    mode='lines',
                    name='Tendência',
                    line=dict(color='red', dash='dash')
                )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro ao criar scatter plot: {e}")
            return self.create_error_figure(f"Erro ao criar gráfico de dispersão: {e}")
    
    def create_correlation_heatmap(self, df: pd.DataFrame, columns: List[str] = None,
                                  title: str = None) -> go.Figure:
        """Cria um heatmap de correlação"""
        try:
            if columns is None:
                # Selecionar apenas colunas numéricas
                numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                columns = numeric_columns[:self.settings.max_correlation_matrix_size]
            
            if len(columns) < 2:
                return self.create_error_figure("Necessário pelo menos 2 colunas numéricas para correlação")
            
            # Calcular matriz de correlação
            corr_matrix = df[columns].corr()
            
            if title is None:
                title = "Matriz de Correlação"
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title=title,
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1
            )
            
            fig.update_layout(**self.setup_plotly_theme()['layout'])
            fig.update_layout(
                xaxis_title="Variáveis",
                yaxis_title="Variáveis"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro ao criar heatmap de correlação: {e}")
            return self.create_error_figure(f"Erro ao criar matriz de correlação: {e}")
    
    def create_bar_chart(self, df: pd.DataFrame, x_column: str, y_column: str = None,
                        title: str = None, orientation: str = 'v') -> go.Figure:
        """Cria um gráfico de barras"""
        try:
            if y_column is None:
                # Contar valores únicos
                value_counts = df[x_column].value_counts().head(20)  # Limitar a 20 valores
                
                if title is None:
                    title = f"Distribuição de {x_column}"
                
                if orientation == 'h':
                    fig = px.bar(
                        x=value_counts.values,
                        y=value_counts.index,
                        orientation='h',
                        title=title,
                        labels={'x': 'Frequência', 'y': x_column}
                    )
                else:
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=title,
                        labels={'x': x_column, 'y': 'Frequência'}
                    )
            else:
                if title is None:
                    title = f"{y_column} por {x_column}"
                
                fig = px.bar(df, x=x_column, y=y_column, title=title)
            
            fig.update_layout(**self.setup_plotly_theme()['layout'])
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro ao criar gráfico de barras: {e}")
            return self.create_error_figure(f"Erro ao criar gráfico de barras: {e}")
    
    def create_line_chart(self, df: pd.DataFrame, x_column: str, y_column: str,
                         color_column: str = None, title: str = None) -> go.Figure:
        """Cria um gráfico de linhas"""
        try:
            if title is None:
                title = f"{y_column} ao longo de {x_column}"
            
            fig = px.line(
                df,
                x=x_column,
                y=y_column,
                color=color_column,
                title=title
            )
            
            fig.update_layout(**self.setup_plotly_theme()['layout'])
            fig.update_layout(
                xaxis_title=x_column,
                yaxis_title=y_column
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro ao criar gráfico de linhas: {e}")
            return self.create_error_figure(f"Erro ao criar gráfico de linhas: {e}")
    
    def create_pie_chart(self, df: pd.DataFrame, column: str, title: str = None,
                        max_categories: int = 10) -> go.Figure:
        """Cria um gráfico de pizza"""
        try:
            value_counts = df[column].value_counts().head(max_categories)
            
            if title is None:
                title = f"Distribuição de {column}"
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=title
            )
            
            fig.update_layout(**self.setup_plotly_theme()['layout'])
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro ao criar gráfico de pizza: {e}")
            return self.create_error_figure(f"Erro ao criar gráfico de pizza: {e}")
    
    def create_multiple_histograms(self, df: pd.DataFrame, columns: List[str],
                                  title: str = None) -> go.Figure:
        """Cria múltiplos histogramas em subplots"""
        try:
            n_cols = min(3, len(columns))
            n_rows = (len(columns) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=columns,
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            for i, column in enumerate(columns):
                row = i // n_cols + 1
                col = i % n_cols + 1
                
                fig.add_trace(
                    go.Histogram(x=df[column], name=column, showlegend=False),
                    row=row,
                    col=col
                )
            
            if title is None:
                title = "Distribuições das Variáveis Numéricas"
            
            fig.update_layout(
                title=title,
                **self.setup_plotly_theme()['layout']
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro ao criar múltiplos histogramas: {e}")
            return self.create_error_figure(f"Erro ao criar histogramas múltiplos: {e}")
    
    def create_error_figure(self, error_message: str) -> go.Figure:
        """Cria uma figura de erro quando não é possível gerar a visualização"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"Erro na visualização:<br>{error_message}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor='center',
            yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            title="Erro na Visualização",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            **self.setup_plotly_theme()['layout']
        )
        
        return fig
    
    def get_optimal_chart_type(self, df: pd.DataFrame, column: str) -> str:
        """Sugere o melhor tipo de gráfico para uma coluna"""
        try:
            col_dtype = df[column].dtype
            unique_values = df[column].nunique()
            total_values = len(df)
            
            # Coluna numérica
            if col_dtype in ['int64', 'float64']:
                if unique_values > 20:
                    return 'histogram'
                else:
                    return 'bar_chart'
            
            # Coluna categórica
            elif col_dtype in ['object', 'category']:
                if unique_values <= 10:
                    return 'pie_chart'
                elif unique_values <= 20:
                    return 'bar_chart'
                else:
                    return 'bar_chart'  # Top N categories
            
            # Coluna datetime
            elif 'datetime' in str(col_dtype):
                return 'line_chart'
            
            # Coluna booleana
            elif col_dtype == 'bool':
                return 'pie_chart'
            
            else:
                return 'bar_chart'
                
        except Exception as e:
            logger.error(f"Erro ao determinar tipo de gráfico para {column}: {e}")
            return 'bar_chart'
    
    def create_summary_dashboard(self, df: pd.DataFrame) -> List[go.Figure]:
        """Cria um dashboard resumo com as principais visualizações"""
        try:
            figures = []
            
            # Selecionar colunas numéricas e categóricas
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 1. Histogramas das variáveis numéricas (se houver)
            if numeric_columns:
                if len(numeric_columns) > 1:
                    fig_hist = self.create_multiple_histograms(
                        df, 
                        numeric_columns[:6],  # Limitar a 6 para não sobrecarregar
                        "Distribuições das Variáveis Numéricas"
                    )
                    figures.append(fig_hist)
                else:
                    fig_hist = self.create_histogram(df, numeric_columns[0])
                    figures.append(fig_hist)
            
            # 2. Matriz de correlação (se houver pelo menos 2 colunas numéricas)
            if len(numeric_columns) >= 2:
                fig_corr = self.create_correlation_heatmap(df, numeric_columns[:10])
                figures.append(fig_corr)
            
            # 3. Gráficos de barras para variáveis categóricas principais
            for col in categorical_columns[:3]:  # Limitar a 3 categóricas
                fig_bar = self.create_bar_chart(df, col)
                figures.append(fig_bar)
            
            return figures
            
        except Exception as e:
            logger.error(f"Erro ao criar dashboard resumo: {e}")
            return [self.create_error_figure(f"Erro ao criar dashboard: {e}")]