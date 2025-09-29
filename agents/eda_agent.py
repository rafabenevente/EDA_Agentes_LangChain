"""
Agente principal para Análise Exploratória de Dados (EDA)
Integra todas as ferramentas de análise com LangChain e Google Gemini Pro
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings
from utils.memory_manager import MemoryManager
from tools.data_analysis_tools import (
    describe_dataset, analyze_data_types, get_column_info, 
    get_data_quality_report, compare_columns
)
from tools.visualization_tools import (
    create_histogram, create_box_plot, create_scatter_plot,
    create_correlation_matrix, create_bar_chart, create_pie_chart,
    create_dashboard_summary
)
from tools.statistical_tools import (
    calculate_correlation_analysis, perform_normality_tests,
    calculate_descriptive_statistics, perform_pca_analysis
)
from tools.outlier_detection_tools import (
    detect_outliers_iqr, detect_outliers_zscore,
    detect_outliers_isolation_forest, detect_outliers_lof,
    compare_outlier_methods
)

logger = logging.getLogger(__name__)

# Variável global para armazenar o dataframe atual
_current_dataframe: Optional[pd.DataFrame] = None


def set_current_dataframe(df: pd.DataFrame) -> None:
    """Define o dataframe atual para uso pelas ferramentas"""
    global _current_dataframe
    _current_dataframe = df
    logger.info(f"Dataframe definido: {df.shape[0]} linhas, {df.shape[1]} colunas")


def get_current_dataframe() -> Optional[pd.DataFrame]:
    """Retorna o dataframe atual"""
    return _current_dataframe


class EDAAgent:
    """
    Agente principal para Análise Exploratória de Dados
    Utiliza LangChain com Google Gemini Pro para análise conversacional de datasets CSV
    """
    
    def __init__(self, memory_key: str = "default"):
        """
        Inicializa o agente EDA
        
        Args:
            memory_key: Chave para identificar a sessão de conversa
        """
        self.memory_key = memory_key
        self.memory_manager = MemoryManager()
        self.memory = self.memory_manager.get_memory(memory_key)
        
        # Inicializar LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.google_model,
            google_api_key=settings.google_api_key,
            temperature=settings.google_temperature,
            max_tokens=settings.google_max_tokens,
            convert_system_message_to_human=True
        )
        
        # Coletar todas as ferramentas
        self.tools = self._get_all_tools()
        
        # Criar prompt template
        self.prompt = self._create_prompt_template()
        
        # Criar agente
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Criar executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=settings.max_agent_iterations,
            early_stopping_method="generate"
        )
        
        logger.info("EDAAgent inicializado com sucesso")
    
    def _get_all_tools(self):
        """Coleta todas as ferramentas disponíveis"""
        tools = []
        
        # Ferramentas de análise de dados
        tools.extend([
            describe_dataset,
            analyze_data_types,
            get_column_info,
            get_data_quality_report,
            compare_columns
        ])
        
        # Ferramentas de visualização
        tools.extend([
            create_histogram,
            create_box_plot,
            create_scatter_plot,
            create_correlation_matrix,
            create_bar_chart,
            create_pie_chart,
            create_dashboard_summary
        ])
        
        # Ferramentas estatísticas
        tools.extend([
            calculate_correlation_analysis,
            perform_normality_tests,
            calculate_descriptive_statistics,
            perform_pca_analysis
        ])
        
        # Ferramentas de detecção de outliers
        tools.extend([
            detect_outliers_iqr,
            detect_outliers_zscore,
            detect_outliers_isolation_forest,
            detect_outliers_lof,
            compare_outlier_methods
        ])
        
        logger.info(f"Total de {len(tools)} ferramentas carregadas")
        return tools
    
    def _create_prompt_template(self):
        """Cria o template de prompt para o agente"""
        
        system_message = """Você é um especialista em Análise Exploratória de Dados (EDA) com vasta experiência em estatística, visualização de dados e machine learning. 

Sua missão é ajudar usuários a compreender profundamente seus datasets CSV através de análises conversacionais inteligentes.

## SUAS CAPACIDADES:

### 1. ANÁLISE DE DADOS
- Descrição completa de datasets (estatísticas descritivas, tipos de dados, qualidade)
- Análise de colunas individuais e comparações entre variáveis
- Detecção de padrões, tendências e anomalias nos dados
- Avaliação da qualidade dos dados (valores nulos, duplicatas, inconsistências)

### 2. ANÁLISE ESTATÍSTICA
- Cálculo de correlações entre variáveis com interpretação detalhada
- Testes de normalidade (Shapiro-Wilk, D'Agostino)
- Estatísticas descritivas avançadas com interpretação contextual
- Análise de Componentes Principais (PCA) para redução de dimensionalidade

### 3. DETECÇÃO DE OUTLIERS/ANOMALIAS
- Método IQR (Interquartile Range) para detecção clássica
- Z-Score para identificação baseada em desvio padrão
- Isolation Forest para detecção não-supervisionada avançada
- Local Outlier Factor (LOF) para anomalias locais
- Comparação entre métodos para análise robusta

### 4. VISUALIZAÇÕES INTERATIVAS
- Histogramas para distribuições de variáveis
- Box plots para análise de quartis e outliers
- Scatter plots para relações entre variáveis
- Mapas de correlação (heatmaps) para análise multivariada
- Gráficos de barras e pizza para variáveis categóricas
- Dashboards resumidos com múltiplas visualizações

## INSTRUÇÕES ESPECÍFICAS:

### Comportamento Conversacional:
- Seja proativo: sugira análises relevantes baseadas no contexto
- Explique resultados de forma clara e acessível
- Use analogias quando necessário para facilitar compreensão
- Sempre forneça insights acionáveis baseados nos resultados

### Análise Sistemática:
1. **Primeiro contato**: Sempre comece com descrição geral do dataset
2. **Identificação de problemas**: Detecte e reporte questões de qualidade
3. **Análise progressiva**: Aprofunde baseado nas perguntas do usuário
4. **Visualizações complementares**: Sempre que apropriado, crie visualizações
5. **Conclusões e recomendações**: Finalize com insights claros

### Interpretação de Resultados:
- Para correlações: explique força, direção e significância prática
- Para outliers: contextualize se são erros ou observações válidas interessantes
- Para distribuições: comente normalidade, assimetria e implicações
- Para visualizações: destaque padrões, tendências e pontos de atenção

### Tratamento de Erros:
- Se uma ferramenta falhar, explique o problema e sugira alternativas
- Para dados problemáticos, oriente sobre possíveis soluções
- Sempre mantenha o foco na análise mesmo com limitações técnicas

## DIRETRIZES DE COMUNICAÇÃO:

- **Tom**: Professoral mas acessível, como um mentor experiente
- **Estrutura**: Use markdown para organizar respostas complexas
- **Detalhamento**: Equilibre profundidade técnica com clareza
- **Proatividade**: Sugira próximos passos e análises complementares
- **Precisão**: Baseie todas as afirmações nos resultados das ferramentas

Lembre-se: Seu objetivo é transformar dados brutos em insights valiosos através de uma experiência conversacional rica e educativa."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        return prompt
    
    def load_dataset(self, df: pd.DataFrame, filename: str = None) -> Dict[str, Any]:
        """
        Carrega um dataset para análise
        
        Args:
            df: DataFrame do pandas
            filename: Nome do arquivo (opcional)
            
        Returns:
            Informações sobre o carregamento
        """
        try:
            # Definir dataframe global
            set_current_dataframe(df)
            
            # Análise inicial automática
            basic_info = {
                "filename": filename or "dataset",
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
                "load_success": True
            }
            
            logger.info(f"Dataset carregado: {basic_info}")
            return basic_info
            
        except Exception as e:
            error_msg = f"Erro ao carregar dataset: {str(e)}"
            logger.error(error_msg)
            return {"load_success": False, "error": error_msg}
    
    def analyze(self, question: str) -> str:
        """
        Processa uma pergunta sobre o dataset usando o agente
        
        Args:
            question: Pergunta do usuário sobre os dados
            
        Returns:
            Resposta do agente com análise
        """
        try:
            if _current_dataframe is None:
                return "❌ **Erro**: Nenhum dataset foi carregado. Por favor, carregue um arquivo CSV primeiro."
            
            # Executar análise
            response = self.agent_executor.invoke({
                "input": question,
                "chat_history": self.memory.chat_memory.messages
            })
            
            return response["output"]
            
        except Exception as e:
            error_msg = f"Erro na análise: {str(e)}"
            logger.error(error_msg)
            return f"❌ **Erro na análise**: {error_msg}"
    
    def get_conversation_history(self) -> list:
        """Retorna o histórico da conversa"""
        return self.memory_manager.get_conversation_history(self.memory_key)
    
    def clear_memory(self) -> None:
        """Limpa a memória da conversa"""
        self.memory_manager.clear_memory(self.memory_key)
        logger.info("Memória da conversa limpa")
    
    def export_conversation(self, filepath: str) -> bool:
        """Exporta a conversa para arquivo"""
        return self.memory_manager.export_conversation(self.memory_key, filepath)
    
    def get_available_tools(self) -> Dict[str, str]:
        """Retorna informações sobre as ferramentas disponíveis"""
        tools_info = {}
        
        for tool in self.tools:
            tools_info[tool.name] = tool.description
        
        return tools_info
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Retorna informações básicas sobre o dataset atual"""
        if _current_dataframe is None:
            return {"error": "Nenhum dataset carregado"}
        
        df = _current_dataframe
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
            "null_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum())
        }
    
    def suggest_initial_analysis(self) -> str:
        """Sugere análises iniciais para um novo dataset"""
        if _current_dataframe is None:
            return "Nenhum dataset carregado para sugerir análises."
        
        suggestions = """
## 🔍 **Sugestões de Análise Inicial**

Agora que seu dataset está carregado, aqui estão algumas análises que posso realizar:

### 📊 **Análise Básica**
- *"Descreva o dataset"* - Visão geral completa dos dados
- *"Qual a qualidade dos dados?"* - Relatório de qualidade detalhado
- *"Analise os tipos de dados"* - Verificação de tipos e possíveis problemas

### 📈 **Análise Estatística**
- *"Calcule as correlações entre as variáveis"* - Matriz de correlação completa
- *"Faça uma análise estatística descritiva"* - Estatísticas avançadas
- *"Teste a normalidade das variáveis numéricas"* - Testes estatísticos

### 🎯 **Detecção de Anomalias**
- *"Detecte outliers usando diferentes métodos"* - Análise robusta de anomalias
- *"Compare métodos de detecção de outliers"* - Análise comparativa

### 📊 **Visualizações**
- *"Crie um dashboard resumo"* - Visão geral visual dos dados
- *"Mostre a distribuição das variáveis numéricas"* - Histogramas
- *"Crie uma matriz de correlação visual"* - Heatmap interativo

### 💡 **Dica**: 
Você pode fazer perguntas específicas sobre colunas individuais ou solicitar análises customizadas. Sou proativo em sugerir visualizações e análises complementares!

**O que gostaria de explorar primeiro?**
        """
        
        return suggestions


# Funções auxiliares para usar nas ferramentas
def get_current_dataframe() -> Optional[pd.DataFrame]:
    """Função global para acessar o dataframe atual (usada pelas ferramentas)"""
    return _current_dataframe