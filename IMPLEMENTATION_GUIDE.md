# Guia de Implementação - EDA Agentes LangChain com Streamlit

## Visão Geral do Projeto

Este projeto implementa uma ferramenta de Análise Exploratória de Dados (EDA) usando agentes LangChain integrados a uma interface Streamlit. A ferramenta permite que usuários façam perguntas em linguagem natural sobre arquivos CSV e recebam análises completas com visualizações.

## Objetivos

- Criar agentes inteligentes capazes de analisar dados CSV
- Responder perguntas sobre tipos de dados, distribuições, padrões e anomalias
- Gerar visualizações automáticas para embasar as respostas
- Detectar correlações e relações entre variáveis
- Fornecer conclusões baseadas nas análises realizadas
- Manter memória das análises para conversações contextuais

## Arquitetura do Sistema

### Componentes Principais

1. **Interface Streamlit**: Front-end para upload de arquivos e interação com usuário
2. **Agentes LangChain**: Sistema inteligente de análise de dados
3. **Ferramentas Customizadas**: Tools específicos para operações de EDA
4. **Sistema de Memória**: Manutenção do contexto das análises
5. **Motor de Visualizações**: Geração automática de gráficos

### Stack Tecnológico

- **Framework Web**: Streamlit (>=1.28.0)
- **Agentes IA**: LangChain (>=0.0.350)
- **LLM Provider**: Google (Gemini Pro via langchain-google-genai)
- **Manipulação de Dados**: pandas (>=2.1.0), numpy (>=1.24.0)
- **Visualizações**: plotly (>=5.17.0), seaborn (>=0.12.0), matplotlib (>=3.7.0)
- **Análise Estatística**: scipy (>=1.11.0), scikit-learn (>=1.3.0)
- **Configuração**: python-dotenv (>=1.0.0)

## Estrutura do Projeto

```
EDA_Agentes_LangChain/
├── .env                          # Configurações do ambiente
├── .env.example                  # Template de configurações
├── requirements.txt              # Dependências do projeto
├── environment.yml               # Configuração do ambiente Conda
├── app.py                        # Aplicação principal Streamlit
├── config/
│   └── settings.py              # Configurações centralizadas
├── agents/
│   ├── __init__.py
│   ├── eda_agent.py             # Agente principal de EDA
│   ├── data_explorer_agent.py   # Agente explorador de dados
│   ├── statistician_agent.py    # Agente estatístico
│   └── visualizer_agent.py      # Agente de visualizações
├── tools/
│   ├── __init__.py
│   ├── data_analysis_tools.py   # Ferramentas de análise
│   ├── visualization_tools.py   # Ferramentas de visualização
│   ├── statistical_tools.py     # Ferramentas estatísticas
│   └── outlier_detection_tools.py # Detecção de anomalias
├── utils/
│   ├── __init__.py
│   ├── data_loader.py           # Carregamento de dados
│   ├── memory_manager.py        # Gerenciamento de memória
│   └── visualization_helpers.py # Helpers para visualizações
├── data/
│   ├── uploads/                 # Arquivos CSV enviados pelos usuários
│   └── cache/                   # Cache de análises
├── tests/
│   ├── __init__.py
│   ├── test_agents.py           # Testes dos agentes
│   ├── test_tools.py            # Testes das ferramentas
│   └── test_data/               # Dados de teste
│       └── sample.csv
└── docs/
    └── API.md                   # Documentação da API
```

## Implementação Detalhada

### 1. Configuração do Ambiente

#### Arquivo .env
```bash
# Google AI Configuration
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-pro

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# Data Configuration
MAX_FILE_SIZE_MB=100
ALLOWED_EXTENSIONS=csv

# Cache Configuration
ENABLE_CACHE=true
CACHE_TTL_SECONDS=3600

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/eda_agent.log
```

#### Arquivo environment.yml
```yaml
name: eda_lang
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
    - streamlit>=1.28.0
    - langchain>=0.0.350
    - langchain-google-genai>=0.0.5
    - pandas>=2.1.0
    - numpy>=1.24.0
    - plotly>=5.17.0
    - seaborn>=0.12.0
    - matplotlib>=3.7.0
    - scipy>=1.11.0
    - scikit-learn>=1.3.0
    - python-dotenv>=1.0.0
    - streamlit-plotly-events>=0.0.6
    - pytest>=7.4.0
    - black>=23.0.0
    - flake8>=6.0.0
```

### 2. Agente Principal (agents/eda_agent.py)

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from tools.data_analysis_tools import get_data_analysis_tools
from tools.visualization_tools import get_visualization_tools
from tools.statistical_tools import get_statistical_tools
from tools.outlier_detection_tools import get_outlier_detection_tools

class EDAAgent:
    def __init__(self, google_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=google_api_key,
            temperature=0.1
        )
        self.memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Combinar todas as ferramentas
        self.tools = (
            get_data_analysis_tools() +
            get_visualization_tools() +
            get_statistical_tools() +
            get_outlier_detection_tools()
        )
        
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um especialista em Análise Exploratória de Dados (EDA).
            Sua função é analisar datasets CSV e responder perguntas sobre:
            - Tipos de dados e distribuições
            - Padrões e tendências
            - Anomalias e outliers
            - Relações entre variáveis
            - Conclusões e insights
            
            Sempre use as ferramentas disponíveis para gerar visualizações que embasem suas respostas.
            Mantenha o contexto das análises anteriores na conversa."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=10
        )
    
    def analyze(self, query: str, dataframe=None) -> dict:
        if dataframe is not None:
            # Armazenar dataframe no contexto
            self.memory.chat_memory.add_user_message(f"Dataset carregado com {len(dataframe)} linhas e {len(dataframe.columns)} colunas")
        
        response = self.agent_executor.invoke({"input": query})
        return response
```

### 3. Ferramentas de Análise (tools/data_analysis_tools.py)

```python
from langchain.tools import tool
import pandas as pd
import numpy as np
from typing import List, Dict, Any

@tool
def describe_dataset(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """Fornece descrição estatística completa do dataset"""
    return {
        "shape": dataframe.shape,
        "columns": list(dataframe.columns),
        "dtypes": dataframe.dtypes.to_dict(),
        "missing_values": dataframe.isnull().sum().to_dict(),
        "numeric_summary": dataframe.describe().to_dict(),
        "memory_usage": dataframe.memory_usage(deep=True).sum()
    }

@tool
def analyze_data_types(dataframe: pd.DataFrame) -> Dict[str, List[str]]:
    """Analisa e categoriza os tipos de dados das colunas"""
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = dataframe.select_dtypes(include=['datetime64']).columns.tolist()
    boolean_cols = dataframe.select_dtypes(include=['bool']).columns.tolist()
    
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "boolean": boolean_cols
    }

def get_data_analysis_tools() -> List:
    return [describe_dataset, analyze_data_types]
```

### 4. Interface Streamlit (app.py)

```python
import streamlit as st
import pandas as pd
from config.settings import load_settings
from agents.eda_agent import EDAAgent
from utils.data_loader import DataLoader
from utils.memory_manager import MemoryManager

def main():
    st.set_page_config(
        page_title="EDA Agentes LangChain",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🤖 Assistente Inteligente de EDA")
    st.markdown("Faça perguntas sobre seus dados em linguagem natural!")
    
    # Configurações
    settings = load_settings()
    
    # Inicializar agente
    if 'eda_agent' not in st.session_state:
        st.session_state.eda_agent = EDAAgent(settings.GOOGLE_API_KEY)
    
    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "Escolha um arquivo CSV",
        type=['csv'],
        help="Envie um arquivo CSV para análise"
    )
    
    if uploaded_file is not None:
        # Carregar dados
        dataloader = DataLoader()
        df = dataloader.load_csv(uploaded_file)
        
        if df is not None:
            st.success(f"Arquivo carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
            
            # Mostrar preview dos dados
            with st.expander("Preview dos Dados"):
                st.dataframe(df.head())
            
            # Interface de chat
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            # Mostrar histórico de mensagens
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Input do usuário
            if prompt := st.chat_input("Faça uma pergunta sobre os dados..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Resposta do agente
                with st.chat_message("assistant"):
                    with st.spinner("Analisando..."):
                        response = st.session_state.eda_agent.analyze(prompt, df)
                        st.markdown(response["output"])
                        
                        # Adicionar visualizações se houver
                        if "visualizations" in response:
                            for viz in response["visualizations"]:
                                st.plotly_chart(viz, use_container_width=True)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["output"]
                })

if __name__ == "__main__":
    main()
```

## Testes de Funcionalidade

### 1. Teste dos Agentes (tests/test_agents.py)

```python
import pytest
import pandas as pd
from agents.eda_agent import EDAAgent
import os

class TestEDAAgent:
    def setup_method(self):
        self.agent = EDAAgent(os.getenv("GOOGLE_API_KEY"))
        self.sample_df = pd.read_csv("tests/test_data/sample.csv")
    
    def test_agent_initialization(self):
        assert self.agent is not None
        assert len(self.agent.tools) > 0
    
    def test_basic_analysis(self):
        response = self.agent.analyze(
            "Descreva os dados básicos deste dataset",
            self.sample_df
        )
        assert "output" in response
        assert len(response["output"]) > 0
    
    def test_correlation_analysis(self):
        response = self.agent.analyze(
            "Quais são as correlações entre as variáveis numéricas?",
            self.sample_df
        )
        assert "correlação" in response["output"].lower()
```

### 2. Teste das Ferramentas (tests/test_tools.py)

```python
import pytest
import pandas as pd
from tools.data_analysis_tools import describe_dataset, analyze_data_types

class TestDataAnalysisTools:
    def setup_method(self):
        self.df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'C', 'A', 'B'],
            'date_col': pd.date_range('2023-01-01', periods=5)
        })
    
    def test_describe_dataset(self):
        result = describe_dataset.func(self.df)
        assert result["shape"] == (5, 3)
        assert "numeric_col" in result["columns"]
    
    def test_analyze_data_types(self):
        result = analyze_data_types.func(self.df)
        assert "numeric_col" in result["numeric"]
        assert "categorical_col" in result["categorical"]
```

## Comandos de Instalação e Execução

### Configuração do Ambiente

```powershell
# Ativar ambiente conda
conda activate eda_lang

# Instalar dependências
conda env update -f environment.yml

# Instalar dependências adicionais via pip
pip install -r requirements.txt

# Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas configurações
```

### Execução da Aplicação

```powershell
# Ativar ambiente
conda activate eda_lang

# Executar aplicação Streamlit
streamlit run app.py

# Executar testes
pytest tests/ -v

# Formatação de código
black .

# Linting
flake8 .
```

## Checklist de Implementação

### Fase 1: Setup Inicial
- [ ] Configurar ambiente conda `eda_lang`
- [ ] Criar estrutura de pastas
- [ ] Configurar arquivos de dependências
- [ ] Implementar configurações básicas (.env)

### Fase 2: Agentes e Ferramentas
- [ ] Implementar agente principal EDAAgent
- [ ] Criar ferramentas de análise de dados
- [ ] Implementar ferramentas de visualização
- [ ] Desenvolver sistema de memória

### Fase 3: Interface Streamlit
- [ ] Criar interface de upload de arquivos
- [ ] Implementar chat interface
- [ ] Integrar visualizações
- [ ] Adicionar cache e otimizações

### Fase 4: Testes e Validação
- [ ] Escrever testes unitários
- [ ] Criar dados de teste
- [ ] Testar com datasets reais
- [ ] Validar performance

### Fase 5: Deploy e Documentação
- [ ] Preparar para deploy
- [ ] Documentar API
- [ ] Criar guia do usuário
- [ ] Otimizar para produção

## Considerações de Performance

- Implementar cache para análises repetitivas
- Limitar tamanho de arquivos CSV (100MB)
- Usar lazy loading para visualizações complexas
- Otimizar consultas ao LLM com prompts eficientes

## Próximos Passos

1. Implementar a estrutura básica seguindo este guia
2. Testar com datasets pequenos primeiro
3. Expandir funcionalidades gradualmente
4. Otimizar baseado no feedback dos usuários
5. Implementar features avançadas como ML automático