# 📊 EDA Agentes LangChain

Sistema inteligente de **Análise Exploratória de Dados (EDA)** usando **LangChain** e **Google Gemini Pro** com interface web **Streamlit**. Converse com seus dados CSV usando linguagem natural e receba análises completas com visualizações interativas.

## 🌟 Características Principais

- **🤖 Agente Conversacional**: Powered by Google Gemini Pro via LangChain
- **📊 Análise Completa**: Estatísticas descritivas, correlações, testes de normalidade
- **🎯 Detecção de Outliers**: 4 métodos (IQR, Z-Score, Isolation Forest, LOF)
- **📈 Visualizações Interativas**: Plotly, Seaborn, Matplotlib
- **💬 Interface Intuitiva**: Streamlit com chat conversacional
- **🧠 Memória Persistente**: Mantém contexto durante a sessão
- **🔧 21 Ferramentas Especializadas**: Para análise profunda de dados

## 🚀 Começando

### Pré-requisitos

- Python 3.11+
- Conda/Miniconda
- Chave de API do Google AI (Gemini Pro)

### Instalação

1. **Clone o repositório**
```bash
git clone <repository-url>
cd EDA_Agentes_LangChain
```

2. **Crie e ative o ambiente conda**
```bash
conda env create -f environment.yml
conda activate eda_lang
```

3. **Configure as variáveis de ambiente**
```bash
cp .env.example .env
# Edite o arquivo .env com sua chave da API do Google
```

4. **Execute a aplicação**
```bash
streamlit run app.py
```

## 📁 Estrutura do Projeto

Consulte o arquivo `IMPLEMENTATION_GUIDE.md` para uma visão detalhada da arquitetura e implementação.

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar com cobertura
pytest tests/ --cov=. --cov-report=html
```

## 🎯 Funcionalidades

- **Análise Automática de Dados**: Descrição estatística, tipos de dados, valores ausentes
- **Visualizações Inteligentes**: Gráficos interativos gerados automaticamente
- **Detecção de Anomalias**: Identificação de outliers e valores atípicos
- **Análise de Correlações**: Relações entre variáveis numéricas
- **Interface Conversacional**: Chat em linguagem natural
- **Memória Contextual**: Mantém histórico da conversa

## 📊 Exemplos de Perguntas

- "Descreva os dados básicos deste dataset"
- "Quais são as correlações entre as variáveis numéricas?"
- "Existem outliers nos dados?"
- "Mostre-me a distribuição da variável idade"
- "Quais são as principais conclusões sobre estes dados?"

## 🛠️ Desenvolvimento

```bash
# Formatação de código
black .

# Linting
flake8 .

# Organização de imports
isort .

# Verificação de tipos
mypy .
```

## 📈 Performance

- Suporte a arquivos CSV de até 100MB
- Cache inteligente para análises repetitivas
- Visualizações otimizadas com Plotly
- Resposta rápida via Google Gemini Pro

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## ✨ Autor

Rafael Benevente - Projeto EDA Agentes LangChain