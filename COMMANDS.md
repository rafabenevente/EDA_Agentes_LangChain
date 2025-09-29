# Comandos Úteis para o Projeto EDA Agentes LangChain

## Setup Inicial

### Criar e ativar ambiente conda
```powershell
conda env create -f environment.yml
conda activate eda_lang
```

### Instalar dependências adicionais
```powershell
pip install -r requirements.txt
```

### Configurar variáveis de ambiente
```powershell
Copy-Item .env.example .env
# Editar .env com suas configurações
```

## Desenvolvimento

### Executar aplicação
```powershell
conda activate eda_lang
streamlit run app.py
```

### Executar testes
```powershell
# Todos os testes
pytest tests/ -v

# Testes com cobertura
pytest tests/ --cov=. --cov-report=html

# Testes específicos
pytest tests/test_agents.py -v

# Testes marcados como rápidos
pytest -m "not slow"
```

### Formatação e linting
```powershell
# Formatação automática
black .

# Organizar imports
isort .

# Linting
flake8 .

# Verificação de tipos
mypy .

# Executar todos os checks
black . ; isort . ; flake8 . ; mypy .
```

### Gerenciamento de dependências
```powershell
# Atualizar environment.yml
conda env export --no-builds > environment.yml

# Atualizar requirements.txt
pip freeze > requirements.txt

# Instalar nova dependência
conda install package_name
# ou
pip install package_name
```

## Estrutura de Comandos para Deploy

### Preparar ambiente para produção
```powershell
# Definir ambiente como produção
$env:ENVIRONMENT="production"

# Executar com configurações de produção
streamlit run app.py --server.port 8501
```

### Docker (quando implementado)
```powershell
# Build da imagem
docker build -t eda-agents .

# Executar container
docker run -p 8501:8501 --env-file .env eda-agents
```

## Comandos de Debug

### Verificar configurações
```powershell
python -c "from config.settings import get_settings; print(get_settings())"
```

### Testar conexão com Google API
```powershell
python -c "from langchain_google_genai import ChatGoogleGenerativeAI; from config.settings import get_settings; llm = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=get_settings().google_api_key); print(llm.invoke('Hello!'))"
```

### Verificar estrutura do projeto
```powershell
tree /F
```

## Comandos Git

### Setup inicial do repositório
```powershell
git init
git add .
git commit -m "Initial commit: Setup EDA Agentes LangChain project"
```

### Workflow de desenvolvimento
```powershell
# Criar nova branch para feature
git checkout -b feature/nova-funcionalidade

# Commit das mudanças
git add .
git commit -m "feat: adiciona nova funcionalidade"

# Push da branch
git push origin feature/nova-funcionalidade
```

## Monitoramento e Logs

### Visualizar logs em tempo real
```powershell
Get-Content logs/eda_agent.log -Wait
```

### Limpar cache
```powershell
Remove-Item data/cache/* -Recurse -Force
```

### Verificar uso de memória
```powershell
python -c "import psutil; print(f'Memória: {psutil.virtual_memory().percent}%')"
```

## Utilitários

### Gerar dados de teste
```powershell
python -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Gerar dataset de exemplo
np.random.seed(42)
data = {
    'vendas': np.random.normal(1000, 200, 100),
    'regiao': np.random.choice(['Norte', 'Sul', 'Leste', 'Oeste'], 100),
    'data': [datetime.now() - timedelta(days=x) for x in range(100)]
}
df = pd.DataFrame(data)
df.to_csv('tests/test_data/vendas_exemplo.csv', index=False)
print('Dataset de teste criado!')
"
```

### Backup dos dados
```powershell
# Criar backup
Compress-Archive -Path data/ -DestinationPath "backup_data_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip"
```