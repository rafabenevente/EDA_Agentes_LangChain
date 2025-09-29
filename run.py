"""
Script de inicialização e teste do sistema EDA Agentes LangChain
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd

def check_environment():
    """Verifica se o ambiente está configurado corretamente"""
    print("🔍 Verificando ambiente...")
    
    # Verificar se estamos no conda env correto
    try:
        result = subprocess.run(['conda', 'info', '--envs'], 
                              capture_output=True, text=True, shell=True)
        if 'eda_lang' not in result.stdout:
            print("❌ Ambiente conda 'eda_lang' não encontrado")
            return False
        else:
            print("✅ Ambiente conda 'eda_lang' encontrado")
    except Exception as e:
        print(f"⚠️ Não foi possível verificar conda: {e}")
    
    # Verificar arquivo .env
    if not os.path.exists('.env'):
        print("❌ Arquivo .env não encontrado")
        print("   Crie o arquivo .env baseado no .env.example")
        return False
    else:
        print("✅ Arquivo .env encontrado")
    
    # Verificar pastas necessárias
    required_dirs = ['data/uploads', 'data/cache', 'config', 'agents', 'tools', 'utils']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 Pasta criada: {dir_path}")
        else:
            print(f"✅ Pasta existe: {dir_path}")
    
    return True


def test_imports():
    """Testa as importações dos módulos"""
    print("\n📦 Testando importações...")
    
    test_modules = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'langchain',
        'langchain_google_genai',
        'pydantic_settings',
        'scipy',
        'sklearn'
    ]
    
    failed_imports = []
    
    for module in test_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Falha na importação de: {', '.join(failed_imports)}")
        print("Execute: conda env update -f environment.yml")
        return False
    
    return True


def test_local_imports():
    """Testa as importações dos módulos locais"""
    print("\n🏠 Testando módulos locais...")
    
    try:
        from config.settings import settings
        print("✅ config.settings")
        
        # Verificar se as configurações essenciais estão presentes
        if not settings.google_api_key:
            print("⚠️ GOOGLE_API_KEY não configurada no .env")
        else:
            print("✅ GOOGLE_API_KEY configurada")
        
    except Exception as e:
        print(f"❌ config.settings: {e}")
        return False
    
    try:
        from utils.data_loader import DataLoader
        from utils.memory_manager import MemoryManager
        from utils.visualization_helpers import VisualizationHelpers
        print("✅ utils modules")
    except Exception as e:
        print(f"❌ utils modules: {e}")
        return False
    
    try:
        from tools.data_analysis_tools import describe_dataset
        from tools.visualization_tools import create_histogram
        from tools.statistical_tools import calculate_correlation_analysis
        from tools.outlier_detection_tools import detect_outliers_iqr
        print("✅ tools modules")
    except Exception as e:
        print(f"❌ tools modules: {e}")
        return False
    
    try:
        from agents.eda_agent import EDAAgent
        print("✅ agents.eda_agent")
    except Exception as e:
        print(f"❌ agents.eda_agent: {e}")
        return False
    
    return True


def create_test_data():
    """Cria dados de teste se não existirem"""
    print("\n📊 Verificando dados de teste...")
    
    test_files = [
        'tests/test_data/vendas_exemplo.csv',
        'tests/test_data/clientes_exemplo.csv',
        'tests/test_data/vendas_pequeno.csv'
    ]
    
    missing_files = [f for f in test_files if not os.path.exists(f)]
    
    if missing_files:
        print("🔧 Criando dados de teste...")
        try:
            from tests.create_test_data import save_test_datasets
            save_test_datasets()
            print("✅ Dados de teste criados")
        except Exception as e:
            print(f"❌ Erro ao criar dados de teste: {e}")
            return False
    else:
        print("✅ Dados de teste já existem")
    
    return True


def test_eda_agent():
    """Testa o agente EDA com dados de exemplo"""
    print("\n🤖 Testando agente EDA...")
    
    try:
        from agents.eda_agent import EDAAgent
        from utils.data_loader import DataLoader
        
        # Carregar dados de teste
        data_loader = DataLoader()
        df = data_loader.load_csv('tests/test_data/vendas_pequeno.csv')
        
        if df is None:
            print("❌ Erro ao carregar dados de teste")
            return False
        
        print(f"✅ Dados carregados: {df.shape}")
        
        # Inicializar agente
        agent = EDAAgent(memory_key="test_session")
        
        # Carregar dataset no agente
        load_info = agent.load_dataset(df, "vendas_pequeno.csv")
        
        if not load_info.get('load_success'):
            print(f"❌ Erro ao carregar dataset: {load_info.get('error')}")
            return False
        
        print("✅ Dataset carregado no agente")
        
        # Teste simples de análise
        print("🔍 Testando análise básica...")
        response = agent.analyze("Descreva brevemente este dataset")
        
        if "erro" in response.lower() or "error" in response.lower():
            print(f"❌ Erro na análise: {response[:200]}...")
            return False
        
        print("✅ Análise executada com sucesso")
        print(f"📝 Resposta (preview): {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste do agente: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def run_streamlit_app():
    """Executa a aplicação Streamlit"""
    print("\n🚀 Iniciando aplicação Streamlit...")
    print("📱 A aplicação será aberta no navegador")
    print("⚠️ Use Ctrl+C para parar o servidor")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.address', 'localhost',
            '--server.port', '8501',
            '--server.headless', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Aplicação encerrada pelo usuário")
    except Exception as e:
        print(f"❌ Erro ao executar Streamlit: {e}")


def main():
    """Função principal de inicialização"""
    print("=" * 60)
    print("🚀 EDA AGENTES LANGCHAIN - INICIALIZAÇÃO")
    print("=" * 60)
    
    # Verificações do ambiente
    if not check_environment():
        print("\n❌ Falha na verificação do ambiente")
        return False
    
    if not test_imports():
        print("\n❌ Falha nas importações")
        return False
    
    if not test_local_imports():
        print("\n❌ Falha nos módulos locais")
        return False
    
    if not create_test_data():
        print("\n❌ Falha na criação dos dados de teste")
        return False
    
    # Teste opcional do agente (pode ser demorado)
    test_agent = input("\n🤖 Testar agente EDA? (pode demorar) [y/N]: ").lower()
    if test_agent in ['y', 'yes', 's', 'sim']:
        if not test_eda_agent():
            print("\n⚠️ Teste do agente falhou, mas continuando...")
    
    print("\n" + "=" * 60)
    print("✅ SISTEMA PRONTO!")
    print("=" * 60)
    
    print("""
📋 PRÓXIMOS PASSOS:
1. Configure sua GOOGLE_API_KEY no arquivo .env
2. Execute: streamlit run app.py
3. Carregue um arquivo CSV
4. Converse com o agente EDA!

📊 DATASETS DE TESTE DISPONÍVEIS:
- tests/test_data/vendas_exemplo.csv (1000 linhas)
- tests/test_data/clientes_exemplo.csv (500 linhas)
- tests/test_data/vendas_pequeno.csv (100 linhas)

🔧 COMANDOS ÚTEIS:
- streamlit run app.py --server.port 8502  (porta alternativa)
- python tests/create_test_data.py         (recriar dados teste)
    """)
    
    # Oferecer para executar Streamlit automaticamente
    run_app = input("🚀 Executar aplicação Streamlit agora? [Y/n]: ").lower()
    if run_app not in ['n', 'no', 'não']:
        run_streamlit_app()
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Inicialização falhou")
        sys.exit(1)