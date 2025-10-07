"""
Aplicação Streamlit para EDA Agentes LangChain
Interface web conversacional para análise exploratória de dados usando IA
"""

import streamlit as st
import pandas as pd
import logging
import os
from datetime import datetime
from typing import Optional
import traceback

# Configuração da página
st.set_page_config(
    page_title="EDA Agentes LangChain",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Imports locais (após configuração de logging)
try:
    from config.settings import settings
    from agents.eda_agent import EDAAgent
    from utils.data_loader import DataLoader
    from utils.memory_manager import MemoryManager
    from tools.visualization_tools import get_created_visualizations, clear_visualizations
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()


def initialize_session_state():
    """Inicializa variáveis do estado da sessão"""
    if 'eda_agent' not in st.session_state:
        st.session_state.eda_agent = None
    
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    
    if 'dataset_info' not in st.session_state:
        st.session_state.dataset_info = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = []


def display_header():
    """Exibe o cabeçalho da aplicação"""
    st.title("📊 EDA Agentes LangChain")
    st.markdown("""
    ### Análise Exploratória de Dados com Inteligência Artificial
    
    Carregue seu dataset CSV e converse com um agente especialista em análise de dados!
    O agente pode realizar análises estatísticas, criar visualizações, detectar outliers e muito mais.
    """)


def display_sidebar():
    """Exibe a barra lateral com controles"""
    with st.sidebar:
        st.header("🎛️ Controles")
        
        # Seção de carregamento de dados
        st.subheader("📂 Carregamento de Dados")
        
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV",
            type="csv",
            help="Carregue um arquivo CSV para análise (máximo 50MB)",
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Carregar Dataset"):
                load_dataset(uploaded_file)
        
        # Informações do dataset atual
        if st.session_state.dataset_loaded:
            st.subheader("📊 Dataset Atual")
            info = st.session_state.dataset_info
            if info:
                st.metric("Linhas", info['shape'][0])
                st.metric("Colunas", info['shape'][1])
                st.text(f"Tamanho: {info['memory_usage']}")
                
                with st.expander("📋 Colunas do Dataset"):
                    for col in info['columns']:
                        st.text(f"• {col}")
        
        # Controles da conversa
        st.subheader("💬 Conversa")
        
        if st.button("🧹 Limpar Conversa"):
            clear_conversation()
        
        if st.button("💾 Exportar Conversa"):
            export_conversation()
        
        # Configurações
        st.subheader("⚙️ Configurações")
        
        st.info(f"""
        **Modelo**: {settings.google_model}
        **Sessão**: {st.session_state.session_id[:12]}...
        **Ferramentas**: 21 disponíveis
        """)
        
        # Links úteis
        st.subheader("🔗 Links Úteis")
        st.markdown("""
        - [Documentação LangChain](https://langchain.readthedocs.io/)
        - [Google Gemini Pro](https://ai.google.dev/)
        - [Streamlit Docs](https://docs.streamlit.io/)
        """)


def load_dataset(uploaded_file):
    """Carrega o dataset enviado pelo usuário"""
    try:
        with st.spinner("🔄 Carregando dataset..."):
            # Salvar arquivo temporariamente
            temp_path = f"data/uploads/{uploaded_file.name}"
            os.makedirs("data/uploads", exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Carregar dados
            data_loader = DataLoader()
            df = data_loader.load_csv(temp_path)
            
            if df is not None:
                # Inicializar agente se necessário
                if st.session_state.eda_agent is None:
                    st.session_state.eda_agent = EDAAgent(
                        memory_key=st.session_state.session_id
                    )
                
                # Carregar dataset no agente
                load_info = st.session_state.eda_agent.load_dataset(
                    df, filename=uploaded_file.name
                )
                
                if load_info.get('load_success'):
                    st.session_state.dataset_loaded = True
                    st.session_state.current_dataset = df
                    st.session_state.dataset_info = load_info
                    
                    st.success(f"✅ Dataset carregado com sucesso!")
                    st.balloons()
                    
                    # Adicionar sugestões iniciais ao chat
                    suggestions = st.session_state.eda_agent.suggest_initial_analysis()
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": suggestions,
                        "timestamp": datetime.now()
                    })
                    
                    logger.info(f"Dataset carregado: {uploaded_file.name}")
                else:
                    st.error(f"❌ Erro ao carregar dataset: {load_info.get('error')}")
            else:
                st.error("❌ Erro ao processar o arquivo CSV")
    
    except Exception as e:
        st.error(f"❌ Erro inesperado: {str(e)}")
        logger.error(f"Erro ao carregar dataset: {e}")
        logger.error(traceback.format_exc())


def display_chat_interface():
    """Exibe a interface de chat"""
    if not st.session_state.dataset_loaded:
        st.info("👆 Carregue um dataset CSV na barra lateral para começar a análise!")
        return
    
    # Container para o histórico de chat
    chat_container = st.container()
    
    with chat_container:
        # Exibir histórico de mensagens
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Exibir visualizações se existirem no histórico
                if message.get("visualizations"):
                    st.markdown("---")
                    st.markdown("### 📊 Visualizações")
                    for j, viz in enumerate(message["visualizations"]):
                        try:
                            if viz.get('title'):
                                st.markdown(f"**{viz['title']}**")
                            if viz.get('description'):
                                st.markdown(viz['description'])
                            
                            # Reexibir o gráfico se a figura foi salva
                            if viz.get('figure'):
                                # Usar key único baseado no índice da mensagem e visualização
                                msg_index = st.session_state.chat_history.index(message)
                                chart_key = f"history_chart_{msg_index}_{j}"
                                st.plotly_chart(viz['figure'], use_container_width=True, key=chart_key)
                            else:
                                st.info(f"📊 Gráfico {viz.get('type', 'desconhecido')} foi gerado nesta conversa")
                        except Exception as viz_error:
                            st.warning(f"⚠️ Não foi possível reexibir o gráfico: {viz.get('title', 'sem título')}")
                            logger.error(f"Erro ao reexibir visualização: {viz_error}")
    
    # Input para nova mensagem
    if prompt := st.chat_input("Faça uma pergunta sobre seus dados..."):
        # Adicionar mensagem do usuário
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        })
        
        # Exibir mensagem do usuário
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Processar resposta do agente
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analisando..."):
                try:
                    # Limpar visualizações anteriores
                    clear_visualizations()
                    
                    # Processar resposta do agente
                    response = st.session_state.eda_agent.analyze(prompt)
                    
                    # Exibir resposta
                    st.markdown(response)
                    
                    # Verificar se foram criadas visualizações
                    new_visualizations = get_created_visualizations()
                    message_visualizations = []
                    
                    if new_visualizations:
                        st.markdown("---")
                        st.markdown("### 📊 Visualizações Geradas")
                        
                        for i, viz in enumerate(new_visualizations):
                            try:
                                # Exibir título da visualização
                                if viz.get('title'):
                                    st.markdown(f"**{viz['title']}**")
                                
                                # Exibir descrição se existir
                                if viz.get('description'):
                                    st.markdown(viz['description'])
                                
                                # Exibir o gráfico
                                if viz.get('figure'):
                                    # Usar key único para cada gráfico para evitar conflitos
                                    chart_key = f"chart_{len(st.session_state.chat_history)}_{i}"
                                    st.plotly_chart(viz['figure'], use_container_width=True, key=chart_key)
                                    
                                    # Armazenar visualização completa para o histórico
                                    message_visualizations.append({
                                        'type': viz['type'],
                                        'title': viz.get('title', ''),
                                        'description': viz.get('description', ''),
                                        'figure': viz['figure']  # Armazenar a figura também
                                    })
                                
                            except Exception as viz_error:
                                st.error(f"Erro ao exibir gráfico: {str(viz_error)}")
                                logger.error(f"Erro na visualização: {viz_error}")
                    
                    # Adicionar ao histórico com visualizações
                    message_data = {
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now()
                    }
                    
                    if message_visualizations:
                        message_data["visualizations"] = message_visualizations
                    
                    st.session_state.chat_history.append(message_data)
                    
                except Exception as e:
                    error_msg = f"❌ Erro na análise: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Erro na análise: {e}")
                    logger.error(traceback.format_exc())


def clear_conversation():
    """Limpa o histórico da conversa"""
    st.session_state.chat_history = []
    st.session_state.visualizations = []
    clear_visualizations()  # Limpar visualizações das ferramentas
    if st.session_state.eda_agent:
        st.session_state.eda_agent.clear_memory()
    st.success("🧹 Conversa limpa!")
    st.rerun()


def export_conversation():
    """Exporta a conversa atual"""
    try:
        if not st.session_state.chat_history:
            st.warning("⚠️ Nenhuma conversa para exportar")
            return
        
        # Preparar dados para exportação
        export_data = []
        for msg in st.session_state.chat_history:
            export_data.append({
                "timestamp": msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Criar DataFrame
        df_export = pd.DataFrame(export_data)
        
        # Converter para CSV
        csv = df_export.to_csv(index=False)
        
        # Botão de download
        st.download_button(
            label="📥 Baixar Conversa",
            data=csv,
            file_name=f"conversa_eda_{st.session_state.session_id}.csv",
            mime="text/csv"
        )
        
        st.success("💾 Conversa pronta para download!")
        
    except Exception as e:
        st.error(f"❌ Erro ao exportar: {str(e)}")
        logger.error(f"Erro na exportação: {e}")


def display_footer():
    """Exibe o rodapé da aplicação"""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🤖 Powered by:**")
        st.markdown("- LangChain")
        st.markdown("- Google Gemini Pro")
        st.markdown("- Streamlit")
    
    with col2:
        st.markdown("**📊 Capacidades:**")
        st.markdown("- Análise Estatística")
        st.markdown("- Visualizações Interativas") 
        st.markdown("- Detecção de Outliers")
    
    with col3:
        st.markdown("**💡 Dicas:**")
        st.markdown("- Seja específico nas perguntas")
        st.markdown("- Explore visualizações")
        st.markdown("- Peça sugestões de análise")


def main():
    """Função principal da aplicação"""
    try:
        # Verificar configurações
        if not settings.google_api_key:
            st.error("❌ Configure sua chave da API do Google Gemini no arquivo .env")
            st.stop()
        
        # Inicializar estado da sessão
        initialize_session_state()
        
        # Layout da aplicação
        display_header()
        display_sidebar()
        display_chat_interface()
        display_footer()
        
    except Exception as e:
        st.error(f"❌ Erro na aplicação: {str(e)}")
        logger.error(f"Erro na aplicação principal: {e}")
        logger.error(traceback.format_exc())
        
        # Mostrar detalhes do erro em desenvolvimento
        if st.checkbox("🔍 Mostrar detalhes do erro (desenvolvimento)"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()