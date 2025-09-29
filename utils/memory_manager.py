"""
Gerenciador de memória para agentes LangChain
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from config.settings import get_settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """Classe para gerenciar a memória conversacional dos agentes"""
    
    def __init__(self, session_id: str = "default", window_size: int = None):
        self.settings = get_settings()
        self.session_id = session_id
        self.window_size = window_size or self.settings.agent_memory_window_size
        
        # Inicializar memória
        self.memory = ConversationBufferWindowMemory(
            k=self.window_size,
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        
        # Pasta para salvar histórico de conversas
        self.memory_folder = self.settings.cache_folder_path / "memory"
        self.memory_folder.mkdir(parents=True, exist_ok=True)
        
        # Arquivo para esta sessão
        self.memory_file = self.memory_folder / f"session_{session_id}.json"
        
        # Carregar memória existente se houver
        self.load_memory()
    
    def add_user_message(self, message: str, metadata: Dict[str, Any] = None) -> None:
        """Adiciona uma mensagem do usuário à memória"""
        try:
            self.memory.chat_memory.add_user_message(message)
            
            # Salvar metadados se fornecidos
            if metadata:
                self._save_metadata(message, metadata, "user")
            
            logger.debug(f"Mensagem do usuário adicionada: {message[:100]}...")
            
        except Exception as e:
            logger.error(f"Erro ao adicionar mensagem do usuário: {e}")
    
    def add_ai_message(self, message: str, metadata: Dict[str, Any] = None) -> None:
        """Adiciona uma mensagem da IA à memória"""
        try:
            self.memory.chat_memory.add_ai_message(message)
            
            # Salvar metadados se fornecidos
            if metadata:
                self._save_metadata(message, metadata, "ai")
            
            logger.debug(f"Mensagem da IA adicionada: {message[:100]}...")
            
        except Exception as e:
            logger.error(f"Erro ao adicionar mensagem da IA: {e}")
    
    def get_chat_history(self) -> List[BaseMessage]:
        """Retorna o histórico de chat atual"""
        try:
            return self.memory.chat_memory.messages
        except Exception as e:
            logger.error(f"Erro ao obter histórico de chat: {e}")
            return []
    
    def get_formatted_history(self) -> List[Dict[str, str]]:
        """Retorna o histórico formatado para exibição"""
        try:
            formatted_history = []
            messages = self.get_chat_history()
            
            for message in messages:
                if isinstance(message, HumanMessage):
                    formatted_history.append({
                        "role": "user",
                        "content": message.content,
                        "timestamp": getattr(message, 'timestamp', None)
                    })
                elif isinstance(message, AIMessage):
                    formatted_history.append({
                        "role": "assistant",
                        "content": message.content,
                        "timestamp": getattr(message, 'timestamp', None)
                    })
            
            return formatted_history
            
        except Exception as e:
            logger.error(f"Erro ao formatar histórico: {e}")
            return []
    
    def clear_memory(self) -> None:
        """Limpa toda a memória da sessão"""
        try:
            self.memory.clear()
            
            # Remover arquivo de memória se existir
            if self.memory_file.exists():
                self.memory_file.unlink()
            
            logger.info(f"Memória da sessão {self.session_id} limpa")
            
        except Exception as e:
            logger.error(f"Erro ao limpar memória: {e}")
    
    def save_memory(self) -> None:
        """Salva a memória atual em arquivo"""
        try:
            history = self.get_formatted_history()
            
            memory_data = {
                "session_id": self.session_id,
                "window_size": self.window_size,
                "timestamp": datetime.now().isoformat(),
                "chat_history": history
            }
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Memória salva em {self.memory_file}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar memória: {e}")
    
    def load_memory(self) -> None:
        """Carrega memória de arquivo se existir"""
        try:
            if not self.memory_file.exists():
                logger.debug(f"Arquivo de memória não encontrado: {self.memory_file}")
                return
            
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # Restaurar histórico de chat
            chat_history = memory_data.get("chat_history", [])
            
            for entry in chat_history:
                role = entry.get("role")
                content = entry.get("content", "")
                
                if role == "user":
                    self.memory.chat_memory.add_user_message(content)
                elif role == "assistant":
                    self.memory.chat_memory.add_ai_message(content)
            
            logger.info(f"Memória carregada: {len(chat_history)} mensagens")
            
        except Exception as e:
            logger.error(f"Erro ao carregar memória: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da memória atual"""
        try:
            messages = self.get_chat_history()
            
            stats = {
                "total_messages": len(messages),
                "user_messages": sum(1 for m in messages if isinstance(m, HumanMessage)),
                "ai_messages": sum(1 for m in messages if isinstance(m, AIMessage)),
                "window_size": self.window_size,
                "session_id": self.session_id,
                "memory_file_exists": self.memory_file.exists()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas da memória: {e}")
            return {}
    
    def _save_metadata(self, message: str, metadata: Dict[str, Any], role: str) -> None:
        """Salva metadados da mensagem"""
        try:
            metadata_file = self.memory_folder / f"metadata_{self.session_id}.json"
            
            # Carregar metadados existentes
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    all_metadata = json.load(f)
            else:
                all_metadata = []
            
            # Adicionar novos metadados
            metadata_entry = {
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "message_preview": message[:100],
                "metadata": metadata
            }
            
            all_metadata.append(metadata_entry)
            
            # Salvar
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(all_metadata, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error(f"Erro ao salvar metadados: {e}")
    
    def export_conversation(self, format: str = "json") -> Optional[str]:
        """
        Exporta a conversa atual em diferentes formatos
        
        Args:
            format: Formato de exportação ("json" ou "txt")
            
        Returns:
            Caminho para o arquivo exportado ou None se houver erro
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == "json":
                export_file = self.memory_folder / f"export_{self.session_id}_{timestamp}.json"
                
                export_data = {
                    "session_id": self.session_id,
                    "export_timestamp": datetime.now().isoformat(),
                    "chat_history": self.get_formatted_history(),
                    "memory_stats": self.get_memory_stats()
                }
                
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            elif format.lower() == "txt":
                export_file = self.memory_folder / f"export_{self.session_id}_{timestamp}.txt"
                
                with open(export_file, 'w', encoding='utf-8') as f:
                    f.write(f"Conversa - Sessão: {self.session_id}\n")
                    f.write(f"Exportado em: {datetime.now().isoformat()}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for entry in self.get_formatted_history():
                        role = "Usuário" if entry["role"] == "user" else "Assistente"
                        f.write(f"{role}: {entry['content']}\n")
                        f.write("-" * 30 + "\n")
            
            else:
                raise ValueError(f"Formato não suportado: {format}")
            
            logger.info(f"Conversa exportada para: {export_file}")
            return str(export_file)
            
        except Exception as e:
            logger.error(f"Erro ao exportar conversa: {e}")
            return None