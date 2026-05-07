class ConversationMemory:
    def __init__(self):
        self._store = {}
        
    def save_context(self, session_id: str, content: str, source_type: str):
        self._store[session_id] = {
            "content": content,
            "type": source_type
        }
        
    def get_context(self, session_id: str) -> dict | None:
        # print("getting mem context for:", session_id)
        return self._store.get(session_id)
        
    def clear_context(self, session_id: str):
        if session_id in self._store:
            del self._store[session_id]

# Singleton instance
memory_manager = ConversationMemory()
