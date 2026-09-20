class ConversationMemory:
    """
    Stores recent conversation history for the RAG system.
    """

    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages = []

    def add_user_message(self, message: str):
        self.messages.append({
            "role": "user",
            "content": message,
        })
        self._trim()

    def add_assistant_message(self, message: str):
        self.messages.append({
            "role": "assistant",
            "content": message,
        })
        self._trim()

    def get_history(self) -> list[dict]:
        return self.messages.copy()

    def clear(self):
        self.messages.clear()

    def _trim(self):
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]