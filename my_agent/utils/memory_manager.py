from mem0 import MemoryClient

class MemoryManager:
    def __init__(self, api_key):
        self.client = MemoryClient(api_key=api_key)

    def add_to_memory(self, messages, user_id):
        self.client.add(messages, user_id=user_id)

    def get_relevant_context(self, query, user_id):
        return self.client.search(query, user_id=user_id)

    def get_user_memories(self, user_id):
        return self.client.get_all(user_id=user_id)