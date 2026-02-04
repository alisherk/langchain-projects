from app.chat.memories.sql_memory import sql_memory_registry
from app.chat.memories.window_memory import window_memory_registry

# Unified memory registry combining all memory types
memory_registry = {
    **sql_memory_registry,
    **window_memory_registry,
}
