import random
from app.chat.redis import client

def get_random_component_by_score(component_registry, component_type: str) -> str:
    #  make sure component_type is one of 'llm', 'retriever', 'memory'
    if component_type not in ['llm', 'retriever', 'memory']:
        raise ValueError("Invalid component_type. Must be one of 'llm', 'retriever', 'memory'.")

    # from redis, get the hash containing the sum total scores for the component type
    values = client.hgetall(f"{component_type}_score_values")

    # from redis, get the hash containing the number of times each component has been scored
    counts = client.hgetall(f"{component_type}_score_counts")

    # get all the valid component names for the component_registry
    valid_names = [name for name in component_registry.keys() if name in values and name in counts]

    # loop over those valid names and use them to calculate average scores add average score to a dictionary
    average_scores = {}
    for name in valid_names:
        score = int(values.get(name, 1))
        count = int(counts.get(name, 1))
        average_score = score / count if count > 0.1 else 0.1
        average_scores[name] = average_score

    #if there is no avarge score choose randomly from the component registry
    if not average_scores:
        return random.choice(list(component_registry.keys()))

    # this implements weighted random selection based on average scores
    """
    Example: If average_scores = {"gpt-3.5": 3.5, "gpt-4o": 8.5}, then sum_scores = 12.0
    
    The selection ranges would be:
    [----gpt-3.5----][----------gpt-4o----------]
     0              3.5                        12.0
                             ^ random_value (7.2)

    First iteration: cumulative = 3.5. Is 7.2 <= 3.5? No, continue
    Second iteration: cumulative = 12.0. Is 7.2 <= 12.0? Yes, return "gpt-4o"
    Result: gpt-4o has 8.5/12 = ~71% chance of being selected, gpt-3.5 has 3.5/12 = ~29%
    """
    sum_scores = sum(average_scores.values())
    random_value = random.uniform(0, sum_scores)
    cumulative = 0.0
    for name, avg_score in average_scores.items():
        cumulative += avg_score
        
        if random_value <= cumulative:
            return name

def score_conversation(
    conversation_id: str, 
    score: float, 
    llm: str, 
    retriever: str, 
    memory: str
) -> None:
    """
    This function interfaces with langfuse to assign a score to a conversation, specified by its ID.
    It creates a new langfuse score utilizing the provided llm, retriever, and memory components.
    The details are encapsulated in JSON format and submitted along with the conversation_id and the score.

    :param conversation_id: The unique identifier for the conversation to be scored.
    :param score: The score assigned to the conversation.
    :param llm: The Language Model component information.
    :param retriever: The Retriever component information.
    :param memory: The Memory component information.

    Example Usage:

    score_conversation('abc123', 0.75, 'llm_info', 'retriever_info', 'memory_info')
    """
    score = min(max(score, 0), 1)  # Ensure score is between 0 and 1

    client.hincrby("llm_score_values", llm, score)
    client.hincrby("llm_score_counts", llm, 1)

    client.hincrby("retriever_score_values", retriever, score)
    client.hincrby("retriever_score_counts", retriever, 1)

    client.hincrby("memory_score_values", memory, score)
    client.hincrby("memory_score_counts", memory, 1)


def get_scores():
    """
    Retrieves and organizes scores from the langfuse client for different component types and names.
    The scores are categorized and aggregated in a nested dictionary format where the outer key represents
    the component type and the inner key represents the component name, with each score listed in an array.

    The function accesses the langfuse client's score endpoint to obtain scores.
    If the score name cannot be parsed into JSON, it is skipped.

    :return: A dictionary organized by component type and name, containing arrays of scores.

    Example:

        {
            'llm': {
                'chatopenai-3.5-turbo': [score1, score2],
                'chatopenai-4': [score3, score4]
            },
            'retriever': { 'pinecone_store': [score5, score6] },
            'memory': { 'persist_memory': [score7, score8] }
        }
    """
    aggregate = {"llm": {}, "retriever": {}, "memory": {}}

    for component_type in aggregate.keys():
        scores = client.hgetall(f"{component_type}_score_values")
        counts = client.hgetall(f"{component_type}_score_counts")

        for name in scores.keys():
            score = int(scores.get(name, 1))
            count = int(counts.get(name, 1))
            avg = score / count if count > 0.1 else 0.1
            aggregate[component_type].setdefault(name, []).append(avg)

    return aggregate