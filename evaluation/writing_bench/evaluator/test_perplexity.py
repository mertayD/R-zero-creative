from llm import ClaudeAgent

agent = ClaudeAgent(system_prompt="Respond all the messages as Cartman from south park.")

print(agent.run("what is my name?"))