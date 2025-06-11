import autogen
import os

gemini_api_key = os.getenv("GEMINI_API_KEY")
config_list = [
    {"model": "gemini-2.0-flash", "api_key": "{gemini_api_key}", "api_type": "google"},
    {
        "model": "llama3:latest",
        "api_type": "ollama",
        "client_host": "http://192.168.0.1:11434",
    },
]

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
    human_input_mode="TERMINATE",
)

llm_config = {"config_list": config_list, "seed": 42}
coder = autogen.AssistantAgent(name="Coder", llm_config=llm_config)
