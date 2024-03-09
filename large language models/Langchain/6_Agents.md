# Agents
- Agents can use multiple sources of information and chain them to do stuffs.

```python
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model=llm_model)

tools = load_tools(["llm-math","wikipedia"], llm=llm)  # 2 sources. llm-math is built in calculator, wikipdia searches for answers in wikepedia


agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)  # AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION is optimized for chat applications

agent("What is the the diameter of earth?")
```
- output, it already knows that it should look at wikepedia, and then uses calculator thats cool.
```bash

    > Entering new AgentExecutor chain...
    Thought: I should use Wikipedia to find the diameter of the Earth.
    Action:
    ```
    {
      "action": "Wikipedia",
      "action_input": "Diameter of Earth"
    }
    ```

    Observation: Page: Earth radius
    Summary: {summary from wikipedia here}

<!---->

    Thought:The diameter of the Earth is approximately twice its equatorial radius, which is 6,378 km according to the Wikipedia page. So the diameter of the Earth is approximately 12,756 km.
    Action:
    ```
    {
      "action": "Calculator",
      "action_input": "2*6378"
    }
    ```

    Observation: Answer: 12756
    Thought:The final answer is 12,756 km. 
    Final Answer: 12,756 km.

    > Finished chain.

<!---->

    {'input': 'What is the the diameter of earth?', 'output': '12,756 km.'}

```

- for simple only uses calclator
```bash

    > Entering new AgentExecutor chain...
    Thought: I need to use a calculator to find the answer to this math problem.

    Action:
    ```
    {
      "action": "Calculator",
      "action_input": "0.25*75"
    }
    ```


    Observation: Answer: 18.75
    Thought:I have successfully calculated the answer to the math problem.

    Final Answer: 18.75

    > Finished chain.

<!---->

    {'input': 'What is the 25% of 75', 'output': '18.75'}

```

## Python agent example
```python
agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)
customer_list = [["Harrison", "Chase"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
# for detailed outputs
import langchain
langchain.debug=True
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
langchain.debug=False
```
- Output is gonna be huge, but one thing that helped me understand it better is this prompt
```bash

    {
      "prompts": [
        "Human: You are an agent designed to write and execute python code to answer questions.\nYou have access to a python REPL, which you can use to execute python code.\nIf you get an error, debug your code and try again.\nOnly use the output of your code to answer the question. \nYou might know the answer without running any code, but you should still run the code to get the answer.\nIf it does not seem like you can write code to answer the question, just return \"I don't know\" as the answer.\n\n\nPython REPL: A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [Python REPL]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin!\n\nQuestion: Sort these customers by last name and then first name and print the output: [['Harrison', 'Chase'], ['Lang', 'Chain'], ['Dolly', 'Too'], ['Elle', 'Elem'], ['Geoff', 'Fusion'], ['Trance', 'Former'], ['Jen', 'Ayai']]\nThought:"
      ]
    }

```

## Custom tool
```python
from langchain.agents import tool
from datetime import date
@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())
agent= initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
try:
    result = agent("whats the date today?") 
except: 
    print("exception on external access")
```
```bash

    > Entering new AgentExecutor chain...
    Thought: I need to use the `time` tool to get today's date.
    Action:
    ```
    {
      "action": "time",
      "action_input": ""
    }
    ```

    Observation: 2024-03-08
    Thought:I have successfully retrieved today's date using the `time` tool.
    Final Answer: Today's date is 2024-03-08.

    > Finished chain.

```