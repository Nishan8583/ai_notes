# Understanding LLMs, AI Agents, Harnesses, and MCP

## 1. Large Language Models (LLMs)

An LLM is fundamentally a **next-token prediction model**.

- It receives input (prompt/context).
- Predicts the next token.
- Continues predicting tokens until it generates a response.

An LLM does not inherently execute shell commands, query databases, access APIs, or interact with the operating system.

So how can an AI application execute tools?

The answer is **Tool Calling + Agent Runtime**.

---

## 2. How AI Agents Work

An AI agent is a software system that uses an LLM to pursue a goal, potentially by interacting with external tools.

The LLM handles decision-making and generates proposed actions, while the surrounding application executes those actions.

### Agent Execution Flow

1. User provides a task.
2. Agent sends the user's request, context, and available tool definitions to the LLM.
3. LLM generates either:
   - A normal response.
   - A structured tool-call request.
4. Agent application receives the response.
5. If a tool call is requested, the application executes the corresponding function.
6. Tool result is returned to the LLM as additional context.
7. LLM evaluates the result and decides what to do next.
8. This process repeats until the task is completed.

### Agent Loop

```text
          User Request
                |
                v
          Agent Application
                |
                v
              LLM
                |
                v
       Generate Next Response
                |
          +-----+------+
          |            |
       Answer       Tool Call
          |            |
          v            v
      Return       Tool Dispatcher
      Response          |
                        v
                   Execute Tool
                        |
                        v
                   Tool Result
                        |
                        v
                   Send Result
                    Back to LLM
                        |
                        v
                  Repeat Process
```

**Key concept:**

The LLM generates the tool request. The agent application executes the actual tool.

The LLM does not directly execute the underlying function.

---

## 3. How Does an LLM Request a Tool?

The agent application provides the LLM with available tool definitions.

A tool definition describes:

- Tool name.
- Tool description.
- Required arguments.
- Argument types/schema.

Example tool definition:

```json
{
  "name": "search_process_events",
  "description": "Search endpoint telemetry for a process",
  "parameters": {
    "process_name": "string"
  }
}
```

Suppose the user asks:

> Find events involving powershell.exe.

The LLM might generate a structured tool call:

```json
{
  "tool": "search_process_events",
  "arguments": {
    "process_name": "powershell.exe"
  }
}
```

The agent application:

1. Identifies the requested tool.
2. Extracts its arguments.
3. Executes the corresponding function.
4. Collects the result.
5. Sends the result back to the LLM.

The LLM then generates its next response based on the returned information.

**Important:** Tool calling is a capability supported by the model and its API interface. The application implements the actual execution mechanism.

---

## 4. What Is an AI Agent?

An AI agent is a system that combines an LLM, tools, and an execution mechanism to accomplish a task.

Unlike a traditional fixed workflow, an agent can dynamically decide what action to take next based on previous results.

### Example: Threat-Hunting Agent

User request:

> Investigate powershell.exe with PID 4820.

Possible execution:

1. LLM decides to retrieve process details.
2. Agent executes `get_process_details(4820)`.
3. Tool returns:

```text
Image: powershell.exe
Parent: winword.exe
CommandLine: powershell.exe -enc ...
```

4. LLM identifies potentially suspicious behavior.
5. LLM requests another tool to decode the PowerShell command.
6. Agent executes the requested function.
7. Results return to the LLM.
8. LLM summarizes the investigation.

The LLM does not need to know the investigation results beforehand. It obtains information through tools and uses that information to decide subsequent actions.

### Agent Mental Model

```text
Decide -> Act -> Observe -> Decide Again
```

---

## 5. What Is an Agent Harness?

A harness is the infrastructure/runtime responsible for running and controlling the agent.

It provides the mechanisms that allow the LLM and tools to work together.

### Typical Harness Responsibilities

| Responsibility | Description |
|---|---|
| Model communication | Sends prompts and receives model responses. |
| Tool registration | Provides available tool definitions to the model. |
| Tool dispatching | Maps tool requests to actual functions. |
| Tool execution | Executes functions or communicates with MCP servers. |
| Context management | Maintains conversation history and tool results. |
| Error handling | Handles failed tool calls, timeouts, and retries. |
| Safety controls | Enforces permissions, execution limits, and validation. |
| Agent loop | Repeatedly calls the model and executes requested tools. |

### Agent vs Harness

| AI Agent | Harness |
|---|---|
| Pursues a task or goal. | Provides the runtime to execute the agent. |
| Uses the LLM to determine the next action. | Sends requests to the LLM. |
| Decides which tool would be useful. | Dispatches the requested tool. |
| Evaluates tool results. | Passes results back to the LLM. |
| Can dynamically plan an investigation. | Manages execution, context, and safety. |

These responsibilities are conceptual, not strict boundaries. In many implementations, the agent and harness are part of the same application.

**Important:** A harness does not necessarily imply an autonomous agent. It can also run simple LLM workflows.

---

## 6. What Is MCP (Model Context Protocol)?

MCP is a standardized protocol for connecting AI applications to external tools and resources.

It provides a common communication interface between an application (MCP client) and external tool providers (MCP servers).

Without MCP, developers generally implement their own integrations and interfaces for external tools.

With MCP, compatible applications can discover and invoke tools exposed by MCP servers through a standardized protocol.

### MCP Architecture

```text
             User
              |
              v
       Agent Application
              |
              v
             LLM
              |
              v
       Structured Tool Call
              |
              v
       Agent Application
              |
              v
          MCP Client
              |
              v
         MCP Server
              |
              v
     Underlying Tool / API
              |
              v
        Tool Execution
              |
              v
          MCP Server
              |
              v
          MCP Client
              |
              v
       Agent Application
              |
              v
             LLM
```

### What MCP Provides

| Feature | Description |
|---|---|
| Tool discovery | Discover tools exposed by an MCP server. |
| Tool definitions | Retrieve tool names, descriptions, and input schemas. |
| Tool invocation | Request execution through a standardized protocol. |
| Structured results | Return tool execution results to the client. |
| Error responses | Communicate execution errors. |

MCP also supports other capabilities, including resources and prompts.

---

## 7. Does an AI Agent Require MCP?

No.

An agent can execute tools directly through ordinary application code.

MCP is optional.

### Option 1: Without MCP

The agent application directly executes its own functions.

```text
          LLM
           |
           v
     Tool Call Request
           |
           v
     Go Application
           |
           v
     Tool Dispatcher
           |
           v
     Go Function
           |
           v
       SIEM API
```

Example:

```go
switch toolCall.Name {
case "search_process_events":
    result = searchProcessEvents(toolCall.Arguments)

case "get_parent_process":
    result = getParentProcess(toolCall.Arguments)
}
```

The Go application already knows how to execute these functions.

No MCP server is required.

### Option 2: With MCP

The application uses an MCP client to communicate with MCP servers.

```text
          LLM
           |
           v
     Tool Call Request
           |
           v
     Go Application
           |
           v
       MCP Client
           |
           v
       MCP Server
           |
           v
       SIEM API
```

The MCP server executes the requested operation or delegates it to the underlying system.

The result returns through the MCP server and client to the agent application, which passes it back to the LLM.

### Direct Functions vs MCP

| Consideration | Direct Functions | MCP |
|---|---|---|
| Initial implementation | Often simpler | Requires MCP integration |
| Communication structure | Developer-defined | Standardized protocol |
| Tool discovery | Implement yourself | Supported by MCP |
| Reusability | Requires custom integration or shared code | Easier across compatible applications |
| Control | Direct control over implementation | Depends partly on server |
| Complexity | Lower for small projects | Additional client/server components |

### Why Use MCP?

The biggest advantage is **interoperability and reusability**.

For example, imagine someone has already developed an MCP server that exposes database tools.

Instead of implementing a custom integration for every database operation, your agent can connect to that server, discover its available tools, and invoke them through MCP.

The same MCP server can potentially be reused by multiple compatible AI applications.

However, MCP does not automatically make an agent more intelligent or its tool execution safer.

---

## 8. Important: LLM Communication vs MCP Communication

These are two different communication paths.

| Communication | Mechanism |
|---|---|
| LLM ↔ Agent Application | Model provider API and tool-calling format |
| Agent Application ↔ MCP Server | MCP protocol |
| MCP Server ↔ Underlying System | Server-specific implementation/API |

The LLM does not directly communicate with the MCP server.

Instead:

1. LLM generates a tool-call request.
2. Agent application interprets the request.
3. Application maps it to an available MCP tool.
4. MCP client sends the request to the MCP server.
5. MCP server executes the operation.
6. Result returns to the agent application.
7. Application sends the result back to the LLM.

The agent application acts as the coordinator between the LLM and MCP servers.

---

## 9. Putting Everything Together: Go Threat-Hunting Agent

A possible project architecture:

```text
threat-hunter/
|
├── agent/
│   └── investigator.go
|
├── harness/
│   ├── loop.go
│   ├── context.go
│   └── policy.go
|
├── tools/
│   ├── siem.go
│   └── process.go
|
├── mcp/
│   └── client.go
|
└── main.go
```

### Component Responsibilities

| Component | Responsibility |
|---|---|
| LLM (Ollama) | Generates responses and tool-call requests. |
| Agent | Implements task-oriented investigation behavior. |
| Harness | Manages the execution loop, context, and controls. |
| Tools | Implements actual functionality such as SIEM queries. |
| MCP Client | Communicates with external MCP servers. |
| MCP Server | Exposes tools through the MCP protocol. |

This is one possible architecture. Direct tools and MCP tools can coexist in the same agent.

### Recommended Learning Approach

1. Build a basic Go agent using Ollama.
2. Implement a simple agent loop.
3. Implement direct Go functions as tools.
4. Understand how tool definitions and structured tool calls work.
5. Return tool results to the LLM.
6. Add MCP integration.
7. Connect to an existing MCP server and experiment with tool discovery and invocation.

This approach helps understand the underlying agent architecture before introducing MCP.

---

## 10. Key Takeaways

| Concept | Mental Model |
|---|---|
| LLM | Predicts tokens and generates responses or tool-call requests. |
| AI Agent | Uses an LLM and tools to pursue a goal. |
| Harness | Runtime infrastructure that manages the agent's execution. |
| Tool Calling | Mechanism through which the model requests application actions. |
| Tool Dispatcher | Maps model-generated requests to actual functions. |
| MCP | Standardized protocol for application-to-tool communication. |
| MCP Client | Connects an application to MCP servers. |
| MCP Server | Exposes tools and other capabilities through MCP. |

**The most important distinction:**

- The LLM generates the proposed action.
- The agent uses the LLM to pursue a goal.
- The harness manages execution.
- The tool performs the actual work.
- MCP standardizes communication between the application and external tool servers.

### Final Mental Model

```text
LLM = Decision-making component

Agent = Goal-oriented system

Harness = Execution infrastructure

Tools = Actual capabilities

MCP = Standardized tool communication
```

**An AI agent is not necessarily a separate AI model. It is a software system built around an LLM, tools, and an execution loop.**
