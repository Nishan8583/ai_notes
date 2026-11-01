# MCP and AI Agents — Notes

## 1. What Is an AI Agent?

An **AI agent** is a system that uses an LLM to interpret a goal, decide which actions to take, use tools, observe results, and continue until it can provide an answer or reach a stopping condition.

Unlike a basic LLM interaction, an agent can interact with external systems through tools.

### Core components

| Component | Purpose |
|---|---|
| LLM | Interprets requests, reasons about evidence, and selects tools. |
| Instructions | Define the agent's role, objectives, and constraints. |
| Tools | Functions that allow the agent to retrieve information or perform actions. |
| Orchestrator | Manages the agent loop, tool execution, and conversation history. |
| State / context | Stores the conversation and tool results needed for subsequent decisions. |
| Guardrails | Restrict which tools and operations are permitted. |

### The agent loop

An agent commonly follows this cycle:

1. **Reason:** Interpret the request and decide what information is needed.
2. **Act:** Request a tool call.
3. **Observe:** Receive the tool's result.
4. **Reason again:** Analyze the result and decide whether another tool is necessary.
5. **Finish:** Return the final answer when enough information is available.

```text
User request
     |
     v
    LLM
     |
     v
Select tool + arguments
     |
     v
Application executes tool
     |
     v
Tool result returned to LLM
     |
     v
Analyze results
     |
     +---- Need more evidence? ---- Yes ----> Select another tool
     |
     No
     |
     v
Final answer
```

The loop is typically controlled by the application, not by the LLM alone.

---

## 2. What Is MCP?

**MCP (Model Context Protocol)** is a standard protocol that allows AI applications to communicate with external tools and data sources through a consistent interface.

Without MCP, an application may need custom integration code for every external system. MCP standardizes how compatible clients discover and invoke tools exposed by MCP servers.

For example, a threat-hunting agent might need access to:

- SIEM queries
- EDR process telemetry
- Security event databases
- Local log files
- Threat intelligence services

An MCP server can expose these capabilities to an AI application.

### MCP architecture

| Component | Responsibility |
|---|---|
| MCP host / AI application | Coordinates the LLM, tools, and conversation. |
| MCP client | Communicates with an MCP server using the MCP protocol. |
| MCP server | Exposes tools and resources and handles incoming requests. |
| Underlying system | The actual SIEM, database, API, filesystem, or other system accessed by the server. |

**Important:** An MCP server is a program, not an LLM. It executes implemented logic when it receives a valid request. It does not independently decide what an investigation should do unless it has separately been designed with such logic.

### MCP server capabilities

MCP defines several capability types:

- **Tools:** Functions the application can invoke, such as searching security events.
- **Resources:** Data or content that can be exposed to the application.
- **Prompts:** Reusable prompt templates that a server can provide.

Tools are particularly relevant to an AI agent because they allow it to retrieve evidence from external systems.

### MCP communication

MCP uses JSON-RPC messages. Common transports include:

- **STDIO:** Communication through standard input and output, often for locally launched servers.
- **Streamable HTTP:** Communication with servers over HTTP.

---

## 3. How Does an LLM Know Which Tool to Use?

The LLM is provided with a description of the available tools and their argument schemas as part of the model interaction.

For example, the application might provide this tool definition:

```json
{
  "type": "function",
  "function": {
    "name": "search_events",
    "description": "Search security events by keyword.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Keyword to search for"
        }
      },
      "required": ["query"]
    }
  }
}
```

Suppose the user asks:

> Search for PowerShell activity.

The LLM can select `search_events` and generate a structured tool call with the argument `query = "PowerShell"`.

Conceptually, the result looks like this:

```json
{
  "name": "search_events",
  "arguments": {
    "query": "PowerShell"
  }
}
```

This is a simplified representation of a tool call, not necessarily the exact response format of every model API.

The important distinction is that **the model requests a tool call; it does not execute the Go function or communicate directly with the underlying security system by itself.**

The application receives the tool call and decides how to execute it.

---

## 4. How Does an LLM Request a Tool Through MCP?

The LLM does not normally send an MCP request directly to an MCP server. The AI application sits between them.

### Complete flow

1. The MCP client connects to an MCP server and discovers its available tools.
2. The AI application makes those tool definitions available to the LLM through the model's supported tool-calling interface.
3. The user asks a question.
4. The LLM selects a tool and generates a tool call.
5. The AI application receives that tool call.
6. The application routes it through the MCP client.
7. The MCP client sends a `tools/call` request to the MCP server.
8. The MCP server executes the corresponding function.
9. The result travels back through the MCP client to the application.
10. The application sends the tool result to the LLM.
11. The LLM analyzes the result and either requests another tool or returns a final answer.

### Example MCP request

The MCP client might send the following JSON-RPC request to the server:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/call",
  "params": {
    "name": "search_events",
    "arguments": {
      "query": "PowerShell"
    }
  }
}
```

Here:

- `jsonrpc`: Specifies the JSON-RPC protocol version.
- `id`: Identifies the request so its response can be correlated.
- `method`: `tools/call` asks the server to execute a tool.
- `params.name`: Identifies the tool to execute.
- `params.arguments`: Supplies the tool's arguments.

The MCP server receives this request, identifies `search_events`, executes its implementation, and returns the result.

**The model's tool call and the MCP request are two separate messages at two different layers.** The application and MCP client bridge them.

---

## 5. How Tool Calling Works Without MCP

In our Go + Ollama threat-hunting agent, we do not currently use MCP.

Instead, the Go application executes the tools directly.

### Current architecture

```text
User
 |
 v
Go application
 |
 v
Ollama API request
(includes conversation + tool definitions)
 |
 v
Local LLM
 |
 v
Structured tool call
 |
 v
Go agent loop
 |
 v
executeTool()
 |
 +--> searchEvents()
 |
 +--> getProcessDetails()
 |
 +--> getNetworkEvents()
 |
 v
Tool result appended to conversation
 |
 v
Ollama API called again
 |
 v
Final investigation report
```

The `executeTool()` function is the dispatch point between model output and executable Go code.

For example:

```go
switch call.Function.Name {
case "search_events":
    // Validate arguments and execute the search.

case "get_process_details":
    // Validate PID and retrieve process events.

case "get_network_events":
    // Validate PID and retrieve network events.

default:
    return "", fmt.Errorf("tool not allowed")
}
```

The model cannot execute arbitrary Go functions simply by naming them. The application must implement and permit each tool.

After executing a tool, the application adds its result to the conversation and calls the model again. This allows the LLM to use the retrieved evidence when deciding what to do next.

---

## 6. Direct Tool Calling vs. MCP

| Feature | Direct tool calling | Tool calling through MCP |
|---|---|---|
| Tool definitions | Defined in the application | Discovered or obtained through MCP servers and made available to the model |
| Tool execution | Application invokes its own functions | Application routes calls through an MCP client to a server |
| Communication | Application-specific function dispatch | Standardized MCP protocol between client and server |
| External integration | Requires application-specific integration code | Compatible servers provide reusable integrations |
| Best starting point | Small projects and learning | Projects integrating multiple external systems or reusable tools |

MCP is **not required** to build an AI agent. An agent can call local Go functions directly, use MCP servers, or combine both approaches.

---

## 7. Threat-Hunting Agent Example

Imagine that the agent receives this request:

> Investigate PID 4820, identify its parent process, and check its network activity.

The agent might follow this sequence:

1. Call `get_process_details` for PID `4820`.
2. Analyze the returned process events and identify the relevant parent PID.
3. Call `get_process_details` for the parent PID if more evidence is needed.
4. Call `get_network_events` for PID `4820`.
5. Correlate the process ancestry with network activity.
6. Produce a report describing the evidence, suspicious indicators, and uncertainty.

For a real threat-hunting system, the tools would query actual telemetry, such as Sysmon, EDR process events, or SIEM data, rather than our current synthetic dataset.

The agent should distinguish suspicious behavior from confirmed malicious activity. For example, encoded PowerShell may warrant investigation, but its presence alone does not prove compromise.

---

## 8. Security Considerations

When building an agent that accesses security telemetry:

- **Allowlist tools:** Execute only explicitly implemented and permitted tools.
- **Validate arguments:** Check types, ranges, and required fields before execution.
- **Use least privilege:** Give tools only the access they need.
- **Prefer read-only tools initially:** Avoid allowing the first agent version to isolate hosts, kill processes, or modify accounts.
- **Treat retrieved data as untrusted:** Event fields and log messages may contain attacker-controlled text or prompt-injection attempts.
- **Limit execution:** Use a maximum number of agent steps, request timeouts, and result-size limits.
- **Preserve evidence:** Keep relevant tool outputs and timestamps available for verification.
- **Require approval for consequential actions:** Do not let an LLM's request alone authorize destructive or high-impact operations.

The LLM can make mistakes or request an unsuitable tool. The application and tool implementations must enforce security independently.

---

## 9. Key Takeaways

1. An **LLM** interprets the request, selects tools, and analyzes results.
2. An **AI agent** combines the LLM with tools, state, instructions, and an orchestration loop.
3. A **tool call** is a structured request generated by the model, not the execution of the underlying function.
4. The **AI application** receives the tool call and decides how to execute it.
5. An **MCP client** communicates with an MCP server using the MCP protocol.
6. An **MCP server** implements and exposes tools; it is not inherently an LLM.
7. **MCP is optional:** Our Go + Ollama agent can call Go functions directly.
8. The agent loop connects everything: request a tool, execute it, return the result to the LLM, and repeat until the investigation is complete.

### Mental model

- **LLM:** Brain — decides what information it needs.
- **Agent application:** Coordinator — manages the reasoning and execution loop.
- **Tool:** Capability — performs a specific operation.
- **MCP client:** Messenger — communicates with an MCP server.
- **MCP server:** Tool provider — executes the requested operation.
- **Underlying system:** Data source or service — provides the actual evidence.
