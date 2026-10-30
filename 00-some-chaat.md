# Understanding Local LLM Infrastructure: llama.cpp, GGUF, Quantization, GGML & AI Agent Harnesses

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [What Is llama.cpp?](#what-is-llamacpp)
3. [What Is GGUF?](#what-is-gguf)
4. [What Is Quantization?](#what-is-quantization)
5. [What Are Tensors?](#what-are-tensors)
6. [What Is GGML?](#what-is-ggml)
7. [How llama.cpp Performs Inference](#how-llamacpp-performs-inference)
8. [What Is llama-server?](#what-is-llama-server)
9. [What Is an AI Agent Harness?](#what-is-an-ai-agent-harness)
10. [Where Does MCP Fit In?](#where-does-mcp-fit-in)
11. [Putting Everything Together](#putting-everything-together)
12. [Building a Local Threat-Hunting Agent](#building-a-local-threat-hunting-agent)
13. [Key Takeaways](#key-takeaways)

---

# The Big Picture

When running an LLM locally, several components work together.

It helps to understand that these components have different responsibilities:

| Component | Responsibility |
|---|---|
| GGUF | Stores the model weights and metadata |
| Quantization | Reduces the precision and memory requirements of model weights |
| Tensors | Represent the numerical data used by the model |
| GGML | Provides tensor operations and computation infrastructure |
| llama.cpp | Loads the model and performs inference |
| llama-server | Exposes the inference engine through an HTTP API |
| Agent harness | Coordinates the LLM, conversation, and external tools |
| MCP | Standardizes communication between AI applications and external tools |

A simplified architecture:

```text
                 Go Application
                (Agent Harness)
                       |
                       |
                 HTTP API Request
                       |
                       v
                  llama-server
                       |
                       v
                   llama.cpp
                       |
                       v
                     GGML
                       |
                       v
                 GGUF Model File
```

The agent harness also communicates with external systems:

```text
                  Go Agent Harness
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
          LLM API     Elasticsearch  EDR API
       (llama-server)    (ELK)      (CrowdStrike)
```

The important distinction:

**The LLM generates responses and decides when tools are needed. The agent harness is traditional software that actually executes those tools.**

---

# What Is llama.cpp?

`llama.cpp` is an open-source C/C++ project that provides an inference engine for running LLMs locally.

It handles the actual execution of the model.

Think of it as the runtime responsible for loading a model and performing the computations required to generate text.

## What does it do?

Some of its responsibilities include:

- Loading model files, particularly GGUF models.
- Initializing model architecture and weights.
- Managing memory.
- Tokenizing input text.
- Performing transformer computations.
- Executing tensor operations through GGML.
- Sampling the next token.
- Repeating the generation process until completion.

It supports different hardware backends, including:

- CPU
- NVIDIA CUDA GPUs
- Apple Metal
- AMD GPU backends
- Vulkan

This allows models to run on different hardware configurations.

## Why llama.cpp?

Originally, running LLMs locally often involved complicated dependencies and hardware-specific implementations.

`llama.cpp` provides a relatively lightweight way to run models across different hardware.

It also supports quantized models, which makes it possible to run models that would otherwise require considerably more memory.

## llama.cpp vs Ollama

These projects serve different purposes.

| llama.cpp | Ollama |
|---|---|
| Inference engine and supporting runtime | Higher-level local LLM management and runtime |
| Loads and executes models | Makes downloading and managing models easier |
| Supports GGUF models | Uses llama.cpp infrastructure for many model execution paths |
| Provides lower-level configuration | Provides a convenient interface and API |

A useful mental model:

- `llama.cpp` is closer to the engine.
- Ollama provides a more convenient layer for managing and using models.

You do not necessarily need Ollama if you want to work directly with llama.cpp.

---

# What Is GGUF?

GGUF stands for **GPT-Generated Unified Format**.

It is a binary file format used to store LLM models and the metadata required to load them.

For example:

```text
qwen3-14b-q4_k_m.gguf
```

This file contains the model's numerical weights, along with information describing how those weights should be interpreted.

GGUF is not the model's source code. It is a structured container holding the data needed to load and execute the model.

## What does a GGUF file contain?

Conceptually, a GGUF file contains three major sections:

```text
+----------------------------------+
|            GGUF Header           |
+----------------------------------+
|            Metadata              |
+----------------------------------+
|         Tensor Information       |
+----------------------------------+
|           Tensor Data            |
+----------------------------------+
```

### 1. Header

Contains basic information about the file, including:

- GGUF format version.
- Number of metadata entries.
- Number of tensors.

This helps the loader understand how to parse the file.

### 2. Metadata

Contains information about the model itself.

Examples include:

- Model architecture.
- Model name.
- Number of layers.
- Embedding dimensions.
- Attention head configuration.
- Context length.
- Tokenizer information.
- Special token IDs.
- Quantization-related information.

For example, metadata might describe a model's architecture as Qwen or Llama.

### 3. Tensor Information

Describes the tensors stored inside the file.

For each tensor, information includes:

- Tensor name.
- Tensor dimensions.
- Tensor data type.
- Offset to its actual data.

For example:

```text
token_embd.weight
blk.0.attn_q.weight
blk.0.attn_k.weight
blk.0.attn_v.weight
blk.0.attn_output.weight
```

These are examples of tensor names that may appear in transformer models.

### 4. Tensor Data

This is where the actual numerical weight data is stored.

The tensor information tells the inference engine how to interpret this data.

## Why is GGUF useful?

GGUF provides a standardized format that allows compatible inference engines to load model weights without needing a completely different loading implementation for every model file.

It also supports different tensor data types and quantization formats.

**Important distinction:**

GGUF is a container format. It does not itself perform inference or define a quantization algorithm.

---

# What Is Quantization?

Quantization is a technique used to reduce the precision of numerical values.

In LLMs, it is commonly used to reduce the memory required to store model weights.

This is particularly important when running models on consumer GPUs.

## Why do we need quantization?

Consider a model with 14 billion parameters.

If every parameter uses a 32-bit floating-point value:

```text
32 bits = 4 bytes
```

The approximate weight memory requirement is:

```text
14 billion × 4 bytes = 56 GB
```

That is a lot of memory.

Now consider using 16-bit floating-point values:

```text
16 bits = 2 bytes
```

The approximate requirement becomes:

```text
14 billion × 2 bytes = 28 GB
```

With 4-bit values:

```text
4 bits = 0.5 bytes
```

The theoretical raw weight storage becomes:

```text
14 billion × 0.5 bytes = 7 GB
```

| Precision | Bits per parameter | Approximate storage for 14B parameters |
|---|---:|---:|
| FP32 | 32 | 56 GB |
| FP16 | 16 | 28 GB |
| 8-bit | 8 | 14 GB |
| 4-bit | 4 | 7 GB |

These are theoretical estimates based only on parameter count and bit width.

Actual model files are larger because quantization formats also store scales, metadata, and sometimes higher-precision tensors.

## How does quantization work?

Suppose we have a floating-point value:

```text
0.7321
```

Instead of storing the original value using a high-precision floating-point representation, quantization maps it to a lower-precision representation.

Conceptually:

```text
Original value: 0.7321

Quantized value: 7
```

The quantized value is not useful by itself. The inference engine also needs information such as a scale factor to reconstruct an approximation of the original value.

For example:

```text
Original value ≈ Quantized value × Scale
```

The reconstructed value might be:

```text
0.7000
```

It is not exactly the same as the original, but it is close enough for many practical applications.

This is why quantization can reduce memory consumption while retaining much of the model's useful behavior.

## Common GGUF quantization formats

You may encounter names like:

| Format | General description |
|---|---|
| Q4_0 | Basic 4-bit quantization |
| Q4_K_M | 4-bit K-quant format with mixed precision choices |
| Q5_K_M | 5-bit K-quant format with mixed precision choices |
| Q8_0 | 8-bit quantization |

The exact implementation differs between formats.

For example, K-quant formats use blocks and additional quantization metadata. Some tensors may also use higher precision than others.

You do not need to understand every quantization algorithm to start running local models.

The practical takeaway is:

- Lower bit widths generally mean smaller model files.
- Lower bit widths generally reduce memory requirements.
- More aggressive quantization can introduce more accuracy loss.
- The best choice depends on the model, workload, and available hardware.

## Quantization vs GGUF

These are related but different concepts.

```text
GGUF = Container format

Quantization = Technique for reducing numerical precision
```

A GGUF file can contain quantized tensors.

For example:

```text
qwen3-14b-q4_k_m.gguf
```

The filename suggests that the model is stored using a Q4_K_M quantization format.

## One important memory consideration

The model weights are not the only thing that consumes memory during inference.

Other memory requirements include:

- KV cache.
- Computation buffers.
- Runtime overhead.
- Context length-related allocations.

Therefore, a model with 7 GB of weights does not necessarily require only 7 GB of GPU memory.

---

# What Are Tensors?

A tensor is essentially a multidimensional array of numerical values.

You can think of tensors as the generalization of ordinary variables, arrays, and matrices.

They are fundamental to machine learning because neural networks perform mathematical operations on large collections of numbers.

## Tensor dimensions

A tensor can have different numbers of dimensions.

### 0D tensor: Scalar

A single numerical value.

```text
5
```

### 1D tensor: Vector

A one-dimensional array.

```text
[1, 2, 3, 4]
```

Shape:

```text
[4]
```

### 2D tensor: Matrix

A two-dimensional array.

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
```

Shape:

```text
[2, 3]
```

This represents 2 rows and 3 columns.

### 3D tensor

A collection of matrices.

```text
[
  [
    [1, 2],
    [3, 4]
  ],
  [
    [5, 6],
    [7, 8]
  ]
]
```

Shape:

```text
[2, 2, 2]
```

You can continue adding dimensions as needed.

## How are tensors used in LLMs?

LLMs contain many tensors that represent different parts of the neural network.

Examples include:

| Tensor | Purpose |
|---|---|
| Token embeddings | Represents tokens as numerical vectors |
| Query weights (Q) | Used to calculate attention queries |
| Key weights (K) | Used to calculate attention keys |
| Value weights (V) | Used to calculate attention values |
| Attention output weights | Used to transform attention output |
| Feed-forward weights | Used in the network's feed-forward layers |

A simplified transformer layer might contain:

```text
Input
  |
  v
Attention
  |
  +-- Query weights
  +-- Key weights
  +-- Value weights
  +-- Output weights
  |
  v
Feed Forward Network
  |
  v
Output
```

Each of these operations uses tensors.

## Matrix multiplication

One of the most important operations in neural networks is matrix multiplication.

For example:

```text
Y = X × W
```

Where:

- `X` is an input tensor.
- `W` is a weight tensor.
- `Y` is the resulting output tensor.

The model repeatedly performs operations like this while processing input and generating output.

This is one reason GPUs are useful for LLM inference: they can perform large amounts of numerical computation in parallel.

## How does GGUF relate to tensors?

GGUF stores the tensors that make up the model.

It describes:

- The tensor's name.
- Its dimensions.
- Its data type.
- Where its data is located inside the file.

The inference engine reads this information to load the model correctly.

---

# What Is GGML?

GGML is a tensor computation library associated with the llama.cpp ecosystem.

It provides the infrastructure for representing tensors and performing operations on them.

While GGUF describes how model data is stored, GGML provides the computational building blocks used to work with that data.

## What does GGML provide?

Some of its responsibilities include:

- Tensor structures.
- Tensor memory management.
- Mathematical operations.
- Quantized tensor support.
- Computation graphs.
- Hardware backends.

For example, GGML provides operations for:

- Matrix multiplication.
- Addition.
- Multiplication.
- Normalization.
- Activation functions.
- Attention-related computations.

## The `ggml_tensor` structure

At a conceptual level, a GGML tensor contains information such as:

```c
struct ggml_tensor {
    enum ggml_type type;

    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];

    enum ggml_op op;

    struct ggml_tensor * src[GGML_MAX_SRC];

    void * data;

    char name[GGML_MAX_NAME];
};
```

The actual structure and fields can vary between versions, but this illustrates the important concepts.

| Field | Meaning |
|---|---|
| `type` | Tensor data type |
| `ne` | Number of elements in each dimension |
| `nb` | Memory strides for dimensions |
| `op` | Operation associated with the tensor |
| `src` | Source tensors used by an operation |
| `data` | Pointer to tensor data |
| `name` | Tensor name |

The `ne` field describes the tensor's dimensions.

For example:

```text
ne = [3, 4]
```

Conceptually, this is a matrix with 3 elements in one dimension and 4 in the other.

## Computation graphs

One interesting part of GGML is its use of computation graphs.

Instead of necessarily executing every operation immediately, code can construct a graph describing the operations that need to happen.

For example:

```text
Input Tensor
     |
     v
Matrix Multiplication
     |
     v
Normalization
     |
     v
Activation
     |
     v
Output Tensor
```

The graph describes the sequence and dependencies of operations.

A backend can then execute that graph.

This approach helps the runtime organize computation and schedule operations efficiently.

## Hardware backends

GGML supports different backends.

A backend is responsible for executing computations on a particular hardware platform.

For example:

```text
                 GGML
                   |
        +----------+----------+
        |          |          |
        v          v          v
       CPU       CUDA       Metal
```

This allows the same general tensor operations to execute on different hardware.

## Example: Matrix multiplication

Conceptually, GGML can create a matrix multiplication operation using an API such as:

```c
struct ggml_tensor * result =
    ggml_mul_mat(ctx, weights, input);
```

This creates a tensor operation.

Depending on the execution path, the operation may be part of a computation graph that is executed later.

The important point is that GGML provides the tensor operations used by the inference engine.

## GGML vs GGUF vs llama.cpp

| Name | What it is |
|---|---|
| GGUF | Model storage format |
| GGML | Tensor computation library |
| llama.cpp | Application/runtime that loads and executes models |

A useful analogy:

- GGUF is the model's storage format.
- GGML is the mathematical computation infrastructure.
- llama.cpp coordinates the actual inference process.

You do not need to install GGML separately just to use llama.cpp. It is part of the broader project ecosystem.

---

# How llama.cpp Performs Inference

Now that we understand GGUF, tensors, quantization, and GGML, we can look at what happens when llama.cpp runs a model.

Suppose we provide this prompt:

```text
Explain what a process is in operating systems.
```

The inference process is approximately:

```text
User Prompt
    |
    v
Tokenization
    |
    v
Token IDs
    |
    v
Transformer Layers
    |
    v
Tensor Operations
    |
    v
Next Token Probabilities
    |
    v
Sampling
    |
    v
Generated Token
    |
    v
Repeat
```

## Step 1: Load the model

llama.cpp loads the GGUF file.

It reads the metadata and tensor information, then initializes the model's weights.

## Step 2: Tokenization

The input text is converted into token IDs.

For example, a sentence is not necessarily represented as one token per word.

A tokenizer can split text into smaller units.

Conceptually:

```text
"Hello world"

        |

        v

[Token ID 1, Token ID 2]
```

The actual IDs depend on the model's tokenizer.

## Step 3: Transformer computation

The token IDs are converted into numerical representations.

The model then processes them through its transformer layers.

These layers perform operations involving:

- Embeddings.
- Attention.
- Matrix multiplication.
- Normalization.
- Feed-forward networks.

GGML provides many of the underlying tensor operations.

## Step 4: Predict the next token

After processing the input, the model produces scores representing possible next tokens.

For example:

```text
"An operating system manages..."

Possible next tokens:

resources: 0.65
processes: 0.20
hardware: 0.10
other:     0.05
```

These numbers are illustrative, not actual model output.

The runtime uses a sampling strategy to select the next token.

## Step 5: Autoregressive generation

The model generates text one token at a time.

For example:

```text
Input:
"What is an operating system?"

Generation:
"An"
" operating"
" system"
" is"
" software"
" that"
...
```

The generated tokens are fed back into the model's context to generate subsequent tokens.

This continues until the model produces a stop token or reaches a configured generation limit.

This is called autoregressive generation.

---

# What Is llama-server?

`llama-server` is a server application provided by llama.cpp.

It allows other applications to communicate with the inference engine over HTTP.

Instead of embedding the inference engine directly into your Go application, you can run llama-server as a separate process.

Your Go application then sends requests to it.

## Why do we need it?

Imagine you want to build a Go application that uses a local LLM.

You could try to integrate the inference engine directly into your Go program, but that introduces additional complexity.

You would need to deal with:

- Native libraries.
- C/C++ interoperability.
- GPU configuration.
- Model loading.
- Memory management.
- Inference configuration.

Instead, llama-server provides an HTTP API.

Your Go application can simply make HTTP requests.

## Starting llama-server

An example command:

```bash
llama-server \
    -m ./models/qwen3-14b-q4_k_m.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    -ngl 99
```

Explanation:

| Argument | Meaning |
|---|---|
| `-m` | Path to the GGUF model |
| `--host` | Network interface to bind to |
| `--port` | HTTP server port |
| `-ngl 99` | Request offloading up to 99 model layers to the GPU |

Note: `-ngl 99` requests GPU offloading. It does not guarantee every layer will fit into GPU memory.

The server will load the model and expose its API.

## OpenAI-compatible API

One useful feature is that llama-server supports OpenAI-compatible API endpoints.

For example:

```text
POST http://127.0.0.1:8080/v1/chat/completions
```

This means you can use familiar HTTP request structures.

Example request:

```json
{
  "model": "local-model",
  "messages": [
    {
      "role": "user",
      "content": "Explain what a process is."
    }
  ],
  "temperature": 0.7
}
```

The server processes the request and returns a response.

The exact supported options depend on the llama.cpp version and server configuration.

## Calling llama-server from Go

A simplified Go example:

```go
package main

import (
    "bytes"
    "fmt"
    "net/http"
)

func main() {
    jsonData := []byte(`{
        "model": "local-model",
        "messages": [
            {
                "role": "user",
                "content": "Explain what a process is."
            }
        ]
    }`)

    resp, err := http.Post(
        "http://127.0.0.1:8080/v1/chat/completions",
        "application/json",
        bytes.NewBuffer(jsonData),
    )

    if err != nil {
        panic(err)
    }

    defer resp.Body.Close()

    fmt.Println("HTTP status:", resp.Status)
}
```

This example demonstrates the HTTP connection only. A real client should also check the response status and decode the response JSON body.

The important takeaway:

**Your Go application does not need to understand tensors, GGUF, or GPU execution. It communicates with llama-server through HTTP.**

---

# What Is an AI Agent Harness?

This is one of the most important concepts when building your own AI agent.

An AI agent is not simply an LLM.

The LLM is responsible for interpreting information, reasoning, and generating responses.

The agent harness is the traditional software surrounding the LLM that manages the overall workflow.

It coordinates:

- Conversation history.
- System instructions.
- Tool definitions.
- Tool execution.
- Results returned from tools.
- Further LLM requests.
- Final responses.

## LLM vs Agent Harness

Consider this example:

You ask your threat-hunting agent:

```text
Check whether PID 1234 made any suspicious network connections.
```

The LLM might determine that it needs process and network telemetry.

However, the LLM cannot directly query your Elasticsearch instance or execute arbitrary Go functions.

The agent harness is responsible for that.

The workflow looks like this:

```text
User
 |
 v
Go Agent Harness
 |
 v
LLM
 |
 | "I need process information for PID 1234"
 v
Go Agent Harness
 |
 | Executes GetProcessDetails(1234)
 v
Elasticsearch / EDR API
 |
 | Returns process information
 v
Go Agent Harness
 |
 | Sends tool result back to LLM
 v
LLM
 |
 | Analyzes the result
 v
Go Agent Harness
 |
 v
Final Response
```

## What does the LLM actually return?

The LLM can return a structured tool-call request.

For example:

```json
{
  "name": "get_process_details",
  "arguments": {
    "pid": 1234
  }
}
```

This is not an actual function execution.

It is simply the model requesting that a tool be called.

The harness receives the request and decides what to do.

## Example: Go tool implementation

Suppose we have a Go function:

```go
func GetProcessDetails(pid int) (string, error) {
    // Query Elasticsearch or an EDR API.

    return "Process details for PID 1234", nil
}
```

We can expose a description of this function to the LLM:

```json
{
  "name": "get_process_details",
  "description": "Retrieve process information using a process ID.",
  "parameters": {
    "type": "object",
    "properties": {
      "pid": {
        "type": "integer",
        "description": "Process ID"
      }
    },
    "required": ["pid"]
  }
}
```

The description tells the LLM:

- What the tool does.
- When it should be used.
- What arguments it accepts.
- What information it needs to provide.

The LLM can then request the tool when appropriate.

## The Agent Loop

A simplified agent loop looks like this:

```text
1. Receive user input.

2. Send conversation and tool definitions to the LLM.

3. Receive LLM response.

4. Check whether the LLM requested a tool.

5. If a tool was requested:
      a. Validate the tool name.
      b. Validate the arguments.
      c. Execute the corresponding Go function.
      d. Capture the result.
      e. Send the result back to the LLM.
      f. Repeat.

6. If no tool was requested:
      Return the final response.
```

Conceptual Go example:

```go
for {
    response, err := callLLM(messages, tools)
    if err != nil {
        return err
    }

    if len(response.ToolCalls) == 0 {
        fmt.Println(response.Content)
        break
    }

    for _, toolCall := range response.ToolCalls {
        result, err := executeTool(toolCall)

        if err != nil {
            result = err.Error()
        }

        messages = append(messages, result)
    }
}
```

This is intentionally simplified pseudocode. A real implementation must preserve the correct assistant tool-call and tool-result message structure expected by the API.

## Important security consideration

Never blindly execute whatever the LLM requests.

The harness should:

- Allowlist available tools.
- Validate arguments.
- Enforce permissions.
- Apply timeouts.
- Limit the amount of returned data.
- Handle errors.
- Log tool execution.
- Require approval for sensitive actions when appropriate.

For a threat-hunting agent, this is especially important.

For example, a tool that searches Elasticsearch is generally different from a tool that isolates a machine or kills a process.

Read-only operations should be separated from actions that modify systems.

---

# Where Does MCP Fit In?

MCP stands for **Model Context Protocol**.

It is a protocol designed to standardize how AI applications communicate with external tools and context providers.

One important distinction:

**MCP is not an LLM, and it is not an agent harness.**

It is a communication protocol.

## What problem does MCP solve?

Imagine you build an agent that needs access to:

- Elasticsearch.
- CrowdStrike Falcon.
- IDA Pro.
- A filesystem.
- A database.

Without a common protocol, you might implement a different integration mechanism for every tool.

MCP provides a standardized way to expose tools and resources.

Conceptually:

```text
                 Agent Harness
                       |
                    MCP Client
                       |
             +---------+---------+
             |         |         |
             v         v         v
          MCP Server MCP Server MCP Server
             |         |         |
             v         v         v
            ELK       IDA       Filesystem
```

An MCP server exposes capabilities, and an MCP client communicates with that server.

## Does MCP execute the tool?

The MCP server handles the requested operation and returns a result.

However, MCP does not independently decide what the LLM should do.

The overall agent application still needs to:

1. Provide relevant tool descriptions to the model.
2. Receive the model's tool request.
3. Route the request appropriately.
4. Use an MCP client to invoke the server.
5. Return the result to the model.

The harness remains responsible for orchestrating the workflow.

## Direct function calls vs MCP

There are two possible approaches.

### Approach 1: Direct Go function calls

Your harness directly maps tool names to Go functions.

```text
LLM
 |
 v
Go Agent Harness
 |
 v
Go Function
 |
 v
Elasticsearch
```

For example:

```go
func executeTool(name string, args map[string]any) (string, error) {
    switch name {
    case "search_elk":
        return searchELK(args)

    case "get_process_details":
        return getProcessDetails(args)

    default:
        return "", fmt.Errorf("unknown tool: %s", name)
    }
}
```

Advantages:

- Simple architecture.
- Easy to understand.
- No additional protocol implementation.
- Good for learning and small projects.

Disadvantages:

- Integrations are tightly coupled to your application.
- Reusing tools in other applications requires additional work.
- You must maintain your own integration conventions.

### Approach 2: MCP

Your harness communicates with MCP servers.

```text
LLM
 |
 v
Go Agent Harness
 |
 v
MCP Client
 |
 v
MCP Server
 |
 v
Elasticsearch
```

Advantages:

- Standardized tool communication.
- Integrations can be reused by different MCP-compatible applications.
- Tools can be developed and maintained independently.
- Cleaner separation between the agent and external integrations.

Disadvantages:

- Additional protocol and client/server concepts to learn.
- More infrastructure.
- Potentially unnecessary complexity for a very small project.

## Is MCP better?

MCP is useful when you want standardized, reusable integrations.

However, it is not automatically better for every project.

For learning how agents work, direct Go function calls are a better starting point.

Once the basic agent loop makes sense, introducing MCP becomes much easier.

The important thing is to understand the separation:

| Component | Responsibility |
|---|---|
| LLM | Decides what information it needs and generates tool requests |
| Agent harness | Coordinates the LLM and tool execution |
| MCP client | Communicates with MCP servers |
| MCP server | Exposes tools and handles their execution |
| External system | Provides the actual data or functionality |

Also, MCP and model-native tool calling are separate concepts. MCP standardizes communication between applications and tool providers; the model still needs an appropriate tool-calling interface or structured-output mechanism.

---

# Putting Everything Together

Now we can combine everything into one architecture.

```text
                         USER
                           |
                           v
                    GO AGENT HARNESS
                           |
                 +---------+---------+
                 |                   |
                 v                   v
             LLM REQUEST          TOOL ROUTER
                 |                   |
                 v                   v
            llama-server         MCP Client
                 |                   |
                 v                   v
             llama.cpp          MCP Servers
                 |                   |
                 v             +-----+------+
               GGML            |     |      |
                 |             v     v      v
                 v            ELK   IDA    Falcon
             GGUF MODEL
```

Let's walk through a hypothetical request:

```text
User:
Investigate whether PID 1234 performed suspicious activity.
```

### Step 1: User input

The Go harness receives the request.

### Step 2: Send request to the LLM

The harness sends:

- System instructions.
- Conversation history.
- User question.
- Available tool definitions.

The request is sent to llama-server.

### Step 3: Model reasoning

The model determines that it needs telemetry about PID 1234.

It returns a tool-call request.

### Step 4: Harness executes the tool

The Go harness validates the tool request and executes the appropriate function or invokes an MCP server.

For example:

```text
search_process_events(pid=1234)
```

### Step 5: External system returns data

Elasticsearch returns matching events.

### Step 6: Send results back to the LLM

The harness adds the tool result to the conversation and sends another request to llama-server.

### Step 7: Model analyzes the information

The LLM reviews the telemetry and may request additional information.

For example:

- Network connections.
- Parent process information.
- Child processes.
- File modifications.

The harness executes those requests as well.

### Step 8: Final response

Once the LLM has enough information, it generates a final response.

The harness returns it to the user.

---

# Building a Local Threat-Hunting Agent

The goal is to build a Go application that uses a local LLM and can interact with security telemetry.

A sensible learning progression is to build the simplest version first.

## Phase 1: Run llama.cpp

Start by getting a local model running.

Tasks:

- Build or install llama.cpp.
- Download a compatible GGUF model.
- Understand the model's quantization format.
- Run the model locally.
- Experiment with prompts and generation settings.

Example:

```bash
llama-cli -m ./models/model.gguf -p "Explain process injection."
```

## Phase 2: Run llama-server

Expose the model through an HTTP API.

Example:

```bash
llama-server \
    -m ./models/model.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    -ngl 99
```

Test the API using curl or an HTTP client.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "local-model",
      "messages": [
        {
          "role": "user",
          "content": "Explain process injection."
        }
      ]
    }'
```

## Phase 3: Build a Go HTTP client

Write a small Go application that sends prompts to llama-server.

At this stage, do not worry about agents or tools.

Just understand:

- HTTP requests.
- JSON serialization.
- Response parsing.
- Conversation messages.
- Error handling.

## Phase 4: Implement a simple tool

Create one Go function.

For example:

```go
func SearchELK(query string) (string, error) {
    // Execute an Elasticsearch query.
    return "Example search results", nil
}
```

Expose its description to the model.

Make sure the model can request the function through the supported tool-calling interface.

Important: not every model or GGUF chat template supports reliable tool calling. Verify that your selected model, its chat template, and your llama.cpp version support the tool-calling format you intend to use.

## Phase 5: Build the agent loop

Implement the workflow:

```text
User Input
    |
    v
Call LLM
    |
    v
Tool Requested?
    |
   Yes
    |
    v
Execute Go Function
    |
    v
Return Tool Result to LLM
    |
    v
Call LLM Again
    |
    v
Final Answer
```

At this point, you have a basic agent.

## Phase 6: Add MCP

Once the direct function-calling implementation makes sense:

- Learn the MCP client/server architecture.
- Create or connect to an MCP server.
- Expose your Elasticsearch integration through MCP.
- Update the Go harness to invoke MCP tools.
- Compare the direct-function and MCP implementations.

## Phase 7: Expand the threat-hunting capabilities

Potential tools:

| Tool | Purpose |
|---|---|
| `search_elk` | Search Elasticsearch events |
| `get_process_details` | Retrieve process information |
| `get_network_connections` | Retrieve network telemetry |
| `get_child_processes` | Find process descendants |
| `get_file_events` | Retrieve file activity |
| `get_registry_events` | Retrieve registry modifications |
| `get_user_logons` | Investigate authentication events |

Initially, keep these tools read-only.

You can introduce controlled response actions later, with explicit authorization and auditing.

---

# Key Takeaways

| Concept | Remember this |
|---|---|
| llama.cpp | The inference engine that runs the model |
| GGUF | The binary model storage format |
| Quantization | Reduces numerical precision to save memory |
| Tensor | A multidimensional array of numerical values |
| GGML | Provides tensor operations and computation infrastructure |
| llama-server | Exposes llama.cpp through an HTTP API |
| Agent harness | Traditional application code that coordinates the LLM and tools |
| MCP | Standardized communication protocol for tools and context |

## The most important mental models

**1. GGUF is not an inference engine.**

It stores the model's weights and metadata.

**2. Quantization is not a file format.**

It is a technique for representing numerical values with less precision.

**3. GGML is not the LLM.**

It provides the tensor computation infrastructure used by the inference runtime.

**4. llama.cpp performs inference.**

It loads the model and executes the computations required to generate tokens.

**5. llama-server makes inference accessible over HTTP.**

Your Go application can communicate with it without directly integrating C/C++ inference code.

**6. An agent is not just an LLM.**

The harness is traditional software responsible for managing the model, conversation, and tool execution.

**7. MCP is not mandatory.**

You can implement tool calling directly in Go. MCP becomes useful when you want standardized and reusable integrations.

---

## Recommended Learning Order

For this project, follow this order:

```text
GGUF
  |
  v
Quantization
  |
  v
Tensors
  |
  v
GGML
  |
  v
llama.cpp
  |
  v
llama-server
  |
  v
Go HTTP Client
  |
  v
Agent Harness
  |
  v
Tool Calling
  |
  v
MCP
  |
  v
Threat-Hunting Agent
```

The end goal is not to build another LLM.

It is to build a Go application that uses a local LLM as its reasoning component and gives it controlled access to real security tools and telemetry.

---


# Further Reading: Local LLM Infrastructure

## 1. llama.cpp – Inference Engine

- [llama.cpp GitHub Repository](https://github.com/ggml-org/llama.cpp)

**What to explore:**
- Building llama.cpp.
- Running GGUF models.
- GPU offloading.
- Model loading and inference.
- `llama-server`.

## 2. GGUF – Model File Format

- [GGUF Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

**What to explore:**
- GGUF binary file structure.
- Metadata key-value pairs.
- Tensor information.
- Quantization types.
- Alignment and memory mapping.

## 3. GGUF Python Utilities – Inspect Model Files

- [GGUF Python Package and Utilities](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py)

**Useful utilities:**
- `gguf_dump.py` – Inspect GGUF metadata.
- `reader.py` – Understand how GGUF files are read.
- `writer.py` – Understand how GGUF files are constructed.

**Practical exercise:** Use `gguf_dump.py` to inspect one of your locally downloaded models.

## 4. GGML – Tensor Computation Library

- [GGML GitHub Repository](https://github.com/ggml-org/ggml)

**What to explore:**
- `include/ggml.h` – Tensor structures and API declarations.
- `src/ggml.c` – Core implementation, where applicable in the current repository layout.
- Tensor operations.
- Computation graphs.
- Backend implementations.

## 5. llama-server – HTTP API

- [llama.cpp GitHub Repository](https://github.com/ggml-org/llama.cpp)
- [llama-server Documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)

**What to explore:**
- Starting the HTTP server.
- Chat completion endpoints.
- OpenAI-compatible API.
- Server configuration.
- GPU configuration.
- Tool calling.

## 6. Function Calling – Connect the LLM to Go Tools

- [llama.cpp Function Calling Documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)

**What to explore:**
- Structured tool calls.
- Chat templates.
- Model compatibility.
- Tool definitions.
- Tool-call arguments.

Pay particular attention to `--jinja` and chat-template requirements. Not every model supports reliable function calling.

## 7. MCP – Model Context Protocol

- [Official MCP Introduction](https://modelcontextprotocol.io/docs/getting-started/intro)
- [MCP Architecture Overview](https://modelcontextprotocol.io/docs/learn/architecture)
- [MCP Specification](https://modelcontextprotocol.io/specification)

**What to explore:**
- Hosts.
- Clients.
- Servers.
- Tools.
- Resources.
- Prompts.
- Client-server communication.

---

## Recommended Learning Order

| Order | Resource | Learning objective |
|---|---|---|
| 1 | llama.cpp README | Understand how to run a model |
| 2 | GGUF specification | Understand model storage |
| 3 | GGUF Python utilities | Inspect an actual model |
| 4 | GGML repository | Explore tensor structures and operations |
| 5 | llama-server documentation | Build a Go HTTP client |
| 6 | Function calling documentation | Implement tool calling |
| 7 | MCP documentation | Understand standardized tool integration |

## Practical Exercises

1. Inspect a local GGUF model using `gguf_dump.py`.
2. Identify its architecture, metadata, tensor names, and quantization format.
3. Run the model using `llama-server`.
4. Send an HTTP request to it using Go.
5. Implement a simple Go function such as `search_elk`.
6. Expose that function to the LLM using tool calling.
7. Implement the agent loop.
8. Explore MCP and compare it with direct Go function calls.

**End goal:** Build a Go-based local threat-hunting agent that uses an LLM for reasoning and controlled access to Elasticsearch, CrowdStrike, and other security tools.
