# TODOS
- Give list of tools to LLM. Payload exmaple below:
```json
{
  "model": "prutser/gemma-4-26B-A4B-it-ara-abliterated:Q3_K_M",
  "messages": [
    { "role": "user", "content": "Look up IP 1.1.1.1 and see if it's on any blacklists." }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "ip_threat_lookup",
        "description": "Retrieves reputation metrics and active abuse reports for a given IP address.",
        "parameters": {
          "type": "object",
          "properties": {
            "ip_address": {
              "type": "string",
              "description": "The IPv4 or IPv6 address to investigate."
            }
          },
          "required": ["ip_address"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "dns_record_fetch",
        "description": "Queries DNS records (A, AAAA, MX, TXT) for a specified domain.",
        "parameters": {
          "type": "object",
          "properties": {
            "domain": { "type": "string" }
          },
          "required": ["domain"]
        }
      }
    }
  ],
  "stream": false
}

```

- Setup a Tool call flow. Example below:
- Initial request
```json
{
  "model": "prutser/gemma-4-26B-A4B-it-ara-abliterated:Q3_K_M",
  "messages": [
    { "role": "user", "content": "Check the threat status of IP 1.1.1.1" }
  ],
  "tools": [ /* your tool schemas here */ ],
  "stream": false
}
```
- LLM response:
```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": { "name": "ip_threat_lookup", "arguments": { "ip_address": "1.1.1.1" } }
      }
    ]
  }
}
```
- Follow up response:
```json
{
  "model": "prutser/gemma-4-26B-A4B-it-ara-abliterated:Q3_K_M",
  "messages": [
    // 1. The original prompt
    { "role": "user", "content": "Check the threat status of IP 1.1.1.1" },
    
    // 2. The exact assistant message returned in Step 2 (including the tool_calls data)
    { 
      "role": "assistant", 
      "content": "", 
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": { "name": "ip_threat_lookup", "arguments": { "ip_address": "1.1.1.1" } }
        }
      ] 
    },
    
    // 3. Your new tool response message (matches the ID above)
    { 
      "role": "tool", 
      "tool_call_id": "call_abc123", 
      "name": "ip_threat_lookup",
      "content": "{\"status\": \"clean\", \"reputation_score\": 98, \"malware_reports\": 0}" 
    }
  ],
  "tools": [ /* keep sending the tool schemas here in case it needs another tool next */ ],
  "stream": false
}

```
- If LLM wants to call the tool:
```json
{
  "model": "prutser/gemma-4-26B-A4B-it-ara-abliterated:Q3_K_M",
  "created_at": "2026-11-03T15:10:00Z",
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "id": "call_xyz789",
        "type": "function",
        "function": {
          "name": "ip_threat_lookup",
          "arguments": {
            "ip_address": "1.1.1.1"
          }
        }
      }
    ]
  },
  "done_reason": "stop",
  "done": true
}
```
- When its done
```json
{
  "model": "prutser/gemma-4-26B-A4B-it-ara-abliterated:Q3_K_M",
  "created_at": "2026-11-03T15:12:00Z",
  "message": {
    "role": "assistant",
    "content": "Based on the threat lookup report, IP 1.1.1.1 is fully clean and safe to use. It currently maintains a high reputation score with zero listed malware tracking associations.",
    "tool_calls": null 
  },
  "done_reason": "stop",
  "done": true
}
```
- Your Parser Logic: If message.tool_calls is null or empty, treat the string value inside message.content as the final text response ready to present to your end use

### References
- [Ollama Tool Calling](https://docs.ollama.com/capabilities/tool-calling)