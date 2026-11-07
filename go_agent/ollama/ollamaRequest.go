package ollama

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"
)

const DefaultModel = "qwen3.5:27b"

// stream false, to only send the final message
// Gemma respons was this:
/*
You are getting multiple JSON bodies because Ollama streams its responses by default. Instead of making you wait for the entire paragraph to generate, it sends the response token-by-token (word-by-word) as a series of independent JSON objects separated by newlines (NDJSON format).
If you want to receive exactly one single JSON body that you can unmarshal all at once, you need to turn streaming off by adding "stream": false to your request payload.
*/
const defaultReqBody = `{
      "model": "prutser/gemma-4-26B-A4B-it-ara-abliterated:Q3_K_M",
      "messages": [
        {
          "role": "user",
          "content": "%s",
		  "stream":"false"
        }
      ]
}`
const DefaultOllamaURL = "http://127.0.0.1:11434"

var ErrEmptyOllamaStructValue = errors.New("nil value for ollama response or request struct")

// ollamaRequest is the request we send to ollama
type ollamaRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Tools    []Tools         `json:"tools"`
	Stream   bool            `json:"stream"`
}
type ollamaMessage struct {
	Role       string      `json:"role"`
	Content    string      `json:"content"`
	ToolCallId string      `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCalls `json:"tool_calls,omitempty"`
	Name       string      `json:"name,omitempty"`
}

type IPAddress struct {
	Type        string `json:"type"`
	Description string `json:"description"`
}
type Properties struct {
	IPAddress IPAddress `json:"ip_address"`
}
type Parameters struct {
	Type       string     `json:"type"`
	Properties Properties `json:"properties"`
	Required   []string   `json:"required"`
}
type Function struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Parameters  Parameters `json:"parameters"`
}

type ollamaResponseFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Arguments   json.RawMessage `json:"arguments"`
}
type Tools struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type Arguments struct {
	IPAddress string `json:"ip_address"`
}

func SendRequest(url string, message string) error {

	//reqBodyStruct
	ollamaReq := ollamaRequest{
		Model: DefaultModel,
		Messages: []ollamaMessage{
			{
				Role:    "user",
				Content: message,
			},
		},
		Tools: []Tools{
			{
				Type: "function",
				Function: Function{
					Name:        "ip_threat_lookup",
					Description: "Retrieves reputation metrics and active abuse reports for a given IP address.",
					Parameters: Parameters{
						Type: "object",
						Properties: Properties{
							IPAddress: IPAddress{
								Type:        "string",
								Description: "The IPv4 or IPv6 address to investigate.",
							},
						},
						Required: []string{"ip_address"},
					},
				},
			},
		},

		Stream: false,
	}

	// Just do a string here, no need for marshalling, easier
	//reqStr := fmt.Sprintf(defaultReqBody, message)
	//reqBody := strings.NewReader(reqStr)
	//fmt.Printf("Sending request: %+v\n", tempMessage)
	fmt.Printf("Sending request %+v\n", ollamaReq)
	b, err := json.Marshal(&ollamaReq)
	if err != nil {
		return err
	}

	// create a new request
	httpRequest, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(b))
	if err != nil {
		return err
	}
	httpRequest.Header.Set("Content-Type", "Application/json")

	client := &http.Client{
		Timeout: 20 * time.Second,
	}

	resp, err := client.Do(httpRequest)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	ollamaResponse := &ollamaResponseStruct{}

	/*
		respStr, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}

		fmt.Println(string(respStr))
	*/

	if err := json.NewDecoder(resp.Body).Decode(ollamaResponse); err != nil {
		return err
	}

	ollamaResponse, err = handleToolCalls(url, &ollamaReq, ollamaResponse)
	if err != nil {
		fmt.Printf("error for handleTollCalls(): %v\n", err)
		return err
	}

	fmt.Printf("Response:\n%+v\n", ollamaResponse)

	return nil
}

func handleToolCalls(url string, ollamaReq *ollamaRequest, ollamaResp *ollamaResponseStruct) (*ollamaResponseStruct, error) {
	if ollamaResp == nil || ollamaReq == nil {
		return nil, ErrEmptyOllamaStructValue
	}

	ollamaClient := &http.Client{
		Timeout: 120 * time.Second,
	}

	cur_resp := ollamaResp
	maxIterations := 5
	curIter := 0
	for len(cur_resp.Message.ToolCalls) != 0 { // if LLM responded with some Tool Calls
		curIter += 1
		if curIter >= maxIterations {
			fmt.Println("Reached max iteration breaking")
			break
		}
		for _, v := range cur_resp.Message.ToolCalls { // Loop through all the tool calls
			switch v.Function.Name {
			case "ip_threat_lookup": // For IP threat lookup do this

				// Append tool call request and response in the original message, we want the entire history of message
				ollamaReq.Messages = append(ollamaReq.Messages, cur_resp.Message)
				ollamaReq.Messages = append(ollamaReq.Messages, ollamaMessage{
					Role:       "tool",
					ToolCallId: v.ID,
					Name:       v.Function.Name,
					Content:    "{\"status\": \"clean\", \"reputation_score\": 98, \"malware_reports\": 0}",
				})
			}
		}

		//fmt.Printf("Request after executing tool call: %+v\n", ollamaReq)
		// Pretty print debug
		p1, err := json.MarshalIndent(ollamaReq, "", "	")
		if err != nil {
			return nil, err
		}
		fmt.Println(string(p1))
		fmt.Println("---------------------------------------------------------")

		b, err := json.Marshal(&ollamaReq)
		if err != nil {
			return nil, err
		}
		fmt.Printf("\n--------------Actual final String: %s\n------------", string(b))
		req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(b))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "Application/json")

		resp, err := ollamaClient.Do(req)
		if err != nil {
			return nil, err
		}

		// Decoding it in current response
		// This makes it difficult to reason about the conversation state and can leave old fields around if the next response doesn't overwrite them.
		// A cleaner approach is to decode each response into a new variable.
		// encoding/json does not reset fields that are absent from the incoming JSON when decoding into an existing struct.
		//if err := json.NewDecoder(resp.Body).Decode(cur_resp); err != nil {
		//	return nil, err
		//}

		nextResp := &ollamaResponseStruct{}
		if err := json.NewDecoder(resp.Body).Decode(nextResp); err != nil {
			resp.Body.Close()
			return nil, err
		}
		cur_resp = nextResp
		fmt.Printf("---------------LLM RESPONSE-----------------------\n")
		fmt.Printf("%+v\n", cur_resp)
		fmt.Printf("--------------------------------------------------\n")
		resp.Body.Close()
	}

	// We reach here if tool call is zero.
	return cur_resp, nil

}
