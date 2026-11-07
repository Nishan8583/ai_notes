package ollama

import (
	"bytes"
	"encoding/json"
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

// ollamaRequest is the request we send to ollama
type ollamaRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Tools    []Tools         `json:"tools"`
	Stream   bool            `json:"stream"`
}
type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
type Tools struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type ollamaResponseStruct struct {
	Model              string    `json:"model"`
	CreatedAt          time.Time `json:"created_at"`
	Message            message   `json:"message"`
	Done               bool      `json:"done"`
	DoneReason         string    `json:"done_reason"`
	TotalDuration      int       `json:"total_duration"`
	LoadDuration       int       `json:"load_duration"`
	PromptEvalCount    int       `json:"prompt_eval_count"`
	PromptEvalDuration int       `json:"prompt_eval_duration"`
	EvalCount          int       `json:"eval_count"`
	EvalDuration       int       `json:"eval_duration"`
}

type Arguments struct {
	IPAddress string `json:"ip_address"`
}
type ToolCalls struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Function Function `json:"function"`
}
type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`

	ToolCalls []ToolCalls `json:"tool_calls"`
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

	ollamaResponse := ollamaResponseStruct{}

	/*
		respStr, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}

		fmt.Println(string(respStr))
	*/

	if err := json.NewDecoder(resp.Body).Decode(&ollamaResponse); err != nil {
		return err
	}

	fmt.Printf("Response:\n%+v\n", ollamaResponse)

	return nil
}
