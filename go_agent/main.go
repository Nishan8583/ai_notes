package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

const defaultModel = "prutser/gemma-4-26B-A4B-it-ara-abliterated:Q3_K_M"

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
const defaultOllamaURL = "http://127.0.0.1:11434"

type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ollamaRequest is the request we send to ollama
type ollamaRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
}
type ollamaResponseStruct struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Message   struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	Done               bool   `json:"done"`
	DoneReason         string `json:"done_reason"`
	TotalDuration      int    `json:"total_duration"`
	LoadDuration       int    `json:"load_duration"`
	PromptEvalCount    int    `json:"prompt_eval_count"`
	PromptEvalDuration int    `json:"prompt_eval_duration"`
	EvalCount          int    `json:"eval_count"`
	EvalDuration       int    `json:"eval_duration"`
}

func sendRequest(url string, message string) error {

	reqBodyStruct := ollamaRequest{
		Model: defaultModel,
		Messages: []ollamaMessage{
			{
				Role:    "user",
				Content: message,
			},
		},
		Stream: false,
	}
	// Just do a string here, no need for marshalling, easier
	//reqStr := fmt.Sprintf(defaultReqBody, message)
	//reqBody := strings.NewReader(reqStr)
	fmt.Printf("Sending request: %+v\n", reqBodyStruct)

	b, err := json.Marshal(&reqBodyStruct)
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

	fmt.Printf("Response:\n%s\n", ollamaResponse.Message.Content)

	return nil
}

func main() {
	url := defaultOllamaURL + "/api/chat"
	msg := "Explain Process Injection to me"

	if err := sendRequest(url, msg); err != nil {
		log.Fatalf("While sending request: %v\n", err)
	}
}
