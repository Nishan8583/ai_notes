package ollama

import (
	"time"
)

type ToolCalls struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function ollamaResponseFunction `json:"function"`
}

type ollamaResponseStruct struct {
	Model              string        `json:"model"`
	CreatedAt          time.Time     `json:"created_at"`
	Message            ollamaMessage `json:"message"`
	Done               bool          `json:"done"`
	DoneReason         string        `json:"done_reason"`
	TotalDuration      int           `json:"total_duration"`
	LoadDuration       int           `json:"load_duration"`
	PromptEvalCount    int           `json:"prompt_eval_count"`
	PromptEvalDuration int           `json:"prompt_eval_duration"`
	EvalCount          int           `json:"eval_count"`
	EvalDuration       int           `json:"eval_duration"`
}
