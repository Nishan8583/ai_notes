package main

import (
	"go_agent/ollama"
	"log"
)

func main() {
	url := ollama.DefaultOllamaURL + "/api/chat"
	msg := "Find information about ip address 1.1.1.1"

	if err := ollama.SendRequest(url, msg); err != nil {
		log.Fatalf("While sending request: %v\n", err)
	}
}
