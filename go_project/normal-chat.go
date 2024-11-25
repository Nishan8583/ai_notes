package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/tmc/langchaingo/llms/ollama"
)

func normalQA() {
	// Step 1: Initialize Ollama LLM
	ollamaClient, err := ollama.New(ollama.WithModel("llama3.2")) // Replace "your-model-name" with the actual model
	if err != nil {
		log.Fatalf("Failed to initialize Ollama LLM: %v", err)
	}

	// Step 2: Prepare for user input
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("Interactive Q&A with Ollama LLM. Type 'exit' to quit.")

	for {
		// Step 3: Read user input
		fmt.Print("You: ")
		query, err := reader.ReadString('\n')
		if err != nil {
			log.Fatalf("Failed to read input: %v", err)
		}
		query = strings.TrimSpace(query)

		// Check for exit condition
		if strings.ToLower(query) == "exit" {
			fmt.Println("Exiting. Goodbye!")
			break
		}

		// Step 4: Query the LLM
		response, err := ollamaClient.Call(context.Background(), query)
		if err != nil {
			log.Printf("Error querying LLM: %v\n", err)
			continue
		}

		// Step 5: Print the response
		fmt.Printf("LLM: %s\n", response)
	}
}
