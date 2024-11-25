package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/ollama/ollama/api"
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
)

func textToSplit() []schema.Document {
	f, err := os.Open("./ApplicationLevelRootkitDetectionProgramforDebianLinux.pdf")
	if err != nil {
		fmt.Println("Error opening file: ", err)
	}

	p := documentloaders.NewPDF(f, 1024)

	split := textsplitter.NewRecursiveCharacter()
	split.ChunkSize = 300   // size of the chunk is number of characters
	split.ChunkOverlap = 30 // overlap is the number of characters that the chunks overlap
	docs, err := p.LoadAndSplit(context.Background(), split)
	if err != nil {
		fmt.Println("Error loading document: ", err)
	}

	log.Println("Document loaded: ", len(docs))

	return docs
}

func llama() {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	messages := []api.Message{
		{
			Role:    "system",
			Content: "You are a expert in cyber security",
		},
		{
			Role:    "user",
			Content: "What is heartbleed vulnerability?",
		},
	}

	ctx := context.Background()
	req := &api.ChatRequest{
		Model:    "llama3.2",
		Messages: messages,
	}

	respFunc := func(resp api.ChatResponse) error {
		fmt.Print(resp.Message.Content)
		return nil
	}

	err = client.Chat(ctx, req, respFunc)
	if err != nil {
		log.Fatal(err)
	}
}

func minstral() {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	req := &api.PullRequest{
		Model: "mistral",
	}
	progressFunc := func(resp api.ProgressResponse) error {
		fmt.Printf(
			"Progress: status=%v, total=%v, completed=%v\n",
			resp.Status,
			resp.Total,
			resp.Completed,
		)
		return nil
	}

	err = client.Pull(ctx, req, progressFunc)
	if err != nil {
		log.Fatal(err)
	}
}

func main() {
	// readPDFmain()
	custom_qa()
}
