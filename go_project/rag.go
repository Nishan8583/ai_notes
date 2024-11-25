package main

import (
	"context"
	"fmt"
	"os"

	"github.com/ledongthuc/pdf"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/schema"
)

func loadPDF(filepath string) ([]schema.Document, error) {
	docs := []schema.Document{}
	file, err := os.Open(filepath)
	if err != nil {
		println("Unable to open file", err)
		return docs, err
	}
	defer file.Close()

	pdf := documentloaders.NewPDF(file, 3000)

	// pdf := NewPDF(file, 3000)
	// splitter := textsplitter.NewRecursiveCharacter()

	docs, err = pdf.Load(context.Background())
	fmt.Println(docs)
	// docs, err = pdf.LoadAndSplit(context.Background(), splitter)
	if err != nil {
		fmt.Println("Could not load and split", err)
	}
	return docs, err
}

func loadText(filepath string) ([]schema.Document, error) {
	docs := []schema.Document{}
	file, err := os.Open(filepath)
	if err != nil {
		println("Unable to open file", err)
		return docs, err
	}
	defer file.Close()

	text := documentloaders.NewText(file)

	// pdf := NewPDF(file, 3000)
	// splitter := textsplitter.NewRecursiveCharacter()

	docs, err = text.Load(context.Background())
	fmt.Println(docs)
	// docs, err = pdf.LoadAndSplit(context.Background(), splitter)
	if err != nil {
		fmt.Println("Could not load and split", err)
	}
	return docs, err
}

func custom_qa() {
	llm, err := ollama.New(ollama.WithModel("llama3.2"))
	if err != nil {
		panic(err)
	}

	stuffsQAChain := chains.LoadStuffQA(llm)
	docs, err := loadText(
		"./text_sample.txt",
	)
	if err != nil {
		fmt.Println("Could not open file", err)
		panic(err)
	}

	answer, err := chains.Call(context.Background(), stuffsQAChain, map[string]any{
		"input_documents": docs,
		"question":        "which is EDR ?",
	})
	if err != nil {
		fmt.Println("Error could not get answer", err)
		return
	}
	fmt.Println("Answer is ", answer)
}

func readPDFmain() {
	pdf.DebugOn = true
	content, err := readPdf(
		"./text_sample.txt",
	) // Read local pdf file
	if err != nil {
		panic(err)
	}
	fmt.Println(content)
	return
}

func readPdf(path string) (string, error) {
	f, r, err := pdf.Open(path)
	defer func() {
		_ = f.Close()
	}()
	if err != nil {
		return "", err
	}
	totalPage := r.NumPage()

	for pageIndex := 1; pageIndex <= totalPage; pageIndex++ {
		p := r.Page(pageIndex)
		if p.V.IsNull() {
			continue
		}

		rows, _ := p.GetTextByRow()
		fmt.Println("Number", rows)
		for _, row := range rows {
			println(">>>> row: ", row.Position)
			for _, word := range row.Content {
				fmt.Println(word.S)
			}
		}
	}
	return "", nil
}
