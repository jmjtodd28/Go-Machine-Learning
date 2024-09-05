package neuralnetworks

import (
	"Go-Machine-Learning/utils"
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
)

func readFile() ([]float64, []float64) {
	var labels []float64
	var features []float64

	filePath := "dataSets/mnist/archive/mnist_train.csv"

	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("Failed to open file: %s", err)
	}
	defer file.Close()

	// Create a new CSV reader
	reader := csv.NewReader(file)

	// Read the first line to skip the header
	_, err = reader.Read() // Read the first line (header)
	if err != nil {
		log.Fatalf("Failed to read header: %s", err)
	}

	for {
		// Read each record line-by-line
		record, err := reader.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break // End of file, exit the loop
			}
			log.Fatalf("Failed to read line: %s", err)
		}

		// Convert the first element to float and append to labels slice
		firstElement, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			log.Fatalf("Failed to convert first element to float: %s", err)
		}
		labels = append(labels, firstElement)

		// Convert the rest of the elements to float and append to features slice
		for _, strElem := range record[1:] {
			num, err := strconv.ParseFloat(strElem, 64)
			if err != nil {
				log.Fatalf("Failed to convert element to float: %s", err)
			}
			features = append(features, num)
		}
	}

	for i := range 784 {
		if i%28 == 0 && i > 0 {
			fmt.Println()
		}
		fmt.Printf("%3.0f", features[i+(2*784)])
	}

	fmt.Println()

	l := len(features)
	fmt.Printf("l: %v\n", l)

	return features, labels
}

func MLPExample() {

	featuresData, labelData := readFile()

	X := utils.CreateMatrix(60000, 784, featuresData)
	y := utils.CreateMatrix(60000, 1, labelData)

	y.OneHotEncode(10)

	//normalise data
	for i := range X.Data {
		X.Data[i] = X.Data[i] / 255
	}

	/*
		//Creating a and gate
		X := utils.CreateMatrix(4, 2, []float64{
			0, 0,
			1, 0,
			0, 1,
			1, 1})
		//	X := utils.CreateMatrix(1, 2, []float64{ 1, 1})

		y := utils.CreateMatrix(4, 1, []float64{
			0,
			1,
			1,
			1,
		})
	*/
	//	y := utils.CreateMatrix(1, 1, []float64{ 2})

	mlp := NewMultiLayerPerceptron()

	mlp.Arch = []int{784, 10, 10, 10}
	mlp.Epochs = 100
	mlp.BatchSize = 128
	mlp.LearningRate = 0.01

	mlp.Train(X, y)

}
