package mnist

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func LoadMnistTrain() (*mat.Dense, *mat.Dense) {
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

	XTtrain := mat.NewDense(60000, 784, features)
	yTrain := mat.NewDense(60000, 1, labels)

	return XTtrain, yTrain
}
