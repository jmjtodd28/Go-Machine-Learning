package boston

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func LoadBostonData() (*mat.Dense, *mat.Dense) {

	var labels []float64
	var features []float64

	filePath := "datasets/boston/housing.csv"

	file, err := os.Open(filePath)
	if err != nil {
		log.Fatalf("Failed to open file: %s", err)
	}

	defer file.Close()

	reader := csv.NewReader(file)

	for {
		// Read each record line-by-line
		record, err := reader.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break // End of file, exit the loop
			}
			log.Fatalf("Failed to read line: %s", err)
		}

		//file has delimiter " "
		line := strings.Split(record[0], " ")

		//extracting the features
		featuresStrings := line[0 : len(line)-1]
		for i := range featuresStrings {
			data := strings.TrimSpace(featuresStrings[i])
			if data != "" {
				feature, err := strconv.ParseFloat(data, 64)
				if err != nil {
					log.Fatalf("Failed to convert first element to float: %s", err)
				}
				features = append(features, feature)

			}
		}

		//extracting the label
		label, err := strconv.ParseFloat(line[len(line)-1], 64)
		if err != nil {
			log.Fatalf("Failed to convert first element to float: %s", err)
		}
		labels = append(labels, label)
	}

	X := mat.NewDense(506, 13, features)
	y := mat.NewDense(506, 1, labels)

	fmt.Println("Successfully read boston dataset")

	return X, y
}
