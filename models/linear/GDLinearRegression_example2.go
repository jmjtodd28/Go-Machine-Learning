package models

import (
	"Go-Machine-Learning/utils"
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
)

// This example of Linear regression reads from a larger file of 100000 randomly generated examples
func readFloat64FromFile(filePath string) ([]float64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening file %s: %w", filePath, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var numbers []float64

	for scanner.Scan() {
		line := scanner.Text()
		value, err := strconv.ParseFloat(line, 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing float value %s: %w", line, err)
		}
		numbers = append(numbers, value)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file %s: %w", filePath, err)
	}

	return numbers, nil
}

func ExampleGDLinearRegression2() {

	// Read features from the CSV file
	featuresPath := "data/linear/features.csv"
	features, err := readFloat64FromFile(featuresPath)
	if err != nil {
		log.Fatalf("Error reading features file: %v", err)
	}

	// Read targets from the CSV file
	targetsPath := "data/linear/targets.csv"
	targets, err := readFloat64FromFile(targetsPath)
	if err != nil {
		log.Fatalf("Error reading targets file: %v", err)
	}

	X := utils.CreateMatrix(100000, 1, features)
	y := utils.CreateMatrix(100000, 1, targets)

	//Setting up hyperparameters
	glr := NewGDLinearRegression()
	glr.LearningRate = 0.01
	glr.MaxIter = 1000
	glr.GDescentType = "miniBatch"
	glr.batchSize = 128
	glr.earlyStopping = true
	glr.nIterNoChange = 5
	glr.Tol = 1e-4

	//Control whether you want information printed to the console
	glr.Verbose = true

	// Fit the model
	glr.Fit(X, y)

	// Get the R-Squared value of the model
	fmt.Println("R-Squared Error", RSquared(glr, X, y))

	//Get the coefficients and the bias
	fmt.Println("Bias:, ", glr.Bias, " Weights: ", glr.Coeffs.Data)

	//Make a prediciton based on the fitted model
	predict := utils.CreateMatrix(1, 1, []float64{4})
	yHat, err := glr.Predict(predict)
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println("Prediction:", yHat)
	}

}
