package mnist

import (
	"archive/zip"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func Unzip(zipPath, destPath string) error {
	r, err := zip.OpenReader(zipPath)
	defer r.Close()

	if err != nil {
		return err
	}

	// Iterate through the files in the archive
	for _, f := range r.File {

		// Skip __MACOSX directory and its contents
		if f.Name == "__MACOSX" || strings.HasPrefix(f.Name, "__MACOSX/") {
			continue
		}

		// Create the full path for the file/directory to extract
		fpath := filepath.Join(destPath, f.Name)

		// Check if the file already exists
		if _, err := os.Stat(fpath); err == nil {
			continue
		}

		// Check if the entry is a directory or file
		if f.FileInfo().IsDir() {
			// Create the directory
			os.MkdirAll(fpath, os.ModePerm)
		} else {
			// Ensure the directory exists for the file
			if err := os.MkdirAll(filepath.Dir(fpath), os.ModePerm); err != nil {
				return err
			}

			// Create the destination file
			outFile, err := os.OpenFile(fpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
			if err != nil {
				return err
			}
			defer outFile.Close()

			// Open the zip file entry for reading
			rc, err := f.Open()
			if err != nil {
				return err
			}

			// Copy the contents of the file to the destination
			_, err = io.Copy(outFile, rc)

			rc.Close()

			if err != nil {
				return err
			}
		}
	}

	return nil
}

func LoadMnistTrain() (*mat.Dense, *mat.Dense) {

	err := Unzip("datasets/mnist/archive.zip", "datasets/mnist/")
	if err != nil {
		fmt.Println(err)
	} else{
    fmt.Println("Extracted mnist dataset")
  }

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

  fmt.Println("Successfully read mnist dataset")

	return XTtrain, yTrain
}
