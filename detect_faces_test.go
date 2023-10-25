package main

import (
	"fmt"
	"testing"
)

func Test_detectFacesWithClassifier(t *testing.T) {
	imageFile := "images/01.png"
	modelFile := "data/haarcascade_frontalface_default.xml"
	outputFile := "output_01.jpg"

	// call the detectFacesWithClassifier function
	err := detectFacesWithClassifier(imageFile, modelFile, outputFile)
	if err != nil {
		fmt.Println(err)
	}
}

func Test_detectWithSSD(t *testing.T) {
	imgFile := "images/02.png"
	proto := "data/deploy.phototxt"
	model := "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
	outputFile := "output_02.jpg"

	err := detectWithSSD(imgFile, proto, model, outputFile)
	if err != nil {
		fmt.Println(err)
	}
}
