package main

import (
	"fmt"
	"gocv.io/x/gocv"
	"image"
	"image/color"
)

// detectFaces is a function to perform face detection on a given image file.
func detectFaces(imageFile string, xmlFile string) error {
	// color for the rect when faces detected
	blue := color.RGBA{0, 0, 255, 0}

	// load classifier to recognize faces
	classifier := gocv.NewCascadeClassifier()
	defer classifier.Close()

	if !classifier.Load(xmlFile) {
		return fmt.Errorf("Error reading cascade file: %v", xmlFile)
	}

	// load image from file
	img := gocv.IMRead(imageFile, gocv.IMReadColor)
	if img.Empty() {
		return fmt.Errorf("Error reading image from file: %v", imageFile)
	}

	rects := classifier.DetectMultiScale(img)
	fmt.Printf("found %d faces\n", len(rects))
	// draw a rectangle around each face on the original image,
	// along with text identifying as "Human"
	for _, r := range rects {
		gocv.Rectangle(&img, r, blue, 3)

		size := gocv.GetTextSize("Human", gocv.FontHersheyPlain, 1.2, 2)
		pt := image.Pt(r.Min.X+(r.Min.X/2)-(size.X/2), r.Min.Y-2)
		gocv.PutText(&img, "Human", pt, gocv.FontHersheyPlain, 1.2, blue, 2)
	}

	// Save the result image to a file
	outputFile := "output.jpg"
	gocv.IMWrite(outputFile, img)
	fmt.Printf("saved to %s\n", outputFile)

	return nil
}

func main() {
	imageFile := "images/01.png"
	xmlFile := "data/haarcascade_frontalface_default.xml"

	// call the detectFaces function
	err := detectFaces(imageFile, xmlFile)
	if err != nil {
		fmt.Println(err)
	}
}
