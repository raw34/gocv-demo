package main

import (
	"fmt"
	"gocv.io/x/gocv"
	"image"
	"image/color"
)

// detectFacesWithClassifier is a function to perform face detection on a given image file.
func detectFacesWithClassifier(imageFile string, xmlFile string, outputFile string) error {
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
	gocv.IMWrite(outputFile, img)
	fmt.Printf("saved to %s\n", outputFile)

	return nil
}

func min(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// detectWithSSD performs object detection on the given image using the SSD model,
// and saves the result to the specified output file.
func detectWithSSD(imgFile string, proto string, model string, outputFile string) error {
	img := gocv.IMRead(imgFile, gocv.IMReadColor)
	if img.Empty() {
		return fmt.Errorf("error reading image from file: %v", imgFile)
	}
	defer img.Close()

	net := gocv.ReadNetFromCaffe(proto, model)
	if net.Empty() {
		return fmt.Errorf("error reading network model from : %v %v", proto, model)
	}
	defer net.Close()

	W := float32(img.Cols())
	H := float32(img.Rows())

	blob := gocv.BlobFromImage(img, 1.0, image.Pt(128, 96), gocv.NewScalar(104.0, 177.0, 123.0, 0), false, false)
	defer blob.Close()

	net.SetInput(blob, "data")
	detBlob := net.Forward("detection_out")
	defer detBlob.Close()

	detections := gocv.GetBlobChannel(detBlob, 0, 0)
	defer detections.Close()

	green := color.RGBA{0, 255, 0, 0}
	for r := 0; r < detections.Rows(); r++ {
		confidence := detections.GetFloatAt(r, 2)
		if confidence < 0.5 {
			continue
		}

		left := detections.GetFloatAt(r, 3) * W
		top := detections.GetFloatAt(r, 4) * H
		right := detections.GetFloatAt(r, 5) * W
		bottom := detections.GetFloatAt(r, 6) * H

		left = min(max(0, left), W-1)
		right = min(max(0, right), W-1)
		bottom = min(max(0, bottom), H-1)
		top = min(max(0, top), H-1)

		rect := image.Rect(int(left), int(top), int(right), int(bottom))
		gocv.Rectangle(&img, rect, green, 3)
	}

	gocv.IMWrite(outputFile, img)
	fmt.Printf("saved result to: %s\n", outputFile)
	return nil
}
