/* 	
 *  Macro script to normalize local contrast of and rescale TEM images
 *  
 *  author: Fabienne K. Roessler, date: 26/07/2018
 */


// Define scaling factor
scalingFactor = 0.25;

// Ask user for input and output folder
inputPath = getDirectory("Choose Experiment Folder");
outputPath = getDirectory("Choose Output Directory");

// Get list of tile set folders saved in input folder
listSquares = getFileList(inputPath);

run("Close All");

setBatchMode(true);

// Loop over tile set folders
for (i=0; i<listSquares.length; i++) {
	
	// Get list of images in the current tile set folder
	inputFolder = inputPath+listSquares[i]+File.separator;
	listImages = getFileList(inputFolder);
	
	// Create output folder (= tile set folder)
	outputFolder = outputPath+listSquares[i]+File.separator;
	if (!File.exists(outputFolder)) {
		File.makeDirectory(outputFolder);
	}

	// Loop over list of images
	for (j=0; j<listImages.length; j++) {

		// Open image
		open(inputFolder+listImages[j]);
		title=getTitle();

		// Run normalize local contrast filter on open image
		run("Normalize Local Contrast", "block_radius_x=300 block_radius_y=300 standard_deviations=4 center stretch");

		// Run scale tool on open image
		run("Scale...", "x="+scalingFactor+" y="+scalingFactor+"  interpolation=Bilinear average create");

		// Convert image to 8-bit (to safe memory space)
		run("8-bit");
	
		// Save image as tiff-file
		saveAs("Tiff", outputFolder+title);
		
		// Close open image
		run("Close All");
	}
	
}