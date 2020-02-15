/*
* @Author: sf942274
* @Date:   2018-05-09 17:28:30
* @Last Modified by:   sf942274
* @Last Modified time: 2018-05-09 19:27:05
*/
/**
 * Function: Review annotation and make changes.
 * Inputs: 
 - cropped 3D images of lamina (tif file)
 - previously saved ROIs (zip file).
 * Outputs:
 - csv file: ROI exports with positions of each points.
 - .zip file: ROIs as raw data that can be imported to imageJ.
 - Snapshot images
 */



 /********************************** main() ***********************************/
print("\\Clear");
run("Close All");
run("Clear Results");

/* set environment */
run("Brightness/Contrast...");
run("Channels Tool...");

/* load original image and ROI*/
// open image
showMessage("Please load image.");
filePath = File.openDialog("Please load image.");
inputDirectory = File.getParent(filePath);
fileName = File.getName(filePath);
open(filePath);

// open ROI
showMessage("Please load ROI.");
roiPath = File.openDialog("Please load temporary result ROI.");
open(roiPath);

print(fileName);
print(roiPath);

// Set output directory
if(getBoolean("Save on Server?")){
	outputMasterDirectory = getDirectory("Choose a directory");
}
else{
	outputMasterDirectory = inputDirectory;
}


reviewBundleDetails(fileName, inputDirectory, outputMasterDirectory);

/* close all. */
showMessage("Program will stop now! \n Please close ROI manager. \n Please check whether files are saved!");
run("Close All");
run("Clear Results");
freeMemory();
/****************************** ebd of main() *********************************/


/****************************** Local Functions *******************************/
function chooseInputFile() {
	filePath=File.openDialog("Select a File (*.tif)");
	fileName=File.getName(filePath);
	inputDirectory = File.directory;
	print("filename: " + fileName);  // contains path + fileName
	return newArray(inputDirectory, fileName, filePath);
}

function reviewBundleDetails(fileName, inputDirectory, outputMasterDirectory) {
	// get the correct zoom and slice.
	setTool("hand");
	waitingMessage = "Get field of view...";
	msg = "Please Select appropriate zoom and slice#. \n Press ok, when done.";
	waitForUser(waitingMessage, msg);
	wait(2000);

	/* start recording */
	isStop = 1; // stop recording
	isSave = 1;  // save temporary results

	while(isStop){
		// record heels and angles
		bundleDetailsFilling(fileName);

		// make decision: save temp files/stop
		if (getBoolean("Save?"))
			isSave = 0;
		if (getBoolean("Stop?"))
			isStop = 0;

		// save files when stop recording.
		if (isStop == 0) {
			if(getBoolean("Done with the entire file?"))
				// save everything.
				saveBundleDetailsInfo(inputDirectory, outputMasterDirectory, fileName);
			else
				// save temporary results.
				saveTempInfo(inputDirectory, outputMasterDirectory, fileName, isStop);
		}
		else {
			// save temp files without stop recording.
			if (isSave == 0) {
				// save only ROI.
				saveTempInfo(inputDirectory, outputMasterDirectory, fileName, isStop);
			}
		}
	}
}

function bundleDetailsFilling(fileName){
	// set up the imageJ working environment
	run("Set Measurements...", "centroid bounding stack display add redirect=None decimal=3");

	// select the image file as the current working window.
	selectWindow(fileName);

	// reviewing
	setTool("multipoint");
	run("Point Tool...", "type=Hybrid color=Cyan size=Large label show counter=0");
	waitingMessage = "Select heels ...";
	msg = "Please start review. Update when necessary.";
	waitForUser(waitingMessage, msg);
	wait(1000);
}



function saveBundleDetailsInfo(inputDirectory, outputMasterDirectory, fileName) {
	// parameters
	myDateArray = defineDateAndTime();
	dateString = myDateArray[0];
	timeString = myDateArray[1];

	// check and/or make output directory.
	outputDirectory = checkDirectory(outputMasterDirectory + File.separator + "AfterReview");
	// ROIDirectory = checkDirectory(outputDirectory + File.separator + "ROI");
	// ssDirectory = checkDirectory(outputDirectory + File.separator + "SnapShots");

	// save heel x,y,z and angle info into csv file.
	selectWindow(fileName);
	csvName = getFileName(substring(fileName,0,indexOf(fileName, ".tif")), "ROI", dateString, timeString, ".csv");
	print(csvName);

	waitingMessage = "Getting ROIs to Measure...";
	msg = "Please select all ROIs and measure. ";
	waitForUser(waitingMessage, msg);
	wait(2000);

	csvpath = outputDirectory + File.separator + csvName;
	saveAs("Results", csvpath);


	// save heels and angles ROI to zip file.
	ROIName = getFileName(substring(fileName,0,indexOf(fileName, ".tif")), "ROI", dateString, timeString, ".zip");
	print(ROIName);
	ROIPath = outputDirectory + File.separator + ROIName;
	roiManager("save", ROIPath);

	//save heels and angles snapshot to tif: overall picture.
	selectWindow(fileName);
	print("file anme = " + fileName + "saving snapshot of overall picture...");
	roiManager("Show All with labels");

	waitingMessage = "Get field of view of Overall...";
	msg = "Snapshot no.1: with as much lamina as possible. \nPlease Select appropriate zoom and slice#. \n Press ok, when done.";
	waitForUser(waitingMessage, msg);
	wait(2000);

	run("Capture Image");

	capturedImageName = substring(fileName,0,indexOf(fileName, ".tif")) + "-1.tif";
	selectWindow(capturedImageName);
	bundleDetailImage = getFileName(substring(fileName,0,indexOf(fileName, ".tif")), "BundleDetailsSnapShot", dateString, timeString, "_Overall.tif");
	saveAs("Tiff", outputDirectory + File.separator + bundleDetailImage);


	//save heels and angles snapshot to tif: detailed picture.
	selectWindow(fileName);
	print("file anme = " + fileName);
	roiManager("Show All with labels");

	waitingMessage = "Get field of view of Zoomed...";
	msg = "Snapshot no.2: with ROI area as big as possible. \nPlease Select appropriate zoom and slice#. \n Press ok, when done.";
	waitForUser(waitingMessage, msg);
	wait(2000);

	run("Capture Image");

	capturedImageName = substring(fileName,0,indexOf(fileName, ".tif")) + "-1.tif";
	selectWindow(capturedImageName);
	bundleDetailImage = getFileName(substring(fileName,0,indexOf(fileName, ".tif")), "BundleDetailsSnapShot", dateString, timeString, "_Zoomed.tif");
	saveAs("Tiff", outputDirectory + File.separator + bundleDetailImage);

	//clear ROI from the image and save.
	selectWindow(fileName);
	run("Remove Overlay");
	run("Save");
}


function saveTempInfo(inputDirectory, outputMasterDirectory, fileName, isStop) {
	// parameters
	myDateArray = defineDateAndTime();
	dateString = myDateArray[0];
	timeString = myDateArray[1];

	// check and/or make output directory.
	outputDirectory = checkDirectory(outputMasterDirectory + File.separator + "Recording");
	ROIDirectory = checkDirectory(outputDirectory + File.separator + "ROIs");
	ssDirectory = checkDirectory(outputDirectory + File.separator + "SnapShots");


	// save heels and angles overlay to zip file.
	ROIName = getFileName(substring(fileName,0,indexOf(fileName, ".tif")), "ROI", dateString, timeString, ".zip");
	print(ROIName);
	ROIPath = ROIDirectory + File.separator + ROIName;

	roiManager("save", ROIPath);


	//If temporarily stop recording (a.k.a. about to close Fiji)
	if(isStop == 0) {
		// save heels and angles snapshot to tif.
		selectWindow(fileName);
		roiManager("Show All with labels");
		waitingMessage = "Get field of view ...";
		msg = "Please Select appropriate zoom and slice#. \n Press ok, when done.";
		waitForUser(waitingMessage, msg);
		wait(2000);

		run("Capture Image");

		capturedImageName = substring(fileName,0,indexOf(fileName, ".tif")) + "-1.tif";
		selectWindow(capturedImageName);
		bundleDetailImage = getFileName(substring(fileName,0,indexOf(fileName, ".tif")), "BundleDetailsSnapShot", dateString, timeString, ".tif");

		saveAs("Tiff", ssDirectory + File.separator + bundleDetailImage);

		// clear overlay from the image and save.
		selectWindow(fileName);
		run("Remove Overlay");
		run("Save");
	}
}

function checkDirectory(customDir){
	print(customDir);
	if (!File.exists(customDir))
		File.makeDirectory(customDir);
	if (!File.exists(customDir))
		exit("Unable to create directory: " + customDir);
	return customDir;
}



function getFileName(inputFileName, middleName, dateString, timeString, suffix){
	outputFilename = inputFileName + "_" + middleName + "_" + dateString + "_" + timeString + suffix;
	print(">> File <" + outputFilename);
	return outputFilename;
}



function defineDateAndTime() {
	getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);

	dateString = toString(year) + toString(month+1) + toString(dayOfMonth);
	timeString = toString(hour) + toString(minute) + toString(second);

	return newArray(dateString, timeString);
}



function freeMemory() {
	run("Collect Garbage");
	run("Collect Garbage");
	run("Collect Garbage");
	run("Collect Garbage");
	run("Collect Garbage");
	run("Collect Garbage");
	run("Collect Garbage");
	run("Collect Garbage");
	print("Memory Freed!");
	run("Monitor Memory...");
}
