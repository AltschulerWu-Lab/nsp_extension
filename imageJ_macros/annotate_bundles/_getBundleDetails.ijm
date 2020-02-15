/*
* @Author: lily
* @Date:   2017-11-01 19:48:04
* @Last Modified by:   sf942274
* @Last Modified time: 2020-02-14 16:20:12
*/
/**
 * Function: annotate lamina images for growth cone morphology analysis. Mark heel bundles and target positions
 * Inputs: 
 - cropped 3D images of lamina (tif file)
 - previously saved ROIs (zip file) if applicable.
 * Outputs:
 - csv file: ROI exports with positions of each points.
 - .zip file: ROIs as raw data that can be imported to imageJ.
 - Snapshot images
**/


/*********************************** main() ***********************************/
print("\\Clear");
run("Close All");
run("Clear Results");

/******* start new data *******/
if (getBoolean("New Data?")){
	/* load original image*/
	// open image
	showMessage("Please load image.");
	filePath = File.openDialog("Please load image.");
	inputDirectory = File.getParent(filePath);
	fileName = File.getName(filePath);
	open(filePath);

	// get output directory
	if(getBoolean("Save on Server?")){
		outputMasterDirectory = getDirectory("Choose a directory");
	}
	else{
		outputMasterDirectory = inputDirectory;
	}

	print(inputDirectory);
	print(fileName);

	selectWindow(fileName);

	recordBundleDetails(fileName, inputDirectory, outputMasterDirectory);
}

/******* loading previous data *******/
else {
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

	// get output directory
	if(getBoolean("Save on Server?")){
		outputMasterDirectory = getDirectory("Choose a directory");
	}
	else{
		outputMasterDirectory = inputDirectory;
	}

	print(fileName);
	print(roiPath);

	selectWindow(fileName);

	recordBundleDetails(fileName, inputDirectory, outputMasterDirectory);
}

/* close all. */
showMessage("Program will stop now! \n Please close ROI manager. \n Please check whether files are saved!");
run("Close All");
run("Clear Results");
freeMemory();
/****************************** ebd of main() *********************************/


/****************************** Local Functions *******************************/
// function chooseInputFile() {
// 	filePath=File.openDialog("Select a File (*.tif)");
// 	fileName=File.getName(filePath);
// 	inputDirectory = File.directory;
// 	print("filename: " + fileName);  // contains path + fileName
// 	return newArray(inputDirectory, fileName, filePath);
// }

function recordBundleDetails(fileName, inputDirectory, outputMasterDirectory) {
	/* set up working environment*/
	// get tools
	run("Brightness/Contrast...");
	run("Channels Tool...");

	// get the correct zoom and slice.
	setTool("hand");
	waitingMessage = "Get field of view...";
	msg = "Please Select appropriate zoom and slice#. \n Press ok, when done.";
	waitForUser(waitingMessage, msg);
	wait(2000);

	/* start recording */
	isStop = 1; // stop recording
	isSave = 1;  // save temporary results
	count = 1;

	while(isStop){
		// set parameter for tools.
		run("Point Tool...", "type=Hybrid color=Cyan size=Large label show counter=0");
		
		// record heels and angles
		bundleDetailsFilling(fileName);
		count = count + 1;

		// make decision: save temp files/stop
		if(count == 5){
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
			count = 1;
		}
		
	}
}

function bundleDetailsFilling(fileName){
	// set up the imageJ working environment
	run("Set Measurements...", "centroid bounding stack display add redirect=None decimal=3");
	// select the image file as the current working window.
	selectWindow(fileName);

	// select heels and bundle center
	setTool("multipoint");
	// run("Show Overlay");
	waitingMessage = "Select heels ...";
	msg = "1. Please Select R cell heels and bundle center. \n Press 't' to add to ROI. \n Press ok when done.";
	waitForUser(waitingMessage, msg);
	wait(1000);
	// run("Add Selection...");
	// run("Show Overlay");

	// targets?
	setTool("point");
	// roiManager("Show All");
	waitingMessage = "Update ...";
	msg = "2. Please select targets for original bundle as well as nearby targets. \n Press 't' to add to ROI. \n Press ok when done.";
	waitForUser(waitingMessage, msg);
	wait(1000);

	// update?
	setTool("point");
	run("Arrow Tool...", "width=2 size=3 color=Cyan style=Filled");
	// roiManager("Show All");
	waitingMessage = "Update ...";
	msg = "3. Please make any adjustments: \n Click 'update' on the ROI manager to update the ROI. \n Press ok when done.";
	waitForUser(waitingMessage, msg);
	wait(1000);
}


function saveBundleDetailsInfo(inputDirectory, outputMasterDirectory, fileName) {
	// parameters
	myDateArray = defineDateAndTime();
	dateString = myDateArray[0];
	timeString = myDateArray[1];

	// check and/or make output directory.
	outputDirectory = checkDirectory(outputMasterDirectory + File.separator + "Recording");
	outputDirectory = checkDirectory(outputDirectory + File.separator +  "Final");

	// save heel and target coordinates info into csv file.
	selectWindow(fileName);
	csvName = getFileName(substring(fileName,0,indexOf(fileName, ".tif")), "ROI", dateString, timeString, ".csv");
	print(csvName);

	waitingMessage = "Getting ROIs to Measure...";
	msg = "Please select all ROIs and measure. ";
	waitForUser(waitingMessage, msg);
	wait(2000);

	csvpath = outputDirectory + File.separator + csvName;
	saveAs("Results", csvpath);


	// save heel and target ROIs to zip file.
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
	ROIDirectory = checkDirectory(outputDirectory + File.separator + "ROI");
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
		exit("Unable to create directory: " + customDir;
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
