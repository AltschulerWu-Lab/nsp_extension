/*
* @Author: sf942274
* @Date:   2019-11-15 17:42:26
* @Last Modified by:   sf942274
* @Last Modified time: 2020-01-29 09:58:01
*/

 /*
Function: change format of the files from ND2 to tif. Modified to run on Wynton cluster.
Input:
- string argument from batch file:'folder_name1, folder_name2|folter_type1, folder_type2|log_file_name.txt'
	* Naming convensions: for Jobs, folder name nees to end with either "eye" or "lamina". For ND-acquisitions, no particular rules.
	* Folder type: "1" for Jobs, "2" for ND-acquisitions
Output:
- tif images stored at corresponding positions.
- log txt file
 */

/********************************************* main() ****************************************/
run("Close All");

/*Define input and output paths*/
input_str = split(getArgument(), '|');
suffix = ".nd2";

inputSupPath = "/awlab/projects/2019_09_NSP_Extension/data/Imaging_Data_Original/Nikon_A1RsiConfocalImages" + File.separator;
outputSupPath = "/awlab/projects/2019_09_NSP_Extension/data/Format_converted" + File.separator;
logSupPath = "/awlab/projects/2019_09_NSP_Extension/code/NSP_codes/image_preprocessing/logs" + File.separator;

/* get input folders and folder inexes*/
// for folder index, ND-acquisition = 1; Jobs = 2;

inputFolders = split(input_str[0], ', ');
print(inputFolders[0]);
folderIndexes = split(input_str[1], ', ');
print(folderIndexes[0]);
log_name = input_str[2]
logFile = File.open(logSupPath+log_name);

/*processing folders*/
if(inputFolders.length == folderIndexes.length){
	processFolders(inputFolders, folderIndexes, suffix, logFile);
	print(logFile, "=============== end ===============");
}
else{
	print("ERROR! Length of folder names and indexes doesn't match!")
}

/***************************************** end of main() **************************************/


/******************************************** functions ***************************************/
/*processFolders function: getting information from indiviual folders*/
function processFolders(inputFolders, folderIndexes, suffix, logFile) {
	for (i = 0; i < (inputFolders.length); i++){
		print(logFile, "===============" + inputFolders[i] + "===============");
		print("===============" + inputFolders[i] + "===============");
		// individual input and output folder
		inputPath = inputSupPath + inputFolders[i] + File.separator;
		outputDir = outputSupPath + inputFolders[i];
		index = folderIndexes[i];
		print(logFile, "inputdir: " + inputPath);
		print(logFile, "outputdir: " + outputDir);
		print(logFile, "index: " + index);

		// Process individual folders
		processFolder(inputPath, outputDir, inputFolders[i], index, logFile);
	}
}


/*processFolder function: Processing individual folders*/
function processFolder(inputPath, outputDir, expName, index, logFile) {
	list = getFileList(inputPath);
	for (iL = 0; iL < list.length; iL ++) {
		print(logFile, list[iL]);
	}
	for (iL = 0; iL < list.length; iL ++) {
		if(File.isDirectory(inputPath + list[iL])) {
			processFolder("" + inputPath + list[iL], outputDir, expName, index, logFile);
		}
		if(endsWith(list[iL], suffix)) {
			processND2(inputPath, outputDir, suffix, list[iL], expName, index, logFile);
		}
	}
}



/*processND2 function: processing nd2 to get compositioned tifs*/
function processND2(inputPath, outputDir, suffix, fileName, expName, index, logFile) {

	/// get output image's name
	outputName = getOutputName(fileName, suffix, expName, index, outputDir + File.separator);
	print(logFile, "### " + outputName + " start! ###");
	print("### " + outputName + " start! ###");
	
	/// get output directory
	if (!File.exists(outputDir))
		File.makeDirectory(outputDir);
	if (!File.exists(outputDir))
		exit("Unable to create directory: " + outputDir);

	/// open ND2
	run("Bio-Formats Macro Extensions");
	setBatchMode(true);
	Ext.openImagePlus(inputPath + fileName);
	
	/// get channel countS
	getDimensions(width, height, channels, slices, frames);
	numOfChannels = channels;

	/// if there's multiple channels: merge and save.
	if (numOfChannels > 1) {
		// split channels
		run("Split Channels");

		// get all channel names and merge
		numOfImages = nImages(); 												// get number of currently opened images.

		mergeCommand = '';
		if(numOfImages < 4) { 													// No DAPI channel
			for (imageID = 1; imageID <= numOfImages; imageID ++) {
				selectImage(imageID);
				imageTitle = getTitle();
				run("Enhance Contrast", "saturated=0.35"); 						// auto-set brightness and contrast
				if(indexOf(imageTitle, "C1") != -1) 							// green channel (GFP) --> green
					mergeCommand = mergeCommand + "c2=[" + imageTitle + "] ";
				else if (indexOf(imageTitle, "C2") != -1) 						// red channel (RFP) -- red
					mergeCommand = mergeCommand + "c1=[" + imageTitle + "] ";
				else if (indexOf(imageTitle, "C3") != -1) { 					// Cy5 channel (24B10) --> grey
					run("Grays");
					mergeCommand = mergeCommand + "c3=[" + imageTitle + "] ";
				}
			}
		}
		else if (numOfImages == 4) { 											// with DAPI channel
			for (imageID = 1; imageID <= numOfImages; imageID ++) {
				selectImage(imageID);
				imageTitle = getTitle();
				run("Enhance Contrast", "saturated=0.35");						// auto-set brightness and contrast
				if(indexOf(imageTitle, "C1") != -1) {							// cyan channel (L-cells) --> blue
					run("Blue");
					mergeCommand = mergeCommand + "c4=[" + imageTitle + "] ";
				} 							
				else if (indexOf(imageTitle, "C2") != -1) 						// green channel (GFP) --> green
					mergeCommand = mergeCommand + "c2=[" + imageTitle + "] ";
				else if (indexOf(imageTitle, "C3") != -1)						// red channel (RFP) -- red
					mergeCommand = mergeCommand + "c1=[" + imageTitle + "] ";
				else if (indexOf(imageTitle, "C4") != -1) { 					// Cy5 channel (24B10) --> gray
					run("Grays");
					mergeCommand = mergeCommand + "c3=[" + imageTitle + "] ";
				}
			}
		}
		
		mergeCommand = mergeCommand + "create";

		run("Merge Channels...", mergeCommand);
	}


	/// transform if image is vertical.
	if (width < height) {
		run("Rotate 90 Degrees Left");
	}

	/// save converted tif file to output folder
	saveAs("Tiff", outputDir + File.separator + outputName);

	/// close images.
	close();

	/// free unused memory
	freeMemory();
	print(logFile, "### " + outputName + " done! ###");
	print("### " + outputName + " done! ###");
}


/* getOutputName function: get non-redundant name for each composite */
function getOutputName(fileName, suffix, expName, index, outputPath) {
	// data collected using Jobs
	if(index == "1") {
		ID1 = indexOf(fileName, "Point") + 5;
		ID2 = indexOf(fileName, "_Channel");
		if(endsWith(expName, "eye")) {
			idExp = indexOf(expName, "_eye");
			suffixExp = "_eye";
		}
		else if (endsWith(expName, "lamina")) {
			idExp = indexOf(expName, "_lamina");
			suffixExp = "_lamina";
		}

		// loop to give unique name to each file
		isExist = 1;
		i = 0;
		while(isExist) {
			// get name
			if(i == 0){
				outputName = substring(expName, 0, idExp) + "_" + substring(fileName, ID1, ID2) + suffixExp + ".tif";
			}
			else{
				outputName = substring(expName, 0, idExp) + "_" + substring(fileName, ID1, ID2) + suffixExp + "_" + i + ".tif";
			}

			//check if name exist already
			if(!File.exists(outputPath + outputName))
				isExist = 0;
			i ++;
		}
	}

	// data collected using ND acquisition
	else {
		//ID1 = indexOf(fileName, "00");
		ID2 = indexOf(fileName, suffix);
		//if(ID1 != -1)
		//	tempName = substring(fileName, 0, ID1) ;
		//else
		tempName = substring(fileName, 0, ID2);
		// loop to give unique name to each file
		isExist = 1;
		i = 0;
		while(isExist) {
			outputName = tempName + "_" + i + ".tif";
			if(!File.exists(outputPath + outputName))
				isExist = 0;
			i ++;
		}
	}

	return outputName;
}


/* function to free unused memory */
function freeMemory() {
	run("Collect Garbage");
}
