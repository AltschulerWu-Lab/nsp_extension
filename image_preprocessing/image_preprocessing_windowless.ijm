/*
* @Author: Weiyue Ji
* @Date:   2020-09-08 14:41:00
* @Last Modified by:   Weiyue Ji
* @Last Modified time: 2020-09-08 14:41:16
*/

/*
Function: change format of the files from ND2 to tif and perform background subtraction. Modified to run on Wynton cluster.
Input:
- string argument from batch file:'folder_name1, folder_name2|folter_type1, folder_type2|log_file_name.txt'
	* include: name of the folders to process (seperated by comma), type of the imaging aquisition method (Jobs or individual ND-acquisitions, also seperated by comma), name of the log file.
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


input_sup_dir = "/awlab/projects/2019_09_NSP_Extension/data/Imaging_Data_Original/Nikon_A1RsiConfocalImages" + File.separator;
output_sup_dir = "/awlab/projects/2019_09_NSP_Extension/data/Format_converted" + File.separator;
log_folder_dir = "/awlab/projects/2019_09_NSP_Extension/code/NSP_extension/imageJ_macros/image_preprocessing/logs" + File.separator;

/* get input folders and folder inexes*/
// for folder index, ND-acquisition = 1; Jobs = 2;

input_folders = split(input_str[0], ', ');
print(input_folders[0]);
folder_indexes = split(input_str[1], ', ');
print(folder_indexes[0]);
log_name = input_str[2]
log_file_dir = File.open(log_folder_dir+log_name);

/*processing folders*/
if(input_folders.length == folder_indexes.length){
	process_folders(input_folders, folder_indexes, suffix, log_file_dir);
	print(log_file_dir, "=============== end ===============");
}
else{
	print("ERROR! Length of folder names and indexes doesn't match!")
}

/***************************************** end of main() **************************************/


/******************************************** functions ***************************************/
/*process_folders function: getting information from indiviual folders*/
function process_folders(input_folders, folder_indexes, suffix, log_file_dir) {
	for (i = 0; i < (input_folders.length); i++){
		print(log_file_dir, "===============" + input_folders[i] + "===============");
		print("===============" + input_folders[i] + "===============");
		// individual input and output folder
		input_dir = input_sup_dir + input_folders[i] + File.separator;
		output_dir = output_sup_dir + input_folders[i] + File.separator;
		index = folder_indexes[i];
		print(log_file_dir, "input_dir: " + input_dir);
		print(log_file_dir, "output_dir: " + output_dir);
		print(log_file_dir, "index: " + index);

		// Process individual folders
		process_folder(input_dir, output_dir, input_folders[i], index, log_file_dir);
	}
}


/*process_folder function: Processing individual folders*/
function process_folder(input_dir, output_dir, exp_name, index, log_file_dir) {
	list = get_file_list(input_dir);
	for (iL = 0; iL < list.length; iL ++) {
		print(log_file_dir, list[iL]);
	}
	for (iL = 0; iL < list.length; iL ++) {
		// If directory, process folder
		if(File.isDirectory(input_dir + list[iL])) {
			process_folder("" + input_dir + list[iL], output_dir, exp_name, index, log_file_dir);
		}
		// If ".nd2" file, proceed to process nd2 files.
		if(endsWith(list[iL], suffix)) {
			process_nd2(input_dir, output_dir, suffix, list[iL], exp_name, index, log_file_dir);
		}
	}
}



/*process_nd2 function: processing nd2 to get compositioned tifs*/
function process_nd2(input_dir, output_dir, suffix, file_name, exp_name, index, log_file_dir) {

	/// get output image's name
	output_name = get_output_name(file_name, suffix, exp_name, index, output_dir + File.separator);
	print(log_file_dir, "### " + output_name + " start! ###");
	print("### " + output_name + " start! ###");
	
	/// get output directory
	if (!File.exists(output_dir))
		File.makeDirectory(output_dir);
	if (!File.exists(output_dir))
		exit("Unable to create directory: " + output_dir);

	/// open ND2
	run("Bio-Formats Macro Extensions");
	setBatchMode(true);
	Ext.openImagePlus(input_dir + file_name);
	
	/// get channel counts
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

	/// run background substratcion
	run("Subtract Background...", "rolling=100 disable");

	/// save converted tif file to output folder
	saveAs("Tiff", output_dir + output_name);

	/// close images.
	close();

	/// free unused memory
	freeMemory();
	print(log_file_dir, "### " + output_name + " done! ###");
	print("### " + output_name + " done! ###");
}


/* get_output_name function: get non-redundant name for each composite */
function get_output_name(file_name, suffix, exp_name, index, output_path) {
	
	// data collected using Jobs
	if(index == "1") {
		ID1 = indexOf(file_name, "Point") + 5;
		ID2 = indexOf(file_name, "_Channel");
		if(endsWith(exp_name, "eye")) {
			id_exp = indexOf(exp_name, "_eye");
			suffix_exp = "_eye";
		}
		else if (endsWith(exp_name, "lamina")) {
			id_exp = indexOf(exp_name, "_lamina");
			suffix_exp = "_lamina";
		}

		// loop to give unique name to each file
		is_exist = 1;
		i = 0;
		while(is_exist) {
			// get name
			if(i == 0){
				output_name = substring(exp_name, 0, id_exp) + "_" + substring(file_name, ID1, ID2) + suffix_exp + ".tif";
			}
			else{
				output_name = substring(exp_name, 0, id_exp) + "_" + substring(file_name, ID1, ID2) + suffix_exp + "_" + i + ".tif";
			}

			//check if name exist already
			if(!File.exists(output_path + output_name))
				is_exist = 0;
			i ++;
		}
	}

	// data collected using ND acquisition
	else {
		//ID1 = indexOf(file_name, "00");
		ID2 = indexOf(file_name, suffix);
		//if(ID1 != -1)
		//	tempName = substring(file_name, 0, ID1) ;
		//else
		tempName = substring(file_name, 0, ID2);
		// loop to give unique name to each file
		is_exist = 1;
		i = 0;
		while(is_exist) {
			output_name = tempName + "_" + i + ".tif";
			if(!File.exists(output_path + output_name))
				is_exist = 0;
			i ++;
		}
	}

	return output_name;
}


/* function to free unused memory */
function freeMemory() {
	run("Collect Garbage");
}
