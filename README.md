# Digit Recognition (Speech) with Hidden Markov Model

Input: Contains all the inputs (all utterances of all digits)
Logs: Intermediate files (all utterances of all digits) 10 folders corresponding to 10 digits (25 folders in each digit folder)

There are three parts in this project (three difference Visual Studio Projects) and has to be run in the below order:
1. Building Universe (just concatenating the files) - BUILD_UNIVERSE
2. Building Centroids of the Universe (Using LBG Algorithm) - BUILD_CENTROIDS
2. Training for all utterances of all digits (getting averaged lambda for every digit) - DIGIT_TRAINING
3. Digit Recognition of the test digit

1. Building Universe (Project Name:BUILD_UNIVERSE)
	-this takes 20 utterances of every digit and builds the universe.
	-Give a proper path to this string "universe_file_str" in the cpp source file, of the universe.txt file to be constructed.
	
2. Building Centroids of the Universe (Using LBG Algorithm) (Project Name:BUILD_CENTROIDS)
	 -Centroids of the universe are created using LBG Algorithm (32 Clusters)
	
3. Training the model for all utterances of all digits (Project Name: DIGIT_TRAINING)
	-This trains the model and builds averaged lambda of all utterances for all the digits.
	-"centroids_file_str" - this should be the path of of the centroids file from "building Centroids"
	- give only till the directory path for these strings :
		1.input_file_old (till input folder)
		2. file_old (till logs folder)
	-give full file paths for these strings:
		1.a_file_str - for 'a' matrix file
		2.b_file_str - 'b' matrix file
		3.pi_file_str - 'pi' matrix file
		
4. Digit Recognition (Project Name: DIGIT_RECOGNITION)
	Give proper string value for all the strings.