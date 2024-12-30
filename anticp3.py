# Import necessary libraries
import argparse
import joblib
import os
import sys
import pandas as pd
import numpy as np
import glob
from Bio import SeqIO
import shutil
from tqdm import tqdm
import time
import requests
import zipfile

# ------------------------------- Downloading Swissprot Database from the server ----------------------------- #
nf_path = os.path.dirname(os.path.abspath(__file__))
swissprot_dir = os.path.join(nf_path, 'blast', 'data')
zip_file_path = os.path.join(nf_path, 'swissprot.zip')
if not os.path.exists(swissprot_dir):
    print("SwissProt database not found. Downloading...")
    try:
        # Download the SwissProt ZIP file
        response = requests.get('https://webs.iiitd.edu.in/raghava/anticp3/swissprot.zip', stream=True)
        if response.status_code == 200:
            with open(zip_file_path, 'wb') as f:
                f.write(response.content)
            print("Download complete. Extracting...")
                
            # Extract the ZIP file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(swissprot_dir)
                
            print("Database extracted successfully.")
            os.remove(zip_file_path)  # Remove the ZIP file after extraction
        else:
            raise Exception(f"Failed to download the database. HTTP Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error during setup: {e}")
else:
    pass
# ----------------------------------------- Setup End ----------------------------------------------------- #

def print_banner():
    banner = """
###################################################################################
#                                                                                 #
#                               Welcome to AntiCP3                                #
#                                                                                 #
#   AntiCP3 is a tool for predicting anticancer and non-anticancer proteins       #
#   from their primary sequence. Developed by Prof. G. P. S. Raghava's team.      #
#   Please cite: AntiCP3; available at https://webs.iiitd.edu.in/raghava/anticp3  #
#                                                                                 #
###################################################################################
"""
    print(banner)
    
def simulate_task(task_name, duration=2):
    """Simulates a task for demo purposes."""
    time.sleep(duration)  # Simulate task duration
    
def check(fasta_input):
    for record in SeqIO.parse(fasta_input, "fasta"):
        if len(record.seq) < 52:
            print(f"Error: Sequence {record.id} is too short. Please use proteins with length more than 50.")
            print(f"Please use AntiCP2 for shorter peptides. Exiting..")
            return False
    return True

std = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def fasta_to_dataframe(fasta_file):
    """
    Converts a FASTA file into a pandas DataFrame.

    Args:
        fasta_file (str): Path to the FASTA file containing sequence data.

    Returns:
        pandas.DataFrame: A DataFrame with two columns:
            - 'ID': The sequence identifiers from the FASTA file.
            - 'Sequence': The corresponding sequences.
    """
    # Initialize a dictionary to store sequence IDs and sequences.
    sequences = {'ID': [], 'Sequence': []}
    # Parse the FASTA file and extract sequence IDs and sequences.
    for record in SeqIO.parse(fasta_file, 'fasta'):
        # Remove the '>' prefix from the sequence ID, if present.
        sequence_id = record.id.lstrip('>')
        sequences['ID'].append(sequence_id)
        sequences['Sequence'].append(str(record.seq))
    # Convert the dictionary into a pandas DataFrame.
    df = pd.DataFrame(sequences)
    # Return the DataFrame containing the parsed sequences.
    return df

def aac_comp(file, out="AAC.csv"):
    """
    Calculates the amino acid composition (AAC) of sequences in a FASTA file 
    and writes the results to a CSV file.

    Args:
        file (str): Path to the input FASTA file containing protein sequences.
        out (str, optional): Path to the output CSV file. Defaults to "AAC.csv".

    Returns:
        None: The results are written directly to the specified output file.
    """
    # Extract the file name and extension of the input file.
    filename, file_extension = os.path.splitext(file)
    # Open the output file for writing.
    f = open(out, 'w')
    # Redirect standard output to the output file.
    sys.stdout = f
    # Convert the FASTA file into a pandas DataFrame.
    df = fasta_to_dataframe(file)
    # Extract the 'Sequence' column containing the sequences.
    zz = df['Sequence']
    # Print the header row for the CSV file with amino acid labels.
    print("AAC_A,AAC_C,AAC_D,AAC_E,AAC_F,AAC_G,AAC_H,AAC_I,AAC_K,AAC_L,AAC_M,AAC_N,AAC_P,AAC_Q,AAC_R,AAC_S,AAC_T,AAC_V,AAC_W,AAC_Y,")
    # Iterate over each sequence in the DataFrame.
    for j in zz:
        # Iterate over each standard amino acid.
        for i in std:
            count = 0
            # Count occurrences of the current amino acid in the sequence.
            for k in j:
                temp1 = k
                if temp1 == i:
                    count += 1
                # Calculate the composition as a percentage of the sequence length.
                composition = (count / len(j)) * 100
            # Print the composition percentage formatted to two decimal places.
            print("%.2f" % composition, end=",")
        print("")
    # Close the output file and restore standard output.
    f.close()
    sys.stdout = sys.__stdout__
    
def gen_pssm(fasta_input, pssm_output_dir):
    """
    Generate Position-Specific Scoring Matrix (PSSM) files using PSIBLAST for protein sequences.

    Args:
        fasta_input (str): Path to a single FASTA file or a directory containing FASTA files.
        pssm_output_dir (str): Directory where the generated PSSM files will be saved.

    Returns:
        None: The function generates PSSM files and saves them to the specified output directory.
    """

    # Ensure necessary directories exist for storing PSSM output.
    os.makedirs(os.path.join(pssm_output_dir, 'pssm_raw1'), exist_ok=True)
    os.makedirs(os.path.join(pssm_output_dir, 'pssm_raw'), exist_ok=True)

    # Handle input: single FASTA file or a directory containing multiple FASTA files.
    if os.path.isfile(fasta_input):
        fasta_files = [fasta_input] # Single FASTA file
    elif os.path.isdir(fasta_input):
        fasta_files = glob.glob(f"{fasta_input}/*.fasta") # All FASTA files in the directory
    else:
        raise FileNotFoundError(f"{fasta_input} is neither a file nor a directory.")
    # Iterate over each FASTA file for processing
    for fasta_file in fasta_files:
        # Check and process each sequence in the FASTA file (handles multi-FASTA).
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequence_id = record.id  # Sequence header
            sequence_file = os.path.join(pssm_output_dir, f"{sequence_id}.fasta")
            
            # Save individual sequence to a temporary FASTA file.
            with open(sequence_file, "w") as seq_file:
                seq_file.write(f">{record.id}\n{record.seq}\n")

            # Define output paths for PSSM and homolog files.
            output_pssm_raw1 = os.path.join(pssm_output_dir, 'pssm_raw1', f"{sequence_id}.cptpssm")
            output_pssm_raw = os.path.join(pssm_output_dir, 'pssm_raw', f"{sequence_id}.pssm")
            homologs_output = os.path.join(pssm_output_dir, 'pssm_raw1', f"{sequence_id}.homologs")

            # Construct the PSIBLAST command.
            cmd = f"./ncbi_blast_2.15/bin/psiblast -out {homologs_output} -outfmt 7 -query {sequence_file} -db ./blast/data/swissprot " \
                  f"-evalue 0.001 -word_size 3 -max_target_seqs 6000 -num_threads 10 -gapopen 11 " \
                  f"-gapextend 1 -matrix BLOSUM62 -comp_based_stats T -num_iterations 3 " \
                  f"-out_pssm {output_pssm_raw1} -out_ascii_pssm {output_pssm_raw} 2>/dev/null"

            result = os.system(cmd)  # Execute the command
            # Check and log success or failure of PSSM generation.
            if result != 0:
                print(f"Error generating PSSM for {sequence_id}. Check the input or PSIBLAST environment.")
            else:
                pass
            
def generate_pssm_features(pssm_dir, fasta_file, possum_script, header_handler_script, output_dir):
    """
    Generate PSSM-based features using POSSUM and adjust headers with a header handler script.

    Args:
        pssm_dir (str): Directory containing raw PSSM files.
        fasta_file (str): Path to the input FASTA file.
        possum_script (str): Hardcoded path to the POSSUM script for feature extraction.
        header_handler_script (str): Hardcoded path to the headerHandler.py script for adjusting feature headers.
        output_dir (str): Directory where the intermediate and final output files will be saved.

    Returns:
        str: Path to the final PSSM features CSV file.
    """
    # Paths for POSSUM input and output files
    possum_input = os.path.join(output_dir, "pssm_input.fasta")
    temp_output = os.path.join(output_dir, "pssm_temp.csv")
    possum_output = os.path.join(output_dir, "pssm_features.csv")

    # Step 1: Prepare the input FASTA file for POSSUM.
    df = fasta_to_dataframe(fasta_file)  # Convert FASTA file into a DataFrame.
    df['ID'] = ">" + df['ID'] # Add '>' to sequence IDs for FASTA format.
    df[['ID', 'Sequence']].to_csv(possum_input, index=False, header=False, sep="\n") # Save in FASTA format.

    # Ensure the POSSUM input file exists.
    if not os.path.isfile(possum_input):
        raise FileNotFoundError(f"{possum_input} is not a valid file. Ensure it's an actual FASTA file.")

    # Step 2: Run the POSSUM script for feature extraction.
    possum_cmd = f"python3 {possum_script} -i {possum_input} -o {temp_output} -t pssm_composition -p {pssm_dir}"
    if os.system(possum_cmd) != 0:
        raise RuntimeError(f"POSSUM script failed for input {possum_input}. Check the script and input files.")

    # Step 3: Add headers using the header handler script.
    header_handler_cmd = f"python3 {header_handler_script} -i {temp_output} -o {possum_output} -p pssm_"
    if os.system(header_handler_cmd) != 0:
        raise RuntimeError(f"Header adjustment script failed for {temp_output}.")

    if not os.path.exists(possum_output):
        raise FileNotFoundError(f"Final PSSM features output {possum_output} not found.")

    return possum_output

def replace_pssm_headers(pssm_file, new_headers):
    """
    Replace the headers of the PSSM features CSV with a list of new headers.
    :param pssm_file: Path to the PSSM features CSV file.
    :param new_headers: List of new headers to replace the existing headers.
    """
    # Read the PSSM file
    df = pd.read_csv(pssm_file, header=0)

    # Validate header length
    if len(new_headers) != df.shape[1]:
        raise ValueError(f"Number of new headers ({len(new_headers)}) does not match number of columns ({df.shape[1]}) in the PSSM file.")

    # Replace the headers
    df.columns = new_headers
    df.to_csv(pssm_file, index=False)
    
def feature_selection(merged_file, feature_list, output_file):
    merged_df = pd.read_csv(merged_file)

    # Ensure only the required features are retained
    subset_df = merged_df[[col for col in merged_df.columns if col in feature_list]]

    # Ensure order matches the feature list
    subset_df = subset_df[feature_list]

    # Save the subset DataFrame to a new CSV file
    subset_df.to_csv(output_file, index=False)

def merge_aac_pssm_features(aac_file, pssm_file, output_file):
    """
    Merge AAC features and updated PSSM features into a single CSV file.
    :param aac_file: Path to the AAC features CSV file.
    :param pssm_file: Path to the updated PSSM features CSV file.
    :param output_file: Path to save the concatenated CSV file.
    """
    # Read AAC and PSSM features
    aac_df = pd.read_csv(aac_file).dropna(how="all", axis=1)
    pssm_df = pd.read_csv(pssm_file)

    # Concatenate the DataFrames
    concatenated_df = pd.concat([aac_df, pssm_df], axis=1)
    concatenated_df.to_csv(output_file, index=False)
    
def aac_pssm_predict(subset_features_file, input_fasta_file, output_file, threshold):
    """
    Predict anticancer proteins using AAC and PSSM features and our pre-trained machine learning model.

    Args:
        subset_features_file (str): Path to the CSV file containing the subset of features for prediction.
        input_fasta_file (str): Path to the input FASTA file with sequences to be classified.
        output_file (str): Path where the prediction results will be saved as a CSV file.
        threshold (float): Probability threshold for classification. Scores above this are classified as 'Anticancer'.

    Returns:
        None: Saves predictions directly to the specified output file.
    """
    # Load the subset features from the CSV file
    subset_df = pd.read_csv(subset_features_file)
    
    # Load the input FASTA file and create a DataFrame
    input_df = fasta_to_dataframe(input_fasta_file)  # Assumes this function handles multifasta files correctly

    # Merge the input DataFrame with the subset features DataFrame on 'ID' column
    # This will add the 'ID' and 'Sequence' columns from the input FASTA to the subset features
    merged_df = pd.concat([input_df[['ID', 'Sequence']], subset_df], axis=1)

    # Load the saved model
    model_file = './model/extra_trees_model_AAC+PSSM_SelectedFeatures.pkl'
    with open(model_file, 'rb') as f:
        model = joblib.load(f)
    
    # Make predictions using the features (excluding 'ID' and 'Sequence')
    features = merged_df.drop(['ID', 'Sequence'], axis=1)
    predictions = model.predict_proba(features)[:, 1]
    
    # Apply the threshold to convert probabilities to binary labels
    binary_predictions = (predictions >= threshold).astype(int)
    
    # Add predictions to the DataFrame
    merged_df['ML_Score'] = predictions
    merged_df['Prediction'] = binary_predictions
    merged_df['Prediction_Label'] = np.where(binary_predictions == 1, 'Anticancer', 'Non-Anticancer')
    
    # Save the DataFrame with predictions to a CSV file
    merged_df[['ID', 'Sequence', 'ML_Score', 'Prediction_Label']].to_csv(output_file, index=False)
    
def run_blast_and_integrate(fasta_input, blast_db, output_file, output_dir):
    """
    Run BLASTP on input sequences and integrate the results with ML model scores to calculate a hybrid score.

    Args:
        fasta_input (str): Path to the input FASTA file containing sequences for BLAST analysis.
        blast_db (str): Path to the BLAST database for sequence alignment.
        output_csv (str): Path to the CSV file containing ML model scores (output from Model 1).
        output_dir (str): Directory where BLAST results and final outputs will be saved.

    Returns:
        None: Saves BLAST results, integrated scores, and predictions to specified output files.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define BLAST output file
    blast_output_file = os.path.join(output_dir, "results_evalue_1e-10.txt")

    # Run BLASTP
    blast_cmd = f"./ncbi_blast_2.15/bin/blastp -query {fasta_input} -db {blast_db} -out {blast_output_file} " \
                f"-evalue 1e-10 -outfmt 6"
    result = os.system(blast_cmd)

    if result != 0:
        raise RuntimeError("BLASTP execution failed. Check BLAST setup and inputs.")

    # Read BLAST results
    blast_columns = ['name', 'hit', 'identity', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9']
    blast_df = pd.read_csv(blast_output_file, sep="\t", names=blast_columns)
    blast_df = blast_df[['name', 'hit']]

    # Preprocess input FASTA file
    headers, sequences = [], []
    for record in SeqIO.parse(fasta_input, "fasta"):
        headers.append(record.id)
        sequences.append(str(record.seq))
    fasta_df = pd.DataFrame({'name': headers, 'Sequence': sequences})

    # Assign BLAST Scores
    blast_scores = []
    for name in fasta_df['name']:
        # Check if the name exists in blast_df
        match = blast_df[blast_df['name'] == name]
        if not match.empty:  # Check if the match is non-empty
            hit_value = match['hit'].iloc[0]  # Extract the 'hit' value for the matched row
            if hit_value.split('_')[-1] == '1':
                blast_scores.append(0.5)
            elif hit_value.split('_')[-1] == '0':
                blast_scores.append(-0.5)
            else:
                blast_scores.append(0)  # Default for unexpected cases
        else:
            blast_scores.append(0)  # No match found

    # Check if the lengths match
    assert len(blast_scores) == len(fasta_df), "Mismatch between blast_scores length and fasta_df length"

    # Add the scores to the DataFrame
    fasta_df['BLAST_Score'] = blast_scores

    # Integrate with ML model scores
    ml_df = pd.read_csv(output_file)  # Output from Model 1
    merged_df = pd.merge(ml_df, fasta_df[['name', 'BLAST_Score']], left_on='ID', right_on='name', how='left')

    # Calculate Hybrid Score
    merged_df['Hybrid Score'] = merged_df['ML_Score'] + merged_df['BLAST_Score']
    merged_df['Hybrid Score'] = merged_df['Hybrid Score'].clip(0, 1)  # Clamp between 0 and 1

    # Final Prediction
    threshold = args.threshold
    merged_df['Prediction_1'] = (merged_df['Hybrid Score'] > threshold).astype(int)
    merged_df['Prediction'] = merged_df['Hybrid Score'].apply(
        lambda x: "Anticancer" if x > threshold else "Non-Anticancer"
    )
    merged_df = merged_df[['ID', 'Sequence', 'ML_Score', 'BLAST_Score', 'Hybrid Score', 'Prediction']]

    # Save final results
    merged_df.to_csv(output_file, index=False)
    
def cleanup(working_dir, files, folders):
    working_dir = args.workingdir
    for file_path in files:
        full_file_path = os.path.join(working_dir, file_path)
        if os.path.isfile(full_file_path):
            try:
                os.remove(full_file_path)
            except Exception as e:
                pass
            
    for folder_path in folders:
        full_folder_path = os.path.join(working_dir, folder_path)
        if os.path.isdir(full_folder_path):
            try:
                shutil.rmtree(full_folder_path)
            except Exception as e:
                pass 

if __name__ == "__main__":
    print_banner()
    print(f"Run Example - \npython3 anticp3.py -i ./example/example_input.fasta -m 2 -t 0.45 -o example_output.csv -wd ./example/\n")
    parser = argparse.ArgumentParser(description="Welcome to AntiCP3. Please provide following arguments.")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file or directory containing FASTA files.")
    parser.add_argument("-m", "--model", type=int, default=2, choices=[1, 2], help="Model choice (1: PSSM + AAC based ExtraTrees, 2: Hybrid (PSSM + AAC + BLAST); default 2).")
    parser.add_argument("-o", "--output", default="out.csv", help="Final output file for predictions.")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Threshold for classification (default: 0.5).")
    parser.add_argument("-wd", "--workingdir", default=os.getcwd(), help="Working directory for output files.")
    args = parser.parse_args()
    
    if not check(args.input):
        exit()
    
    print("\n=== Summary of Parameters ===\n")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("\n=== End of Parameters ===\n")
    steps = [
        "Generate AAC Features",
        "Generate PSSM Files",
        "Generate PSSM-Based Features",
        "Loading the Model",
        "Making Predictions",
    ]
    
    with tqdm(total=len(steps), desc="AntiCP3 Running ..", bar_format="{l_bar}{bar} [Elapsed: {elapsed} | Remaining: {remaining}]") as progress_bar:

        if args.model == 1:
            print("Starting Predictions ...")
            # Define paths
            working_dir = args.workingdir
            aac_output_path = os.path.join(args.workingdir, "AAC.csv")
            pssm_output_dir = os.path.join(args.workingdir, "PSSM")
            pssm_features_output_dir = os.path.join(args.workingdir, "PSSM_Features")
            all_features_output = os.path.join(working_dir, "all_features.csv")
            selected_features_output = os.path.join(working_dir, "selectedFeat.csv")
            prediction_output = os.path.join(working_dir, args.output)
            final_output_path = os.path.join(working_dir, args.output)
            
            # Hardcoded POSSUM script path
            possum_script_path = "./possum/possum.py"
            header_handler_script_path = "./possum/headerHandler.py"

            # Ensure directories exist
            os.makedirs(pssm_features_output_dir, exist_ok=True)

            # Step 1: Generate AAC features
            aac_comp(args.input, aac_output_path)
            simulate_task("Generating Amino Acid Composition")
            progress_bar.update(1)

            # Step 2: Generate PSSM features
            gen_pssm(args.input, pssm_output_dir)
            simulate_task("Generate PSSM Files")
            progress_bar.update(1)

            # Step 3: Generate PSSM-based features
            pssm_features_path = generate_pssm_features(
                pssm_dir=os.path.join(pssm_output_dir, "pssm_raw"),
                fasta_file=args.input,
                possum_script=possum_script_path,
                header_handler_script=header_handler_script_path,
                output_dir=pssm_features_output_dir,
            )
            simulate_task("Generate PSSM-Based Features")
            progress_bar.update(1)
            
            # Step 4: Replace headers and concatenate AAC and PSSM features
            new_pssm_headers = [
                'A_A', 'A_R', 'A_N', 'A_D', 'A_C', 'A_Q', 'A_E', 'A_G', 'A_H', 'A_I', 'A_L', 'A_K', 'A_M', 'A_F', 'A_P', 'A_S', 'A_T', 'A_W', 'A_Y', 'A_V', 'R_A', 'R_R', 'R_N', 'R_D', 'R_C', 'R_Q', 'R_E', 'R_G', 'R_H', 'R_I', 'R_L', 'R_K', 'R_M', 'R_F', 'R_P', 'R_S', 'R_T', 'R_W', 'R_Y', 'R_V', 'N_A', 'N_R', 'N_N', 'N_D', 'N_C', 'N_Q', 'N_E', 'N_G', 'N_H', 'N_I', 'N_L', 'N_K', 'N_M', 'N_F', 'N_P', 'N_S', 'N_T', 'N_W', 'N_Y', 'N_V', 'D_A', 'D_R', 'D_N', 'D_D', 'D_C', 'D_Q', 'D_E', 'D_G', 'D_H', 'D_I', 'D_L', 'D_K', 'D_M', 'D_F', 'D_P', 'D_S', 'D_T', 'D_W', 'D_Y', 'D_V', 'C_A', 'C_R', 'C_N', 'C_D', 'C_C', 'C_Q', 'C_E', 'C_G', 'C_H', 'C_I', 'C_L', 'C_K', 'C_M', 'C_F', 'C_P', 'C_S', 'C_T', 'C_W', 'C_Y', 'C_V', 'Q_A', 'Q_R', 'Q_N', 'Q_D', 'Q_C', 'Q_Q', 'Q_E', 'Q_G', 'Q_H', 'Q_I', 'Q_L', 'Q_K', 'Q_M', 'Q_F', 'Q_P', 'Q_S', 'Q_T', 'Q_W', 'Q_Y', 'Q_V', 'E_A', 'E_R', 'E_N', 'E_D', 'E_C', 'E_Q', 'E_E', 'E_G', 'E_H', 'E_I', 'E_L', 'E_K', 'E_M', 'E_F', 'E_P', 'E_S', 'E_T', 'E_W', 'E_Y', 'E_V', 'G_A', 'G_R', 'G_N', 'G_D', 'G_C', 'G_Q', 'G_E', 'G_G', 'G_H', 'G_I', 'G_L', 'G_K', 'G_M', 'G_F', 'G_P', 'G_S', 'G_T', 'G_W', 'G_Y', 'G_V', 'H_A', 'H_R', 'H_N', 'H_D', 'H_C', 'H_Q', 'H_E', 'H_G', 'H_H', 'H_I', 'H_L', 'H_K', 'H_M', 'H_F', 'H_P', 'H_S', 'H_T', 'H_W', 'H_Y', 'H_V', 'I_A', 'I_R', 'I_N', 'I_D', 'I_C', 'I_Q', 'I_E', 'I_G', 'I_H', 'I_I', 'I_L', 'I_K', 'I_M', 'I_F', 'I_P', 'I_S', 'I_T', 'I_W', 'I_Y', 'I_V', 'L_A', 'L_R', 'L_N', 'L_D', 'L_C', 'L_Q', 'L_E', 'L_G', 'L_H', 'L_I', 'L_L', 'L_K', 'L_M', 'L_F', 'L_P', 'L_S', 'L_T', 'L_W', 'L_Y', 'L_V', 'K_A', 'K_R', 'K_N', 'K_D', 'K_C', 'K_Q', 'K_E', 'K_G', 'K_H', 'K_I', 'K_L', 'K_K', 'K_M', 'K_F', 'K_P', 'K_S', 'K_T', 'K_W', 'K_Y', 'K_V', 'M_A', 'M_R', 'M_N', 'M_D', 'M_C', 'M_Q', 'M_E', 'M_G', 'M_H', 'M_I', 'M_L', 'M_K', 'M_M', 'M_F', 'M_P', 'M_S', 'M_T', 'M_W', 'M_Y', 'M_V', 'F_A', 'F_R', 'F_N', 'F_D', 'F_C', 'F_Q', 'F_E', 'F_G', 'F_H', 'F_I', 'F_L', 'F_K', 'F_M', 'F_F', 'F_P', 'F_S', 'F_T', 'F_W', 'F_Y', 'F_V', 'P_A', 'P_R', 'P_N', 'P_D', 'P_C', 'P_Q', 'P_E', 'P_G', 'P_H', 'P_I', 'P_L', 'P_K', 'P_M', 'P_F', 'P_P', 'P_S', 'P_T', 'P_W', 'P_Y', 'P_V', 'S_A', 'S_R', 'S_N', 'S_D', 'S_C', 'S_Q', 'S_E', 'S_G', 'S_H', 'S_I', 'S_L', 'S_K', 'S_M', 'S_F', 'S_P', 'S_S', 'S_T', 'S_W', 'S_Y', 'S_V', 'T_A', 'T_R', 'T_N', 'T_D', 'T_C', 'T_Q', 'T_E', 'T_G', 'T_H', 'T_I', 'T_L', 'T_K', 'T_M', 'T_F', 'T_P', 'T_S', 'T_T', 'T_W', 'T_Y', 'T_V', 'W_A', 'W_R', 'W_N', 'W_D', 'W_C', 'W_Q', 'W_E', 'W_G', 'W_H', 'W_I', 'W_L', 'W_K', 'W_M', 'W_F', 'W_P', 'W_S', 'W_T', 'W_W', 'W_Y', 'W_V', 'Y_A', 'Y_R', 'Y_N', 'Y_D', 'Y_C', 'Y_Q', 'Y_E', 'Y_G', 'Y_H', 'Y_I', 'Y_L', 'Y_K', 'Y_M', 'Y_F', 'Y_P', 'Y_S', 'Y_T', 'Y_W', 'Y_Y', 'Y_V', 'V_A', 'V_R', 'V_N', 'V_D', 'V_C', 'V_Q', 'V_E', 'V_G', 'V_H', 'V_I', 'V_L', 'V_K', 'V_M', 'V_F', 'V_P', 'V_S', 'V_T', 'V_W', 'V_Y', 'V_V']
            replace_pssm_headers(pssm_features_path, new_pssm_headers)

            # Merge features 
            merge_aac_pssm_features(aac_output_path, pssm_features_path, all_features_output)
            
            #Step 5: Feature Selection
            selected_features_list = ['I_H', 'F_V', 'F_I', 'W_I', 'F_F', 'C_P', 'C_Y', 'D_I', 'Q_Q', 'M_N', 'I_D', 'R_Y', 'AAC_I', 'Q_G', 'Q_E', 'V_H', 'V_E', 'S_H', 'E_E', 'L_T', 'Q_Y', 'Q_I', 'S_K', 'V_C', 'G_Q', 'M_T', 'M_L', 'K_V', 'Q_A', 'Q_F', 'V_Q', 'H_E', 'N_G', 'D_T', 'R_S', 'N_C', 'L_P', 'D_L', 'T_R', 'P_K', 'W_W', 'AAC_S', 'R_W', 'I_V', 'Q_T', 'L_M', 'F_Q', 'N_E', 'F_W', 'I_E', 'W_R', 'Q_L', 'R_Q', 'N_D', 'N_N', 'S_V', 'I_P', 'K_T', 'A_G', 'C_D', 'Q_C', 'M_M', 'N_K', 'Q_P', 'Q_M', 'C_E', 'D_Q', 'D_N', 'V_N', 'E_R', 'S_R', 'E_D', 'K_N', 'E_Y', 'M_E', 'Y_H', 'Q_N', 'M_S', 'A_H', 'K_Y', 'W_L', 'C_T', 'D_A', 'D_K', 'R_M', 'A_I', 'AAC_Y', 'P_R', 'T_Q', 'V_F', 'H_L', 'R_F', 'V_D', 'S_I', 'L_L', 'H_G', 'H_F', 'Y_M', 'L_C', 'M_V', 'T_V', 'E_T', 'P_P', 'T_F', 'R_G', 'P_M', 'D_H', 'Y_T', 'P_I', 'L_F', 'K_H', 'L_I', 'P_E', 'M_P', 'M_D', 'AAC_N', 'D_E', 'G_G', 'Q_K', 'T_C', 'L_V', 'T_G', 'H_V', 'A_Q', 'R_N', 'E_N', 'L_Q', 'P_D', 'K_D', 'I_A', 'V_A', 'P_T', 'R_T', 'V_S', 'N_Q', 'A_D', 'M_W', 'F_M', 'S_Q', 'AAC_R', 'Y_C', 'T_D', 'H_M', 'L_W', 'H_Y', 'P_Y', 'W_M', 'C_G', 'N_S', 'AAC_D', 'H_S', 'N_T', 'M_Y', 'S_S', 'T_E', 'K_E', 'Y_L', 'E_S', 'V_G', 'E_H', 'R_L', 'K_P', 'W_D', 'AAC_V', 'F_S', 'I_C', 'A_P', 'E_A', 'L_A', 'F_A', 'Q_R', 'R_C', 'I_L', 'Q_H', 'E_I', 'V_P', 'W_G', 'Y_F', 'K_K', 'H_N', 'AAC_G', 'F_C', 'AAC_Q', 'A_L', 'N_A', 'D_G', 'G_P', 'S_E', 'H_H', 'T_S', 'G_F', 'F_G', 'G_D', 'P_H', 'T_P', 'Q_D', 'E_P', 'AAC_M', 'Y_W', 'S_N', 'V_V', 'R_I', 'P_A', 'T_H', 'G_I', 'G_H', 'N_R', 'Y_Q', 'H_W', 'H_Q', 'T_W', 'AAC_C', 'V_M', 'E_C', 'S_P']
            
            feature_selection(all_features_output, selected_features_list, selected_features_output)
            simulate_task("Loading the Model")
            progress_bar.update(1)
            
            # Step 6: Make Predictions by Model 1
            aac_pssm_predict(selected_features_output, args.input, prediction_output, args.threshold)
            simulate_task("Making Predictions")
            progress_bar.update(1)
        else:
            # Define paths
            working_dir = args.workingdir
            aac_output_path = os.path.join(args.workingdir, "AAC.csv")
            pssm_output_dir = os.path.join(args.workingdir, "PSSM")
            pssm_features_output_dir = os.path.join(args.workingdir, "PSSM_Features")
            all_features_output = os.path.join(working_dir, "all_features.csv")
            selected_features_output = os.path.join(working_dir, "selectedFeat.csv")
            prediction_output = os.path.join(working_dir, args.output)  # Final predictions output
            

            # Hardcoded POSSUM script path
            possum_script_path = "./possum/possum.py"
            header_handler_script_path = "./possum/headerHandler.py"

            # Ensure directories exist
            os.makedirs(pssm_features_output_dir, exist_ok=True)

            # Step 1: Generate AAC features
            aac_comp(args.input, aac_output_path)
            simulate_task("Generating Amino Acid Composition")
            progress_bar.update(1)

            # Step 2: Generate PSSM features
            gen_pssm(args.input, pssm_output_dir)
            simulate_task("Generate PSSM Files")
            progress_bar.update(1)

            # Step 3: Generate PSSM-based features
            pssm_features_path = generate_pssm_features(
                pssm_dir=os.path.join(pssm_output_dir, "pssm_raw"),
                fasta_file=args.input,
                possum_script=possum_script_path,
                header_handler_script=header_handler_script_path,
                output_dir=pssm_features_output_dir,
            )
            simulate_task("Generate PSSM-Based Features")
            progress_bar.update(1)
            
            # Step 4: Replace headers and concatenate AAC and PSSM features
            new_pssm_headers = [
                'A_A', 'A_R', 'A_N', 'A_D', 'A_C', 'A_Q', 'A_E', 'A_G', 'A_H', 'A_I', 'A_L', 'A_K', 'A_M', 'A_F', 'A_P', 'A_S', 'A_T', 'A_W', 'A_Y', 'A_V', 'R_A', 'R_R', 'R_N', 'R_D', 'R_C', 'R_Q', 'R_E', 'R_G', 'R_H', 'R_I', 'R_L', 'R_K', 'R_M', 'R_F', 'R_P', 'R_S', 'R_T', 'R_W', 'R_Y', 'R_V', 'N_A', 'N_R', 'N_N', 'N_D', 'N_C', 'N_Q', 'N_E', 'N_G', 'N_H', 'N_I', 'N_L', 'N_K', 'N_M', 'N_F', 'N_P', 'N_S', 'N_T', 'N_W', 'N_Y', 'N_V', 'D_A', 'D_R', 'D_N', 'D_D', 'D_C', 'D_Q', 'D_E', 'D_G', 'D_H', 'D_I', 'D_L', 'D_K', 'D_M', 'D_F', 'D_P', 'D_S', 'D_T', 'D_W', 'D_Y', 'D_V', 'C_A', 'C_R', 'C_N', 'C_D', 'C_C', 'C_Q', 'C_E', 'C_G', 'C_H', 'C_I', 'C_L', 'C_K', 'C_M', 'C_F', 'C_P', 'C_S', 'C_T', 'C_W', 'C_Y', 'C_V', 'Q_A', 'Q_R', 'Q_N', 'Q_D', 'Q_C', 'Q_Q', 'Q_E', 'Q_G', 'Q_H', 'Q_I', 'Q_L', 'Q_K', 'Q_M', 'Q_F', 'Q_P', 'Q_S', 'Q_T', 'Q_W', 'Q_Y', 'Q_V', 'E_A', 'E_R', 'E_N', 'E_D', 'E_C', 'E_Q', 'E_E', 'E_G', 'E_H', 'E_I', 'E_L', 'E_K', 'E_M', 'E_F', 'E_P', 'E_S', 'E_T', 'E_W', 'E_Y', 'E_V', 'G_A', 'G_R', 'G_N', 'G_D', 'G_C', 'G_Q', 'G_E', 'G_G', 'G_H', 'G_I', 'G_L', 'G_K', 'G_M', 'G_F', 'G_P', 'G_S', 'G_T', 'G_W', 'G_Y', 'G_V', 'H_A', 'H_R', 'H_N', 'H_D', 'H_C', 'H_Q', 'H_E', 'H_G', 'H_H', 'H_I', 'H_L', 'H_K', 'H_M', 'H_F', 'H_P', 'H_S', 'H_T', 'H_W', 'H_Y', 'H_V', 'I_A', 'I_R', 'I_N', 'I_D', 'I_C', 'I_Q', 'I_E', 'I_G', 'I_H', 'I_I', 'I_L', 'I_K', 'I_M', 'I_F', 'I_P', 'I_S', 'I_T', 'I_W', 'I_Y', 'I_V', 'L_A', 'L_R', 'L_N', 'L_D', 'L_C', 'L_Q', 'L_E', 'L_G', 'L_H', 'L_I', 'L_L', 'L_K', 'L_M', 'L_F', 'L_P', 'L_S', 'L_T', 'L_W', 'L_Y', 'L_V', 'K_A', 'K_R', 'K_N', 'K_D', 'K_C', 'K_Q', 'K_E', 'K_G', 'K_H', 'K_I', 'K_L', 'K_K', 'K_M', 'K_F', 'K_P', 'K_S', 'K_T', 'K_W', 'K_Y', 'K_V', 'M_A', 'M_R', 'M_N', 'M_D', 'M_C', 'M_Q', 'M_E', 'M_G', 'M_H', 'M_I', 'M_L', 'M_K', 'M_M', 'M_F', 'M_P', 'M_S', 'M_T', 'M_W', 'M_Y', 'M_V', 'F_A', 'F_R', 'F_N', 'F_D', 'F_C', 'F_Q', 'F_E', 'F_G', 'F_H', 'F_I', 'F_L', 'F_K', 'F_M', 'F_F', 'F_P', 'F_S', 'F_T', 'F_W', 'F_Y', 'F_V', 'P_A', 'P_R', 'P_N', 'P_D', 'P_C', 'P_Q', 'P_E', 'P_G', 'P_H', 'P_I', 'P_L', 'P_K', 'P_M', 'P_F', 'P_P', 'P_S', 'P_T', 'P_W', 'P_Y', 'P_V', 'S_A', 'S_R', 'S_N', 'S_D', 'S_C', 'S_Q', 'S_E', 'S_G', 'S_H', 'S_I', 'S_L', 'S_K', 'S_M', 'S_F', 'S_P', 'S_S', 'S_T', 'S_W', 'S_Y', 'S_V', 'T_A', 'T_R', 'T_N', 'T_D', 'T_C', 'T_Q', 'T_E', 'T_G', 'T_H', 'T_I', 'T_L', 'T_K', 'T_M', 'T_F', 'T_P', 'T_S', 'T_T', 'T_W', 'T_Y', 'T_V', 'W_A', 'W_R', 'W_N', 'W_D', 'W_C', 'W_Q', 'W_E', 'W_G', 'W_H', 'W_I', 'W_L', 'W_K', 'W_M', 'W_F', 'W_P', 'W_S', 'W_T', 'W_W', 'W_Y', 'W_V', 'Y_A', 'Y_R', 'Y_N', 'Y_D', 'Y_C', 'Y_Q', 'Y_E', 'Y_G', 'Y_H', 'Y_I', 'Y_L', 'Y_K', 'Y_M', 'Y_F', 'Y_P', 'Y_S', 'Y_T', 'Y_W', 'Y_Y', 'Y_V', 'V_A', 'V_R', 'V_N', 'V_D', 'V_C', 'V_Q', 'V_E', 'V_G', 'V_H', 'V_I', 'V_L', 'V_K', 'V_M', 'V_F', 'V_P', 'V_S', 'V_T', 'V_W', 'V_Y', 'V_V']
            replace_pssm_headers(pssm_features_path, new_pssm_headers)

            merge_aac_pssm_features(aac_output_path, pssm_features_path, all_features_output)
            
            #Step 5: Feature Selection
            selected_features_list = ['I_H', 'F_V', 'F_I', 'W_I', 'F_F', 'C_P', 'C_Y', 'D_I', 'Q_Q', 'M_N', 'I_D', 'R_Y', 'AAC_I', 'Q_G', 'Q_E', 'V_H', 'V_E', 'S_H', 'E_E', 'L_T', 'Q_Y', 'Q_I', 'S_K', 'V_C', 'G_Q', 'M_T', 'M_L', 'K_V', 'Q_A', 'Q_F', 'V_Q', 'H_E', 'N_G', 'D_T', 'R_S', 'N_C', 'L_P', 'D_L', 'T_R', 'P_K', 'W_W', 'AAC_S', 'R_W', 'I_V', 'Q_T', 'L_M', 'F_Q', 'N_E', 'F_W', 'I_E', 'W_R', 'Q_L', 'R_Q', 'N_D', 'N_N', 'S_V', 'I_P', 'K_T', 'A_G', 'C_D', 'Q_C', 'M_M', 'N_K', 'Q_P', 'Q_M', 'C_E', 'D_Q', 'D_N', 'V_N', 'E_R', 'S_R', 'E_D', 'K_N', 'E_Y', 'M_E', 'Y_H', 'Q_N', 'M_S', 'A_H', 'K_Y', 'W_L', 'C_T', 'D_A', 'D_K', 'R_M', 'A_I', 'AAC_Y', 'P_R', 'T_Q', 'V_F', 'H_L', 'R_F', 'V_D', 'S_I', 'L_L', 'H_G', 'H_F', 'Y_M', 'L_C', 'M_V', 'T_V', 'E_T', 'P_P', 'T_F', 'R_G', 'P_M', 'D_H', 'Y_T', 'P_I', 'L_F', 'K_H', 'L_I', 'P_E', 'M_P', 'M_D', 'AAC_N', 'D_E', 'G_G', 'Q_K', 'T_C', 'L_V', 'T_G', 'H_V', 'A_Q', 'R_N', 'E_N', 'L_Q', 'P_D', 'K_D', 'I_A', 'V_A', 'P_T', 'R_T', 'V_S', 'N_Q', 'A_D', 'M_W', 'F_M', 'S_Q', 'AAC_R', 'Y_C', 'T_D', 'H_M', 'L_W', 'H_Y', 'P_Y', 'W_M', 'C_G', 'N_S', 'AAC_D', 'H_S', 'N_T', 'M_Y', 'S_S', 'T_E', 'K_E', 'Y_L', 'E_S', 'V_G', 'E_H', 'R_L', 'K_P', 'W_D', 'AAC_V', 'F_S', 'I_C', 'A_P', 'E_A', 'L_A', 'F_A', 'Q_R', 'R_C', 'I_L', 'Q_H', 'E_I', 'V_P', 'W_G', 'Y_F', 'K_K', 'H_N', 'AAC_G', 'F_C', 'AAC_Q', 'A_L', 'N_A', 'D_G', 'G_P', 'S_E', 'H_H', 'T_S', 'G_F', 'F_G', 'G_D', 'P_H', 'T_P', 'Q_D', 'E_P', 'AAC_M', 'Y_W', 'S_N', 'V_V', 'R_I', 'P_A', 'T_H', 'G_I', 'G_H', 'N_R', 'Y_Q', 'H_W', 'H_Q', 'T_W', 'AAC_C', 'V_M', 'E_C', 'S_P']
        
            feature_selection(all_features_output, selected_features_list, selected_features_output)
            simulate_task("Loading the Model")
            progress_bar.update(1)
            
            # Step 6: Make Predictions and integrate blast
            aac_pssm_predict(selected_features_output, args.input, prediction_output, args.threshold)
            blast_db = "./blast/pred_blast/train.fasta"
            run_blast_and_integrate(args.input, blast_db, prediction_output, args.workingdir)
            simulate_task("Making Predictions")
            progress_bar.update(1)

    files_to_remove = ["AAC.csv","all_features.csv", "selectedFeat.csv", "results_evalue_1e-10.txt"]
    folders_to_remove = ["PSSM", "PSSM_Features",]
    cleanup(args.workingdir, files_to_remove, folders_to_remove)