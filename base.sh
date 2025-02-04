#!/bin/bash

#SBATCH --job-name=whisper_parallel                         # Set a job name
#SBATCH --output=data/whisper_parallel_%j.out               # Standard output (%j = job ID)
#SBATCH --error=data/whisper_parallel_%j.err                # Standard error
#SBATCH --time=02:00:00                                     # Total job time
#SBATCH --nodes=1                                           # Request 1 node
#SBATCH --ntasks=1                                          # Total tasks (4 tasks in this case)
#SBATCH --ntasks-per-node=1                                 # 2 tasks per node
#SBATCH --mem=10g                                           # Memory per node

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <file_name> <file_extension> [num_speakers] [whisper_model_size] [hf_token]"
  exit 1
fi

# Assign arguments to variables
file_name=$1                                   # The base file name (without extension)
file_extension=$2                              # The file extension (e.g., wav, mp3)
num_speakers=${3:-2}                           # Default number of speakers is 2 if not provided
whisper_model_size=${4:-"base"}                # Default Whisper model size is "base" if not provided
hf_token=${5:-"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"} # Default Hugging Face token if not provided

# Create the local directory structure
mkdir -p "data"

# Copy the file from the Object Store if it doesn't exist locally
if [ ! -f "data/${file_name}.${file_extension}" ]; then
  echo "Copying file from Object Store... transcript:${file_name}.${file_extension}"
  cpobj "transcript:${file_name}.${file_extension}" "data/${file_name}.${file_extension}"

  # Check if the file was copied successfully
  if [ -f "data/${file_name}.${file_extension}" ]; then
    echo "File copied successfully to data/${file_name}.${file_extension}"
  else
    echo "Failed to copy file. Please check the Object Store path and credentials."
    exit 1
  fi
else
  echo "File data/${file_name}.${file_extension} already exists. Skipping copy."
fi

# Run the Python script with srun
srun --nodes=1 --ntasks=1 --mem=10g bash -c "
  echo 'Running start.py on node 1...';
  python3 data/start.py 'data/${file_name}.${file_extension}' --num_speakers $num_speakers --whisper_model_size $whisper_model_size --hf_token $hf_token > 'data/${file_name}-transcribed.txt'
" &

# Wait for parallel tasks to finish
wait

# Copy the output file back to the transcript folder
cp "data/${file_name}-transcribed.txt" "transcript/${file_name}-transcribed.txt"
echo "Output copied to transcript/${file_name}-transcribed.txt"


echo "All tasks completed."