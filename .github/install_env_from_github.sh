#!/bin/bash

# Check if user provided an argument
if [ "$#" -lt 1 ]; then
    echo "Exiting: Please provide an input environment yaml file like this: $0 <path_to_input_yaml>"
    exit 1
fi

# Initialize flags for special package installations
install_openmdao_from_git=false
install_dymos_from_git=false

# Name of the original YAML file provided by the user
input_yaml=$1

# Name of the intermediate and modified YAML files
intermediate_yaml="intermediate_environment.yml"
output_yaml="modified_environment.yml"

# Name of the new conda environment
if [ "$#" -ge 2 ]; then
    env_name=$2
else
    read -r firstline<$input_yaml
    env_name=${firstline#*name: }
fi

# Determine if mamba is available and use it; otherwise, use conda
if command -v mamba &> /dev/null; then
    pkg_manager="mamba"
else
    pkg_manager="conda"
fi

# Check if an environment with the specified name already exists
if conda env list | grep -q -w "$env_name"; then
    read -p "An environment with the name $env_name already exists. Do you want to remove it before proceeding? (y/n): " answer
    if [ "$answer" = "y" ]; then
        conda env remove -n $env_name
    else
        echo "Exiting. Please choose a different environment name, environment names can be specified as an additional argument."
        exit 1
    fi
else
    echo "Creating the $env_name environment"
fi

# Extract pyoptsparse line
# pyoptsparse_line=$(grep ' pyoptsparse' $input_yaml | sed 's/^    //')

# Remove specified packages and write to an intermediate file
grep -v -e 'aviary' -e 'build-pyoptsparse' -e 'pyoptsparse' -e 'networkx' -e '  - pip:' $input_yaml > $intermediate_yaml

# Check for 'dev' versions of OpenMDAO and Dymos
if grep -q -e 'openmdao.*dev' $intermediate_yaml; then
    install_openmdao_from_git=true
    grep -v -e 'openmdao' $intermediate_yaml > $output_yaml
    mv $output_yaml $intermediate_yaml  # Move output to intermediate for further processing
fi

if grep -q -e 'dymos.*dev' $intermediate_yaml; then
    install_dymos_from_git=true
    grep -v -e 'dymos' $intermediate_yaml > $output_yaml
    mv $output_yaml $intermediate_yaml  # Move output to intermediate for further processing
fi

# Move remaining pip installs to a separate requirements.txt file
# Prevents conda evironment from failing to build if there is an issue with the pip installs
pip_prefix='      - '
requirements_txt=$env_name"_requirements.txt"
rm -f $requirements_txt
while IFS= read -r line
do
    case $line in *$pip_prefix* )
        package=${line#*$pip_prefix}
        echo $package >> $requirements_txt
        ;;
        *)
        echo "$line" >> $output_yaml
    esac
done < "$intermediate_yaml"

# Insert pyoptsparse line after the 'dependencies' line
# awk -v p="$pyoptsparse_line" '/dependencies/ {print; print p; next}1' $output_yaml > tmp && mv tmp $output_yaml

# Create a new conda environment from the modified YAML file
$pkg_manager env create -f $output_yaml -n $env_name

# Attempting to resolve "CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'."
# tmp=${SHELL##*/}
# MYSHELL=${tmp##*\\}
# $pkg_manager init $MYSHELL
# source activate base

# Activate the new conda environment
eval "$(conda shell.bash hook)"
conda activate $env_name

# Check flags and install special packages if needed
pip install -r $requirements_txt

if [ "$install_openmdao_from_git" = true ]; then
    pip install git+https://github.com/OpenMDAO/OpenMDAO.git
fi

if [ "$install_dymos_from_git" = true ]; then
    pip install git+https://github.com/OpenMDAO/dymos.git
fi

# Clean up
# Remove the intermediate file
rm -f $intermediate_yaml
# Remove the final file
rm -f $output_yaml
# Remove the pip requirements
rm -f $requirements_txt

# Print a reminder to install other packages manually
echo "---------------"
echo "Reminder: You need to install Aviary manually using the relevant versions based on what you are trying to debug."
echo "Please also install pyOptSparse with SNOPT manually if you need them to debug your problem."
echo "Activate the new environment using: conda activate $env_name"
