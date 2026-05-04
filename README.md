# Generic Text Analyzer

![Word cloud generated from a recent run of this program.](https://hosting.photobucket.com/bbcfb0d4-be20-44a0-94dc-65bff8947cf2/e210be7f-54e9-4695-bd56-45b81086f3d6.png)

Batch-processes `.txt` files to perform deep NLP-based text analysis, generate summaries and use Claude (via the Anthropic API) to create a plain-language summary of each analysis.

## Overview

The Generic Text Analyzer performs a comprehensive analysis of .txt files found in the `input` directory by extracting linguistic insights and visualizations. It reads each text file, preprocesses the content and generates frequency counts for words, n-grams, parts of speech, named entities and TF-IDF scores.

This program calculates readability metrics and sentiment, performs topic modeling using LDA and visualizes results through word clouds and bar charts. A detailed report is saved as a text file, and the content is also summarized using Anthropic’s Claude API. All outputs, including visualizations and summaries, are stored in the `output` directory.

## Set Up

Below are instructions for installing and running this application on a Linux machine.

### Programs Needed

- [Git](https://git-scm.com/downloads)

- [Python](https://www.python.org/downloads/)

### Steps

1. Install the above programs

2. Open a terminal

3. Clone this repository: `git clone git@github.com:devbret/generic-text-analyzer.git`

4. Navigate to the repo's directory: `cd generic-text-analyzer`

5. Create a virtual environment: `python3 -m venv venv`

6. Activate your virtual environment: `source venv/bin/activate`

7. Install the needed dependencies for running the script: `pip install -r requirements.txt`

8. Download the required spaCy English language model: `python3 -m spacy download en_core_web_sm`

9. Convert the `.env.template` file into a `.env` file

10. Add value for the `ANTHROPIC_API_KEY` environmental variable to the `.env` file

11. Create the `input` and `output` directories if they do not already exist: `mkdir -p input output`

12. Place your `.txt` files into the `input` directory for analysis

13. Process the `.txt` input files: `python3 app.py`

14. Results will be placed in the `output` directory of this project

15. Exit the virtual environment: `deactivate`

## Other Considerations

This project repo is intended to demonstrate an ability to do the following:

- Analyze every `.txt` file in an input folder and generate a detailed text analysis report for each one

- Extract linguistic insights such as word frequencies, n-grams, named entities and sentiment

- Create charts, word clouds and sentiment arc visualizations as supporting outputs

- Use Claude to generate a summary of each completed text analysis report

If you have any questions or would like to collaborate, please reach out either on GitHub or via [my website](https://bretbernhoft.com/).
