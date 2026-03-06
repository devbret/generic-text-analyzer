# Generic Text Analyzer

![Word cloud generated from a recent run of this program.](https://hosting.photobucket.com/bbcfb0d4-be20-44a0-94dc-65bff8947cf2/e210be7f-54e9-4695-bd56-45b81086f3d6.png)

Batch-processes `.txt` files to perform deep NLP-based text analysis, generate summaries and use Claude (via the Anthropic API) to create a plain-language summary of each analysis.

## Overview

The Generic Text Analyzer performs a comprehensive analysis of .txt files found in the `input` directory by extracting linguistic insights and visualizations. It reads each text file, preprocesses the content and generates frequency counts for words, n-grams, parts of speech, named entities and TF-IDF scores.

This program calculates readability metrics and sentiment, performs topic modeling using LDA and visualizes results through word clouds and bar charts. A detailed report is saved as a text file, and the content is also summarized using Anthropic’s Claude API. All outputs, including visualizations and summaries, are stored in the `output` directory.
