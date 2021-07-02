# ADaPT-ML

## To add a new modelling task: ##
1. For Label Studio, create the config.xml file and add a line in docker-compose.yml to initialize the project to the container run command
2. For data programming, add an endpoint to the MLproject, a class for the labels and a file for your labelling functions

A statement of need
The authors should clearly state what problems the software is designed to solve and who the target audience is.

Installation instructions
There should be a clearly-stated list of dependencies. Ideally these should be handled with an automated package management solution.

Good: A package management file such as a Gemfile or package.json or equivalent
OK: A list of dependencies to install
Bad (not acceptable): Reliance on other software not listed by the authors
Example usage
The authors should include examples of how to use the software (ideally to solve real-world analysis problems).

API documentation
Reviewers should check that the software API is documented to a suitable level.

Good: All functions/methods are documented including example inputs and outputs
OK: Core API functionality is documented
Bad (not acceptable): API is undocumented

Community guidelines
There should be clear guidelines for third-parties wishing to:

Contribute to the software
Report issues or problems with the software
Seek support
Functionality
Reviewers are expected to install the software they are reviewing and to verify the core functionality of the software.

Tests
Authors are strongly encouraged to include an automated test suite covering the core functionality of their software.

Good: An automated test suite hooked up to an external service such as Travis-CI or similar
OK: Documented manual steps that can be followed to objectively check the expected functionality of the software (e.g., a sample input file to assert behavior)
Bad (not acceptable): No way for you, the reviewer, to objectively assess whether the software works

#Paper:

Summary: Has a clear description of the high-level functionality and purpose of the software for a diverse, non-specialist audience been provided?
A statement of need: Does the paper have a section titled ‘Statement of Need’ that clearly states what problems the software is designed to solve and who the target audience is?
State of the field: Do the authors describe how this software compares to other commonly-used packages?
Quality of writing: Is the paper well written (i.e., it does not require editing for structure, language, or writing quality)?
References: Is the list of references complete, and is everything cited appropriately that should be cited (e.g., papers, datasets, software)? Do references in the text use the proper citation syntax?