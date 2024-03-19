# Coding Standards

Aviary uses a combination of formal standards and general best practices. To contribute code to the Aviary codebase, we request you follow these guidelines.

In general, always follow the excellent advice given in the [PEP 8 Python style guide](https://peps.python.org/pep-0008/). Consistency is also key - pick a convention and stick with it for an entire file.

## Style and Formatting
The Aviary development team uses the [autopep8 formatter](https://pypi.org/project/autopep8/) to handle formatting in a consistent way across the codebase. Autopep8 is a tool that formats Python code (through alteration of whitespace and line breaks) to follow a consistent style and attempt to keep lines within the character limit whenever possible. Aviary uses autopep8 as part of its [pre-commit](https://pre-commit.com/) scripts. The only required setting (which Aviary automatically enforces) is `max_line_length = 89`. Use of the `aggressive` or `experimental` flag is optional and up to the user, but take care that these settings do not alter code function or significantly hamper readability. The utility [isort](https://pycqa.github.io/isort/) is also recommended for formatting of import statements (something not specifically handled by autopep8), but it is currently not required.

### Pre-Commit Setup
To set up pre-commit in your development python environment, there are a few one-time steps that must be done. The following commands need to be run to install pre-commit.

`pip install pre-commit`

`pre-commit install`

The Aviary repository contains a configuration file that defines what is run when commits are made and with what options enabled. Currently this is limited to autopep8 with a max line length restriction.

### Controlling Display Levels
To make debugging issues easier, it is strongly recommended to make use of the `VERBOSITY` setting. This allows control over how much information is displayed to a user; too much information makes finding relevant information difficult and not enough information can make tracking difficult. `Brief` should be the default in most cases; however, `Quiet` should be the default for tests.

## Naming Conventions
### Variables
When it comes to variable naming, always be verbose! The Aviary team considers long but clear and descriptive names superior to shortened or vague names. Typing out a long name is only difficult once, as most IDEs will help you auto-complete long variable names, but the readability they add lasts a lifetime!
The Aviary variable hierarchy is an excellent example of good variable naming. When adding variables to the hierarchy, adhering to the following naming conventions is requested. Inside the codebase itself, such as inside openMDAO components, it is not required but still highly recommended to follow these guidelines.

**A good variable name should:**
1. Not be ambiguous (avoid names that cannot be understood without context, like *x* or *calc*)
2. Avoid abbreviation (*thrust_to_weight_ratio* preferred to *T_W_ratio*). Note that Aviary will sometimes still shorten extremely long words such as "miscellaneous" to "misc" - use your best judgement!
3. Use physical descriptions rather than jargon or mathematical symbols (*density* preferred to *rho* - even better, include what flight condition this density is at, such as *current*, *sea_level*, etc.)
4. Place adjectives or modifiers after the "main" variable name rather than before (such as *thrust_max*, *thrust_sea_level_static*). This makes it is easier to autocomplete using an IDE - simply typing "thrust" will provide you with a handy list of all of the different kinds of thrust you can use.
5. Be formatted in "[snake case](https://en.wikipedia.org/wiki/Snake_case)", or all lowercase with underscore-delineated words (such as *example_variable*)

### Classes
Class names should be written in "[CamelCase](https://en.wikipedia.org/wiki/Camel_case_)", or naming with no delimiters such as dashes or underscores between words and each word beginning with a capital letter.

### Functions and Methods
Function and method names, similar to variables, should be formatted in "snake case". Class methods that are not intended to be accessed outside of the class definition can append an underscore at the beginning of the method name to mark it as "private", to help other users avoid using those methods incorrectly. An example of this is:
*def _private_method(self):*

## Code Re-Use and Utility Functions
If an identical block of code appears multiple times inside a file, consider moving it to a function to make your code cleaner. Repeated code bloats files and makes them less readable. If that function ends up being useful outside that individual file, move it to a "utils.py" file in the lowest-level directory shared by all files that need that function. If the utility function is useful across all of Aviary and is integral to the tool's operation, the aviary/utils folder is the appropriate place for it.