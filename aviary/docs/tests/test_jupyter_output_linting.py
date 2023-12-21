import unittest
import os.path
import json
import sys

exclude = {
    'tests',
    'test',
    '_build',
    '.ipynb_checkpoints',
    '_srcdocs',
    '__pycache__'
}

directories = None

top = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_dirs():
    global directories
    if directories is None:
        directories = []
        for root, dirs, files in os.walk(top, topdown=True):
            # do not bother looking further down in excluded dirs
            dirs[:] = [d for d in dirs if d not in exclude]
            for dir in dirs:
                directories.append(os.path.join(root, dir))
    return directories


FILES = None


def _get_files():
    global FILES
    if FILES is None:
        FILES = []
        for dir_name in _get_dirs():
            dirpath = os.path.join(top, dir_name)

            # Loop over files
            for file_name in os.listdir(dirpath):
                if not file_name.startswith('_') and file_name.endswith('.ipynb'):
                    FILES.append(dirpath + "/" + file_name)

        if len(FILES) < 1:
            raise RuntimeError(f"No notebooks found. Top directory is {top}.")
    return FILES


@unittest.skipIf(sys.platform == 'win32', "Tests don't work in Windows")
class LintJupyterOutputsTestCase(unittest.TestCase):
    """
    Check Jupyter Notebooks for outputs through execution count and recommend to remove output.
    """

    def test_output(self):
        """
        Check that output has been cleaned out of all cells.
        """
        for file in _get_files():
            with self.subTest(file):
                with open(file) as f:
                    json_data = json.load(f)
                    for cell in json_data['cells']:
                        if 'execution_count' in cell and cell['execution_count'] is not None:
                            msg = "Clear output with 'reset_notebook path_to_notebook.ipynb'"
                            self.fail(f"Output found in {file}.\n{msg}")

    def test_assert(self):
        """
        Make sure any code cells with asserts are hidden.
        """
        for file in _get_files():
            with open(file) as f:
                json_data = json.load(f)
                for block in json_data['cells'][1:]:

                    # Don't check markup cells
                    if block['cell_type'] != 'code':
                        continue

                    tags = block['metadata'].get('tags')
                    if tags:

                        # Don't check hidden cells
                        if ('remove-input' in tags and 'remove-output' in tags) or 'remove-cell' in tags:
                            continue

                        # We allow an assert in a cell if you tag it.
                        if "allow-assert" in tags:
                            continue

                    for line in block['source']:
                        if 'assert' in line:
                            sblock = ''.join(block['source'])
                            stags = tags if tags else ''
                            delim = '-' * 50
                            self.fail(f"Assert found in a code block in {file}:\n"
                                      f"Tags: {stags}\n"
                                      f"Block source:\n{delim}\n{sblock}\n{delim}")

    def test_eval_rst(self):
        """
        Make sure any automethod calls are bracketed with {eval-rst}.
        """
        files = set()

        for file in _get_files():
            with open(file) as f:
                json_data = json.load(f)
                blocks = json_data['cells']
                for block in blocks[1:]:

                    # check only markdown cells
                    if block['cell_type'] != 'markdown':
                        continue

                    code = ''.join(block['source'])
                    if 'automethod::' in code and '{eval-rst}' not in code:
                        files.add(file)

        if files:
            self.fail("'automethod' directive found in the following {} files without"
                      "'eval-rst':\n{}".format(len(files), '\n'.join(files)))


if __name__ == '__main__':
    unittest.main()
