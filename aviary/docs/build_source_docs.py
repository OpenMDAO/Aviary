import os

IGNORE_LIST = []

packages = [
    'interface',
    'utils',
    'variable_info',
]

index_top = """
# Source Docs

"""


def header(filename, path):

    header = """---
orphan: true
---

# %s

```{eval-rst}
    .. automodule:: %s
        :members:
        :undoc-members:
        :special-members: __init__, __contains__, __iter__, __setitem__, __getitem__
        :show-inheritance:
        :noindex:
```
""" % (filename, path)
    return header


def build_src_docs(top, src_dir, project_name='aviary'):
    doc_dir = os.path.join(top, "_srcdocs")

    if not os.path.isdir(doc_dir):
        os.mkdir(doc_dir)

    packages_dir = os.path.join(doc_dir, "packages")
    if not os.path.isdir(packages_dir):
        os.mkdir(packages_dir)

    index_filename = os.path.join(doc_dir, "index.md")
    index = open(index_filename, "w")
    index_data = index_top

    for package in packages:
        sub_packages = []
        package_filename = os.path.join(packages_dir, package + ".md")
        package_name = project_name + "." + package

        package_dir = os.path.join(src_dir, package.replace('.', '/'))
        for sub_listing in sorted(os.listdir(package_dir)):
            if (os.path.isdir(sub_listing) and sub_listing != "tests") or \
                    (sub_listing.endswith(".py") and not sub_listing.startswith('_')):
                sub_packages.append(sub_listing.rsplit('.')[0])

        if len(sub_packages) > 0:
            title = f"[{package}]"
            link = f"(packages/{package}.md)\n"
            index_data += f"- {title}{link}"

            package_dir = os.path.join(packages_dir, package)
            os.mkdir(package_dir)

            package_file = open(package_filename, "w")
            package_data = f"---\norphan: true\n---\n\n# {package_name}\n\n"

            for sub_package in sub_packages:
                SKIP_SUBPACKAGES = ['__pycache__']
                if sub_package not in SKIP_SUBPACKAGES:
                    title = f"[{sub_package}]"
                    link = f"({package}/{sub_package}.md)\n"
                    package_data += f"- {title}{link}"

                    ref_sheet_filename = os.path.join(
                        package_dir, sub_package + ".md")
                    ref_sheet = open(ref_sheet_filename, "w")

                    filename = sub_package + ".py"
                    ref_sheet.write(header(filename, package_name + "." + sub_package))
                    ref_sheet.close()

            package_file.write(package_data)
            package_file.close()

    index.write(index_data)
    index.close()


if __name__ == '__main__':
    build_src_docs(".", "..")
