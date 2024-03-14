$(function () {

  var table;

  var MAX_LINE_LENGTH = 80; // make sure the description field in the tooltip
  // does not exceed this length

  // Read in the json file that the dashboard.py script generated
  fetch('./aviary_vars.json')
    .then((response) => response.json())
    .then((json) => createTabulator(json));

  // Add event listener to the checkbox for showing only aviary variables
  document.getElementById('aviary-vars-filter').addEventListener('change', function () {
    if (this.checked) {
      applyRegexFilter();
    } else {
      removeFilter();
    }
  });

  // Function to apply the regex filter
  function applyRegexFilter() {
    var regex = new RegExp("^(aircraft:|mission:)"); // Replace with your regex
    table.setFilter((data) => {
      return regex.test(data.prom_name); // Assuming you are filtering the 'name' column
    });
  }

  // Function to remove the filter
  function removeFilter() {
    table.clearFilter();
  }

  function isDictionary(obj) {
    return typeof obj === 'object' && obj !== null && !Array.isArray(obj) && !(obj instanceof Function);
  }

  // function to count the initial spaces in a string
  // format floating point numbers using decimal or scientific notation
  //   as needed
  function formatNumberIntelligently(value, precision = 3) {
    // Determine the absolute value to decide on the formatting
    const absValue = Math.abs(value);

    // Define thresholds for using scientific notation
    const upperThreshold = 1e5;
    const lowerThreshold = 1e-3;

    // Use scientific notation for very small or very large numbers
    if (absValue >= upperThreshold || (absValue > 0 && absValue < lowerThreshold)) {
      return value.toExponential(precision);
    }

    // Use fixed-point notation for numbers within a 'normal' range, ensuring the precision is respected
    // Adjust the maximumFractionDigits based on the integer part length to maintain overall precision
    const integerPartLength = Math.floor(value) === 0 ? 1 : Math.floor(Math.log10(Math.abs(Math.floor(value)))) + 1;
    const adjustedPrecision = Math.max(0, precision - integerPartLength);

    return value.toLocaleString('en-US', {
      minimumFractionDigits: adjustedPrecision,
      maximumFractionDigits: precision,
      useGrouping: false,
    });
  }

  /**
   * Converts the input element to a string representation intelligently. 
   * 
   * @param {any} element - The element to be converted to a string. Can be of any type.
   * @returns {string} The string representation of the input element.
   */
  function intelligentStringify(element) {
    if (typeof element === 'string') {
      // Return the string directly without quotes
      return element;
    } else {
      // Use JSON.stringify for other types
      return JSON.stringify(element);
    }
  }

  /**
 * Convert an element to a string that is human readable.
 * @param {Object} element The scalar item to convert.
 * @returns {String} The string representation of the element.
 */
  function elementToString(element) {
    if (typeof element === 'number') {
      if (Number.isInteger(element)) { return element.toString(); }
      return formatNumberIntelligently(element); /* float */
    }

    return intelligentStringify(element);
  }

  /**
       * Convert a value to a string that can be used in Python code.
       * @param {Object} val The value to convert.
       * @returns {String} The string of the converted object.
       */
  function valToCopyString(val, isTopLevel = true) {
    if (!Array.isArray(val)) { return elementToString(val); }
    let valStr;

    if (isTopLevel) {
      valStr = 'array([';
    } else {
      valStr = '[';
    }
    for (const element of val) {
      valStr += valToCopyString(element, false) + ', ';
    }

    if (val.length > 0) {
      if (isTopLevel) {
        return valStr.replace(/^(.+)(, )$/, '$1])');
      } else {
        return valStr.replace(/^(.+)(, )$/, '$1]');
      }
    }
  }

  function countInitialSpaces(str) {
    // This regex matches the initial spaces in the string
    const match = str.match(/^\s*/);
    // If there's a match, return its length; otherwise, return 0
    return match ? match[0].length : 0;
  }

  // split a string so that it isn't too long for the tooltip
  function splitStringIntoLines(str, maxLineLength = 80) {
    // need to preserve the initial spaces of the string!
    lengthOfInitialSpaces = countInitialSpaces(str);
    initialWordOfSpaces = " ".repeat(lengthOfInitialSpaces);
    const words = str.trim().split(' ');
    words.unshift(initialWordOfSpaces);
    const lines = [];
    let currentLine = '';

    words.forEach(word => {
      // Truncate the word if it's longer than maxLineLength
      if (word.length > maxLineLength) {
        word = word.substring(0, maxLineLength);
      }

      if ((currentLine + word).length > maxLineLength) {
        lines.push(currentLine);
        currentLine = word;
      } else {
        currentLine += ' ' + word;
      }
    });

    if (currentLine) {
      lines.push(currentLine);
    }

    return lines;
  }

  // format each individual string for metadata
  function formatIndividualMetadataString(preText, item, maxLineLength = 80) {
    indent = preText.length;
    resultString = "";
    s = JSON.stringify(item);
    lines = splitStringIntoLines(preText + s, maxLineLength = MAX_LINE_LENGTH);
    lines.forEach((line, index) => {
      if (index == 0) {
        resultString += line + "\n";
      }
      else {
        resultString += " ".repeat(indent) + line + "\n";
      }
    })
    return resultString;
  }

  // format the entire text for the metadata tooltip
  function formatMetadataTooltip(cell) {
    prom_name = cell.getValue();
    metadata = cell.getData().metadata;

    // Initialize a string to hold the resulting string
    let resultString = "prom_name: " + prom_name + "\n";

    dictObject = JSON.parse(metadata);

    // Iterate over each key-value pair
    for (let key in dictObject) {
      if (dictObject.hasOwnProperty(key)) {
        // Append key and value to the result string
        if (isDictionary(dictObject[key])) {
          resultString += key + ": " + "\n";
          for (let hnkey in dictObject[key]) {
            if (Array.isArray(dictObject[key][hnkey])) {
              resultString += "    " + hnkey + ": \n";
              for (let item of dictObject[key][hnkey]) {
                resultString += formatIndividualMetadataString("        ", item);
              }
            }
            else {
              resultString += formatIndividualMetadataString("    " + hnkey + ": ", dictObject[key][hnkey]);
            }
          }
        }
        else {
          resultString += formatIndividualMetadataString(key + ": ", dictObject[key]);
        }
      }
    }
    return resultString;
  }

  // The Tabulator formatter function used to include a button to copy the 
  // value in the cell
  function copyButtonFormatter(cell, formatterParams, onRendered) {
    var cellContent = document.createElement("span");
    var cellValue = cell.getValue();
    if (cellValue.length === 0) {
      return "";
    }

    var button = document.createElement("button");
    button.textContent = "Copy";
    button.classList.add("copy-button");

    // Append the button to the cellContent container
    cellContent.appendChild(button);

    var text = document.createElement("value");

    text.textContent = intelligentStringify(cellValue);
    cellContent.appendChild(text);

    onRendered(function () {
      $(button).on('click', (e) => {
        e.stopPropagation();
        copiedCellValue = valToCopyString(cellValue);
        navigator.clipboard.writeText(copiedCellValue);
      });
    }
    );

    return cellContent;
  }

  // Given the variable data as a JSON object, create the Tabulator displays on the Web page
  function createTabulator(tableData) {
    table = new Tabulator("#example-table", {
      height: "100%",
      data: tableData,
      dataTree: true,
      dataTreeFilter: false,
      headerSort: true,
      movableColumns: false,
      dataTreeStartExpanded: false,
      layout: "fitDataStretch",
      columns: [
        {
          title: "Absolute Name",
          field: "abs_name",
          headerFilter: "input",
          width: 550,
          tooltip: function (e, cell) {
            return formatMetadataTooltip(cell);
          },
          formatter: copyButtonFormatter
        },
        {
          title: "Promoted Name",
          field: "prom_name",
          headerFilter: "input",
          width: 300,
          tooltip: function (e, cell) {
            return formatMetadataTooltip(cell);
          },
          formatter: copyButtonFormatter
        },
        {
          title: "Value",
          field: "value",
          width: 300,
          tooltip: function (e, cell) {
            return cell.getValue();
          },
          formatter: copyButtonFormatter
        },
        {
          title: "Units",
          field: "units",
          width: 120,
        },
      ]
    });
  }
});