$(function () {

  var table;

  var MAX_LINE_LENGTH = 80; // make sure the description field in the tooltip
  // does not exceed this length

  // Read in the json file that the dashboard.py script generated
  // fetch('/home/aviary_vars.json')
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
        currentLine += word + ' ';
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
    // button.textContent = "Copy";
    button.style.marginRight = "5px"; // Add some spacing between the cell content and the button
    button.style.padding = "1px"; // Add some spacing between the cell content and the button
    button.classList.add("btn", "btn-light", "btn-sm", "fa-xs"); // Add any additional classes for styling

    // Create an icon element using FontAwesome
    var icon = document.createElement("i");
    icon.className = "fas fa-copy"; // Use the copy icon from FontAwesome
    button.appendChild(icon); // Add the icon to the button

    // Append the button to the cellContent container
    cellContent.appendChild(button);


    var text = document.createElement("value");
    text.textContent = cellValue;
    cellContent.appendChild(text);


    onRendered(function () {
      $(button).on('click', (e) => {
        e.stopPropagation();
        navigator.clipboard.writeText(cellValue);
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
      // fitData - not a big difference maybe because already gave fixed width to the columns
      // fitDataFill - not a big difference maybe because already gave fixed width to the columns
      // fitDataTable - not a big difference maybe because already gave fixed width to the columns
      // fitColumns - not a big difference maybe because already gave fixed width to the columns
      // fitDataStretch - uses the full width for the value, which is good
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
            return formatValueTooltip(cell);
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