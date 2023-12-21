$(function () {

// Read in the json file that the dashboard.py script generated
// fetch('/home/aviary_vars.json')
fetch('./aviary_vars.json')
    .then((response) => response.json())
    .then((json) => createTabulator(json));

function displayValueInToolTip(e, cell, onRendered) {
            //e - mouseover event
            //cell - cell component
            //onRendered - onRendered callback registration function

            var el = document.createElement("div");
            el.style.backgroundColor = "lightgreen";
//            el.innerText = cell.getColumn().getField() + " - " + cell.getValue(); //return cells "field - value";
            el.innerText = cell.getValue(); //return cells "field - value";

            return el;
};

// Given the variable data as a JSON object, create the Tabulator displays on the Web page
function createTabulator(tableData)
{
  var table = new Tabulator("#example-table", {
    height:"100%",
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
    layout:"fitDataStretch",
    columns: [
      {
        title: "Absolute Name",
        field: "abs_name",
        headerFilter: "input",
        width: 550,
      },
      {
        title: "Promoted Name",
        field: "prom_name",
        headerFilter: "input",
        width: 300,
        tooltip: true,
      },
      {
        title: "Value",
        field: "value",
        width: 300,
        tooltip: (e, cell, onRendered) => displayValueInToolTip(e, cell, onRendered),
      },
    ]
  });

}


});