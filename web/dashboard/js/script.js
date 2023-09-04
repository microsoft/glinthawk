// select the parent element of div#gpu_utilization
var gpu_utilization = document.querySelector("#gpu-utilization").closest("div.col-6");

// set the dimensions and margins of the graph
var margin = { top: 10, right: 30, bottom: 30, left: 60 },
  width = gpu_utilization.offsetWidth - margin.left - margin.right,
  height = 400 - margin.top - margin.bottom;


var plot_chart = (elem) => {

  // set the ranges
  var x = d3.scaleTime().range([0, width]);
  var y = d3.scaleLinear().range([height, 0]);

  // append the svg object to the body of the page
  var svg = d3.select(elem)
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
      "translate(" + margin.left + "," + margin.top + ")");

  //Read the data
  d3.csv("data.csv").then((data) => {
    data.forEach((d) => {
      d.date = d3.timeParse("%Y-%m-%d")(d.date);
      d.value = parseFloat(d.value);
    });

    // Scale the range of the data
    x.domain(d3.extent(data, function (d) { return d.date; }));
    y.domain([0, d3.max(data, function (d) { return d.value; })]);

    // Add the valueline path.
    svg.append("path")
      .data([data])
      .attr("class", "line")
      .attr("d", d3.line()
        .x(function (d) { return x(d.date); })
        .y(function (d) { return y(d.value); }));

    // Add the x Axis
    svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x));

    // Add the y Axis
    svg.append("g")
      .call(d3.axisLeft(y));

  });

};

plot_chart("#gpu-utilization");
plot_chart("#gpu-utilization-2");
