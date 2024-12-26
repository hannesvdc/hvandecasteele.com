const scriptElement = document.currentScript; // Capture the script element
console.log('scriptElement', scriptElement);

// Load the header HTML dynamically
document.addEventListener("DOMContentLoaded", function() {
  console.log('Header directory: ', scriptElement.src);
  if (scriptElement) {
    const scriptPath = scriptElement.src; // Get the full path of the current script
    const basePath = scriptPath.substring(0, scriptPath.lastIndexOf('/') + 1); // Extract the base directory
    const headerPath = basePath + 'header_nav.html'; // Append the header file name

    fetch(headerPath)
      .then(response => response.text())
      .then(data => {
        document.getElementById('header-container').innerHTML = data;
      })
      .catch(error => console.error('Error loading header:', error));
  } else {
    console.error("Error: Unable to determine the script path.");
  }
});