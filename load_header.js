const scriptElement = document.currentScript;

document.addEventListener("DOMContentLoaded", function() {
  if (scriptElement) {
//      const scriptPath = scriptElement.src;
//      const basePath = scriptPath.substring(0, scriptPath.lastIndexOf('/') + 1);
      const headerPath = '/header_nav.html';

      fetch(headerPath)
          .then(response => response.text())
          .then(data => {
              document.getElementById('header-container').innerHTML = data;
          })
          .catch(error => console.error('Error loading header:', error));
  } else{
      console.error("Error: Unable to determine the script path.");
  }
});