// Load the header HTML dynamically
document.addEventListener("DOMContentLoaded", function() {
    const scriptPath = document.currentScript.src; // Path to load_header.js
    const basePath = scriptPath.substring(0, scriptPath.lastIndexOf('/') + 1);
    const headerPath = basePath + 'header_nav.html';
    
    fetch(headerPath)
      .then(response => response.text())
      .then(data => {
        document.getElementById('header-container').innerHTML = data;
      })
      .catch(error => console.error('Error loading header:', error));
});