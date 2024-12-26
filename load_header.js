// Load the header HTML dynamically
document.addEventListener("DOMContentLoaded", function() {    
  const scriptPath = document.currentScript.src; // Path to load_header.js
  console.log('scriptpath', scriptPath);
  
    fetch('/header_nav.html')
      .then(response => response.text())
      .then(data => {
        document.getElementById('header-container').innerHTML = data;
      })
      .catch(error => console.error('Error loading header:', error));
});