// Load the header HTML dynamically
document.addEventListener("DOMContentLoaded", function() {
    var header_nav = document.getElementsByTagName("header_nav")
    var header_nav_src = header_nav[header_nav.length-1].src;
    console.log('header_nav:', header_nav);
    console.log('header_nav_src:', header_nav_src);
    
    fetch(header_nav_src)
      .then(response => response.text())
      .then(data => {
        document.getElementById('header-container').innerHTML = data;
      })
      .catch(error => console.error('Error loading header:', error));
});