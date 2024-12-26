// Load the header HTML dynamically
document.addEventListener("DOMContentLoaded", function() {
    var scripts = document.getElementsByTagName("script")
    src = scripts[scripts.length-1].src;
    console.log('scripts:', scripts);
    console.log('src:', src);
    
    fetch('/header_nav.html')
      .then(response => response.text())
      .then(data => {
        document.getElementById('header-container').innerHTML = data;
      })
      .catch(error => console.error('Error loading header:', error));
});