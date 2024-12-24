document.addEventListener("DOMContentLoaded", () => {
  const projects = [
    { title: "Project 1", link: "project1.html" },
    { title: "Project 2", link: "project2.html" },
    { title: "Project 3", link: "project3.html" },
    { title: "Project 4", link: "project4.html" },
    { title: "Project 5", link: "project5.html" },
    { title: "Project 6", link: "project6.html" },
  ];

  const gridContainer = document.getElementById("projects-grid");

  projects.forEach((project) => {
    const gridItem = document.createElement("a");
    gridItem.href = project.link;
    gridItem.className = "grid-item";
    gridItem.textContent = project.title;

    gridContainer.appendChild(gridItem);
  });
});