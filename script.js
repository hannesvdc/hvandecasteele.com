document.addEventListener("DOMContentLoaded", () => {
  const projects = [
    {
      title: "Project 1",
      link: "project1.html",
      description: "This is a brief description of Project 1, explaining its goals and outcomes.",
      image: "images/project1.jpg"  // Add image URL for each project
    },
    {
      title: "Project 2",
      link: "project2.html",
      description: "This is a brief description of Project 2, explaining its key points.",
      image: "images/project2.jpg"
    },
    {
      title: "Project 3",
      link: "project3.html",
      description: "This is a brief description of Project 3, providing context and results.",
      image: "images/project3.jpg"
    },
    {
      title: "Project 4",
      link: "project4.html",
      description: "This is a brief description of Project 4, highlighting its major achievements.",
      image: "images/project4.jpg"
    },
    {
      title: "Project 5",
      link: "project5.html",
      description: "This is a brief description of Project 5, summarizing its main objectives.",
      image: "images/project5.jpg"
    },
  ];

  const gridContainer = document.getElementById("projects-grid");

  projects.forEach((project) => {
    const gridItem = document.createElement("a");
    gridItem.href = project.link;
    gridItem.className = "grid-item";

    // Create Image Element
    const imageElement = document.createElement("img");
    imageElement.src = project.image;
    imageElement.alt = project.title;
    gridItem.appendChild(imageElement);

    // Create Title
    const titleElement = document.createElement("h2");
    titleElement.textContent = project.title;
    gridItem.appendChild(titleElement);

    // Create Description
    const descriptionElement = document.createElement("p");
    descriptionElement.textContent = project.description;
    gridItem.appendChild(descriptionElement);

    gridContainer.appendChild(gridItem);
  });
});