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
      title: "Micro-Macro MCMC Sampling",
      link: "mM_MCMC.html",
      description: "Micro-Macro Markov Chain Monte Carlo Accelerated Sampling for Molecular Dynamics using Reaction Coordinates.",
      image: "images/mMFigure_png.png"
    },
    {
      title: "Micro-Macro Acceleration",
      link: "micro_macro_acc.html",
      description: "Micro-macro acceleration for stochastic differential equations with a time-scale separation.",
      image: "images/micro_macro_acc.png"
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