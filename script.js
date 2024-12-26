document.addEventListener("DOMContentLoaded", () => {
  const projects = [
    {
      title: "PyCont",
      link: "pycont.html",
      description: "Numerical Continuation and Bifurcation Detection Tool written in Python.",
      image: "images/pycont_resized.png"
    },
    {
      title: "Backmapping for Protein Folding",
      link: "backmapping.html",
      description: "Micro-macro Markov chain Monte Carlo sampling with backmapping for protein folding.",
      image: "images/backmapping_resized.png"
    },
    {
      title: "Locating Saddle Points on Manifolds",
      link: "saddle.html",
      description: "Locating Saddle Points on Manifolds defined by Point Clouds using Gentlest Ascent Dynamics and Gradient Extremals.",
      image: "images/saddle_resized.png"
    },
    {
      title: "Micro-Macro MCMC Sampling",
      link: "mM_MCMC.html",
      description: "Micro-Macro Markov Chain Monte Carlo Accelerated Sampling for Molecular Dynamics using Reaction Coordinates.",
      image: "images/mMFigure_png_resized.png"
    },
    {
      title: "Micro-Macro Acceleration",
      link: "micro_macro_acc.html",
      description: "Micro-macro acceleration for stochastic differential equations with a time-scale separation.",
      image: "images/micro_macro_acc_resized.png"
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